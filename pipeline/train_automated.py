import pickle

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from tensorflow import keras
import kerasncp as kncp

import os
from typing import Iterable, Dict
import tensorflow as tf
import kerasncp as kncp
from kerasncp.tf import LTCCell, WiredCfcCell
from tensorflow import keras
import numpy as np
from matplotlib.image import imread
from tqdm import tqdm
from PIL import Image
import pandas as pd
import time
import math
from keras_models import generate_ncp_model
from train_test_loader_with_bg import get_dataset_multi, get_val_dataset_multi


def save_sequence_montage(
    dataset,
    save_path="pipeline/debug/train_montage.png",
    max_frames=16,
    ncols=8,
    dpi=200,
    title="Training Sequence (Random Background)"
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    for images, _ in dataset.take(1):
        seq = images[0].numpy()  # (seq_len, H, W, 3)

        num_frames = min(max_frames, seq.shape[0])
        ncols = min(ncols, num_frames)
        nrows = math.ceil(num_frames / ncols)

        fig_w = ncols * 2
        fig_h = nrows * 2

        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
        axes = np.array(axes).reshape(nrows, ncols)

        for idx in range(nrows * ncols):
            r, c = divmod(idx, ncols)
            ax = axes[r, c]

            if idx < num_frames:
                ax.imshow(seq[idx])
                ax.set_title(f"t={idx}", fontsize=8)
            ax.axis("off")

        plt.suptitle(title, fontsize=12, y=0.995)
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        plt.close()

        print(f"Saved montage to: {save_path}")
        break


def tlen(dataset):
    for (ix, _) in enumerate(dataset):
        pass
    return ix


# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

background_dir = "backgrounds_pngs"
training_root = "../quadrant_wise_dataset/mix_goal_heights_diff_rgba"

DROPOUT = 0.1
DEFAULT_NCP_SEED = 22222

IMAGE_SHAPE = (144, 256, 3)
IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])

batch_size = None
seq_len = 64
augmentation_params = None
single_step = False
no_norm_layer = False

# ══════════════════════════════════════════════════════════════
# TRAINING HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════

# Higher LR for background fine-tuning (10x the previous 0.0001)
# The pretrained model needs to adjust its features to become
# background-invariant too low an LR keeps it in the old minimum
lr: float = 0.001
epochs: int = 200

# Cosine decay with warm restarts: lets the optimizer escape local
# minima periodically, which helps when the input distribution has
# changed (backgrounds added)
#
# first_decay_steps = steps per cosine cycle
# Assuming ~50 batches/epoch (depends on your data), one cycle = 50 epochs
STEPS_PER_EPOCH_APPROX = 50  # adjust if you know the exact count
decay_rate: float = 0.85
lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr, decay_steps=500,
                                                            decay_rate=decay_rate, staircase=True)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

# ══════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════

gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
with strategy.scope():
    mymodel = generate_ncp_model(
        seq_len, IMAGE_SHAPE, augmentation_params, batch_size,
        DEFAULT_NCP_SEED, single_step, no_norm_layer
    )
    mymodel.compile(optimizer=optimizer, loss="mean_squared_error", metrics=['mse'])
    # mymodel.load_weights(
    #     'pipeline/model-ncp_seq-64_lr-0.000291_epoch-096_val-loss:0.0720'
    #     '_train-loss:0.0104_mse:0.0104_2022:04:15:05:11:37.hdf5'
    # )
    mymodel.summary()

# ══════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════

shift: int = 1
stride: int = 1
val_split: float = 0.1
label_scale: float = 1

with tf.device('/cpu:0'):
    # val_bg_mode="background" → validation ALSO gets random backgrounds
    # (but without color jitter) so val_loss actually measures bg robustness
    training_dataset, validation_dataset = get_dataset_multi(
        training_root, IMAGE_SHAPE, seq_len, shift, stride,
        val_split, label_scale,
        background_dir=background_dir,
        extra_data_root=None,
        val_bg_mode="background"  # KEY CHANGE: validate on backgrounds too
    )

# Larger shuffle buffer  100 was only ~1.5 batches, sequences from the
# same trajectory (same background) ended up in the same batch.
# 1000+ ensures diverse backgrounds within each batch.
SHUFFLE_BUFFER = 1000
BATCH_SIZE = 64

training_dataset = training_dataset.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)
validation_dataset = validation_dataset.batch(BATCH_SIZE)

print('load dataset shape', training_dataset.element_spec)

# ── Debug montages ──
os.makedirs("pipeline/debug", exist_ok=True)
save_sequence_montage(
    training_dataset,
    save_path="pipeline/debug/train_montage.png",
    max_frames=16, ncols=8
)
save_sequence_montage(
    validation_dataset,
    save_path="pipeline/debug/val_montage.png",
    title="Validation Sequence (Random Background, No Jitter)"
)

# ── Distribution options ──
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
training_dataset = training_dataset.with_options(options)
validation_dataset = validation_dataset.with_options(options)
training_dataset = training_dataset.prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

# ══════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════

os.makedirs("saved_models", exist_ok=True)
os.makedirs("pipeline", exist_ok=True)

run_name = f"random_bg_cosine_lr{lr}_ep{epochs}"

# 1. CSV logger
csv_logger = tf.keras.callbacks.CSVLogger(
    f'pipeline/{run_name}.csv', separator=',', append=False
)

# 2. Save best model by val_loss
checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
    filepath=f'saved_models/{run_name}_best.h5',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

# 3. Save periodic checkpoints (every 25 epochs)
checkpoint_periodic = tf.keras.callbacks.ModelCheckpoint(
    filepath='saved_models/' + run_name + '_epoch{epoch:03d}.h5',
    save_freq=STEPS_PER_EPOCH_APPROX * 25,  # approx every 25 epochs
    save_weights_only=False,
    verbose=0
)

# 4. Early stopping with generous patience (cosine restarts cause
#    val_loss spikes at restart boundaries — don't stop too early)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=40,        # generous: cosine restarts cause temp spikes
    restore_best_weights=True,
    verbose=1
)

callbacks = [csv_logger, checkpoint_best, checkpoint_periodic, early_stop]

# ══════════════════════════════════════════════════════════════
# TRAIN
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"Training config:")
print(f"  LR schedule   : CosineDecayRestarts, peak={lr}")
print(f"  Epochs         : {epochs}")
print(f"  Batch size     : {BATCH_SIZE}")
print(f"  Shuffle buffer : {SHUFFLE_BUFFER}")
print(f"  Val bg mode    : background (measures bg robustness)")
print(f"  Callbacks      : CSV, BestCheckpoint, PeriodicCkpt, EarlyStop")
print(f"{'='*60}\n")

history = mymodel.fit(
    x=training_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    verbose=1,
    use_multiprocessing=False,
    workers=1,
    max_queue_size=5,
    callbacks=callbacks,
)
print(history)

# ══════════════════════════════════════════════════════════════
# SAVE FINAL MODEL
# ══════════════════════════════════════════════════════════════

train_loss = history.history['loss'][-1]
val_loss = history.history['val_loss'][-1]
best_val_loss = min(history.history['val_loss'])
best_epoch = history.history['val_loss'].index(best_val_loss)

mymodel.save(
    f'saved_models/{run_name}_final_trainloss{train_loss:.5f}'
    f'_valloss{val_loss:.5f}.h5'
)

# ══════════════════════════════════════════════════════════════
# EVALUATE
# ══════════════════════════════════════════════════════════════

train_accuracy = mymodel.evaluate(x=training_dataset, verbose=1)
val_accuracy = mymodel.evaluate(x=validation_dataset, verbose=1)

print(f"\n{'='*60}")
print(f"Final Training Loss: {train_accuracy}")
print(f"Final Val Loss:      {val_accuracy}")
print(f"Best Val Loss:       {best_val_loss:.6f} (epoch {best_epoch})")
print(f"{'='*60}")

# ══════════════════════════════════════════════════════════════
# PLOT TRAINING CURVES
# ══════════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss curves
ax1.plot(history.history['loss'], label='Train Loss', linewidth=1.5)
ax1.plot(history.history['val_loss'], label='Val Loss', linewidth=1.5)
ax1.axhline(y=best_val_loss, color='green', linestyle='--', alpha=0.5,
            label=f'Best Val: {best_val_loss:.5f} (ep {best_epoch})')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('MSE Loss')
ax1.set_title('Training & Validation Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Zoomed view of later epochs
start_zoom = max(0, len(history.history['loss']) - 100)
ax2.plot(range(start_zoom, len(history.history['loss'])),
         history.history['loss'][start_zoom:], label='Train', linewidth=1.5)
ax2.plot(range(start_zoom, len(history.history['val_loss'])),
         history.history['val_loss'][start_zoom:], label='Val', linewidth=1.5)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MSE Loss')
ax2.set_title(f'Loss (last {len(history.history["loss"]) - start_zoom} epochs)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'pipeline/{run_name}_curves.png', dpi=150)
plt.close()
print(f"Saved training curves to pipeline/{run_name}_curves.png")