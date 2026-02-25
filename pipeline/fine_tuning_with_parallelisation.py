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

background_dir = "backgrounds_pngs"
training_root = "../quadrant_wise_dataset/entire_coreset_rgba"

DROPOUT = 0.1

DEFAULT_NCP_SEED = 22222

IMAGE_SHAPE = (144, 256, 3)
IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])

batch_size = None
seq_len = 64
augmentation_params = None
single_step = False
no_norm_layer = False

decay_rate: float = 0.85
lr: float = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr, decay_steps=500,
                                                            decay_rate=decay_rate, staircase=True)
#Adam optimizer
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
with strategy.scope():
    mymodel = generate_ncp_model(seq_len, IMAGE_SHAPE, augmentation_params, batch_size, DEFAULT_NCP_SEED, single_step, no_norm_layer)
    mymodel.compile(optimizer=optimizer, loss="mean_squared_error", metrics=['mse'])
    mymodel.load_weights('saved_models/coreset_random_bg_nopretrainedwt_wscheduler0.85_seed22222_lr0.0001_trainloss0.00454_epoch100.h5')
    mymodel.summary()

shift: int = 1
stride: int = 1
decay_rate: float = 0.85
val_split: float = 0.1
label_scale: float = 1
seq_len = 64
val_split: float = 0.1
label_scale: float = 1

with tf.device('/cpu:0'):
    training_dataset, validation_dataset = get_dataset_multi(training_root, IMAGE_SHAPE, seq_len, shift, stride, val_split, label_scale, background_dir=background_dir, extra_data_root=None, val_bg_mode="background")
    # val_data = get_val_dataset_multi(val_root, IMAGE_SHAPE, seq_len, shift, stride, val_split, label_scale, extra_data_root=None)

training_dataset = training_dataset.shuffle(1000).batch(64)
validation_dataset = validation_dataset.batch(64)
# print('\n\nTraining Dataset Size: %d\n\n' % tlen(dataset))

print('load dataset shape', training_dataset.element_spec)

def save_sequence(dataset, save_dir="debug_seq"):
    os.makedirs(save_dir, exist_ok=True)

    for images, _ in dataset.take(1):
        seq = images[0]
        for i in range(seq.shape[0]):
            plt.imsave(f"{save_dir}/frame_{i:03d}.png", seq[i].numpy())
        print(f"Saved sequence to {save_dir}")
        break

save_sequence(training_dataset, save_dir="v2_coreset_debug_seq_train")
save_sequence(validation_dataset, save_dir="v2_coreset_debug_seq_val")

save_sequence_montage(
    training_dataset,
    save_path="pipeline/debug/v2_coreset_train_montage.png",
    max_frames=16,
    ncols=8
)

# Optional validation montage
save_sequence_montage(
    validation_dataset,
    save_path="pipeline/debug/v2_coreset_val_montage.png",
    title="Validation Sequence (Black Background)"
)

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
training_dataset = training_dataset.with_options(options)
validation_dataset = validation_dataset.with_options(options)
# Have GPU prefetch next training batch while first one runs
training_dataset = training_dataset.prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)


epochs: int = 500
csv_logger = tf.keras.callbacks.CSVLogger(f'pipeline/v2_coreset_random_bg_wscheduler{decay_rate}_seed22222_lr{lr}_epoch{epochs}.csv', separator=',', append=False)
callbacks = None
#setting validation data to None
history = mymodel.fit(x=training_dataset, validation_data=validation_dataset, epochs=epochs,verbose=1, use_multiprocessing=False, workers=1, max_queue_size=5, callbacks=[csv_logger],)
print(history)

# # Extract the final training and validation loss
train_loss = history.history['loss'][-1]
mymodel.save(f'saved_models/v2_coreset_random_bg_nopretrainedwt_wscheduler{decay_rate}_seed22222_lr{lr}_trainloss{train_loss:.5f}_epoch{epochs}.h5')
# val_loss = history.history['val_loss'][-1]



train_accuracy = mymodel.evaluate(x=training_dataset, verbose=1)
val_accuracy = mymodel.evaluate(x=validation_dataset, verbose=1)

print('Final Training Accuracy:', train_accuracy)
print('Final Val Accuracy:', val_accuracy)


