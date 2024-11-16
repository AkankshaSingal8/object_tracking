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
import matplotlib.pyplot as plt
import copy
import json
from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict, Optional, List, Iterable, Union

from tensorflow.python.keras.layers import Conv2D, Dense
from tensorflow.python.keras.models import Functional

from keras_models import generate_ncp_model
from train_test_loader import get_dataset_multi, get_val_dataset_multi

training_root = "../fly_to_target_dataset/diff_coreset"
val_root = "../fly_to_target_dataset/test_data"
DROPOUT = 0.1

DEFAULT_NCP_SEED = 22222

IMAGE_SHAPE = (144, 256, 3)
IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])

shift: int = 1
stride: int = 1
# decay_rate: float = 0.95
val_split: float = 0.2
label_scale: float = 1
seq_len = 64
val_split: float = 0.1
label_scale: float = 1

training_dataset = get_dataset_multi(training_root, IMAGE_SHAPE, seq_len, shift, stride, val_split, label_scale, extra_data_root=None)

print('load dataset shape', training_dataset.element_spec)
training_dataset = training_dataset.batch(64)

batch_size = None
seq_len = 64
augmentation_params = None
single_step = False
no_norm_layer = False

image_pth = "pipeline/goal_img_diff.png"
img = Image.open(image_pth)
img = img.resize(IMAGE_SHAPE_CV)  
img_array = np.array(img) / 255.0
img_arrays = np.stack([img_array] * seq_len, axis = 0)
img_arrays = np.expand_dims(img_arrays, axis = 0) 
goal_image = tf.convert_to_tensor(img_arrays, dtype=tf.float32) 

mymodel = generate_ncp_model(seq_len, IMAGE_SHAPE, augmentation_params, batch_size, DEFAULT_NCP_SEED, single_step, no_norm_layer)
# mymodel.load_weights('saved_models/fine_tuned_woscheduler_seed22222_lr0.0001_trainloss0.00012_valloss0.08719_diff_dataset.h5')

lr: float = 0.001
decay_rate: float = 0.85
# lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr, decay_steps=500,
#                                                             decay_rate=decay_rate, staircase=True)
#Adam optimizer
optimizer = keras.optimizers.Adam(learning_rate=lr)

# Custom training loop
epochs = 100

losses = []

# tf.keras.mixed_precision.set_global_policy('mixed_float16')

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")

    # Iterate over the training dataset
    for step, (x_batch_train, y_batch_train) in enumerate(training_dataset):
        # print(f"Step {step}, x_batch_train shape: {x_batch_train.shape}, y_batch_train shape: {y_batch_train.shape}")
        
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = mymodel(x_batch_train, training=True)

            # Compute the loss between predictions and targets
            loss1 = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_batch_train, y_pred))

            squared_diff = tf.square(tf.cast(x_batch_train, tf.float32) - goal_image)
            mse_per_image = tf.reduce_mean(squared_diff, axis=[2, 3, 4])
            loss2 = tf.reduce_mean(tf.reduce_sum(mse_per_image, axis=1))            

            # Combine the losses 
            total_loss = loss1 + 1e-4 * loss2 
            
        # Compute gradients
        gradients = tape.gradient(total_loss, mymodel.trainable_weights)
        gradient_norms = [tf.norm(g) for g in gradients if g is not None]
        print(f"Gradient norms: {gradient_norms}")
        # Update weights
        optimizer.apply_gradients(zip(gradients, mymodel.trainable_weights))

        print(f"Step {step}, Loss: {total_loss}")
        losses.append(total_loss)

# mymodel.save(f'saved_models/custom_loss_retrained_woscheduler_seed{DEFAULT_NCP_SEED}_lr{lr}_trainloss{losses[-1]:.5f}_diffcoreset900.h5')

plt.figure(figsize=(8, 5))
plt.plot(losses, linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.grid()
plt.show()
plt.close()

