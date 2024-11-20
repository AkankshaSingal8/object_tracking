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
from keras_models import generate_ncp_model

training_root = "../fly_to_target_dataset/diff_coreset"
val_root = "../fly_to_target_dataset/test_data"
DROPOUT = 0.1

DEFAULT_NCP_SEED = 22222

IMAGE_SHAPE = (144, 256, 3)
IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])

# goal_image_pth = "goal_image.png"
# goal_image = imread(goal_image_pth)
# goal_image = np.array(goal_image)
# goal_image = goal_image / 255.0

def get_output_normalization(root):
    training_output_mean_fn = os.path.join(root, 'stats', 'training_output_means.csv')
    if os.path.exists(training_output_mean_fn):
        print('Loading training data output means from: %s' % training_output_mean_fn)
        output_means = np.genfromtxt(training_output_mean_fn, delimiter=',')
    else:
        output_means = np.zeros(4)

    training_output_std_fn = os.path.join(root, 'stats', 'training_output_stds.csv')
    if os.path.exists(training_output_std_fn):
        print('Loading training data output std from: %s' % training_output_std_fn)
        output_stds = np.genfromtxt(training_output_std_fn, delimiter=',')
    else:
        output_stds = np.ones(4)

    return output_means, output_stds


def load_dataset_multi(root, image_size, seq_len, shift, stride, label_scale):
    file_ending = 'png'
    IMAGE_SHAPE = (144, 256, 3)
    IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])

    def sub_to_batch(sub_feature, sub_label):
        sfb = sub_feature.batch(seq_len, drop_remainder=True)
        slb = sub_label.batch(seq_len, drop_remainder=True)
        return tf.data.Dataset.zip((sfb, slb))
        # return sub.batch(seq_len, drop_remainder=True)
    
    # def apply_random_augmentations(image):
    #     # Generate a random number and apply augmentations with a 50% probability
    #     if tf.random.uniform(()) > 0.3:  # 30% chance to apply augmentations
    #         image = tf.image.convert_image_dtype(image, tf.float32)  # Convert to float32 for augmentation
    #         image = tf.image.random_brightness(image, max_delta=0.1)  # Random brightness adjustment
    #         image = tf.image.random_contrast(image, lower=0.8, upper=1.2)  # Random contrast adjustment
    #         image = tf.image.random_saturation(image, lower=0.8, upper=1.2)  # Random saturation adjustment
    #         image = tf.image.convert_image_dtype(image, tf.uint8)  # Convert back to uint8
    #     return image

    
    datasets = []

    #output_means, output_stds = get_output_normalization(root)

    
    for i in range(len(os.listdir(root))):
        directory = i + 1
        csv_file_name = f"{root}/{str(directory)}/data_out.csv"
        labels = np.genfromtxt(csv_file_name, delimiter=',', skip_header=1, dtype=np.float32)
        print("labels", labels)
        # if labels.shape[1] == 4:
        #     labels = (labels - output_means) / output_stds
        #     # labels = labels * label_scale
        # elif labels.shape[1] == 5:
        #     labels = (labels[:, 1:] - output_means) / output_stds
        #     # labels = labels[:,1:] * label_scale
        # else:
        #     raise Exception('Wrong size of input data (expected 4, got %d' % labels.shape[1])
    
        labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
        # n_images = len(os.listdir(os.path.join(root, d))) - 1
        n_images = len([fn for fn in os.listdir(f"./{root}/{str(directory)}") if file_ending in fn])
        print(n_images)
        print("no of imgs", n_images)
        # dataset_np = np.empty((n_images, 256, 256, 3), dtype=np.uint8)
        dataset_np = np.empty((n_images, *image_size), dtype=np.uint8)

        for ix in range(n_images):
            # dataset_np[ix] = imread(os.path.join(root, d, '%06d.jpeg' % ix))
            img_file_name = root + "/" + str(directory) +'/Image' + str(ix + 1) + '.'+ file_ending
            img = Image.open(img_file_name)
            img = img.resize(IMAGE_SHAPE_CV)
            # dataset_np[ix] = img[img.height - image_size[0]:, :, :]
            dataset_np[ix] = img

        images_dataset = tf.data.Dataset.from_tensor_slices(dataset_np)
        # images_dataset = images_dataset.map(apply_random_augmentations, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = tf.data.Dataset.zip((images_dataset, labels_dataset))
        dataset = dataset.window(seq_len, shift=shift, stride=stride, drop_remainder=True).flat_map(sub_to_batch)
        datasets.append(dataset)

    return datasets

def get_dataset_multi(root, image_size, seq_len, shift, stride, validation_ratio, label_scale, extra_data_root=None):
    ds = load_dataset_multi(root, image_size, seq_len, shift, stride, label_scale)
    print('n bags: %d' % len(ds))
    cnt = 0

    for d in ds:
        for (ix, _) in enumerate(d):
            pass
            cnt += ix
    print('n windows: %d' % cnt)

    val_ix = 0

    # val_ix = int(len(ds) * validation_ratio)
    # print('\nval_ix: %d\n' % val_ix)
    # validation_datasets = ds[:val_ix]

    training_datasets = ds[val_ix:]

    # if either dataset has length 0, trying to call flat map raises error that return type is wrong
    # assert len(training_datasets) > 0 and len(validation_datasets) > 0, f"Training or validation dataset has no points!" \
    #                                                                     f"Train dataset len: {len(training_datasets)}" \
    #                                                                     f"Val dataset len: {len(validation_datasets)}"
    training = tf.data.Dataset.from_tensor_slices(training_datasets).flat_map(lambda x: x)
    # validation = tf.data.Dataset.from_tensor_slices(validation_datasets).flat_map(lambda x: x)

    # return training, validation
    return training
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
img_array = np.array(img) / 255
img_arrays = np.stack([img_array] * seq_len, axis = 0)
img_arrays = np.expand_dims(img_arrays, axis = 0) 
goal_image = tf.convert_to_tensor(img_arrays, dtype=tf.float32) 

losses = []
mymodel = generate_ncp_model(seq_len, IMAGE_SHAPE, augmentation_params, batch_size, DEFAULT_NCP_SEED, single_step, no_norm_layer)
mymodel.load_weights('pipeline/saved_models/custom_loss_function_diff_dataset.h5')

lr: float = 0.0001
decay_rate: float = 0.85
# lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr, decay_steps=500,
#                                                             decay_rate=decay_rate, staircase=True)

#Adam optimizer
optimizer = keras.optimizers.Adam(learning_rate=lr)

# Custom training loop
epochs = 100

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")

    # Iterate over the training dataset
    for step, (x_batch_train, y_batch_train) in enumerate(training_dataset):
        # print(f"Step {step}, x_batch_train shape: {x_batch_train.shape}, y_batch_train shape: {y_batch_train.shape}")
        # print(x_batch_train.dtype)
        # print(x_batch_train)
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = mymodel(x_batch_train, training=True)

            # Compute the loss between predictions and targets
            loss1 = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_batch_train, y_pred))

            img_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(tf.cast(x_batch_train, dtype=tf.float32), goal_image))
            
            total_loss = loss1 + 1e-4 * img_loss
             
            
        # Compute gradients
        gradients = tape.gradient(total_loss, mymodel.trainable_weights)
        # gradient_norms = [tf.norm(g) for g in gradients if g is not None]
        # print(f"Gradient norms: {gradient_norms}")

        # Update weights
        optimizer.apply_gradients(zip(gradients, mymodel.trainable_weights))

        print(f"Step {step}, Loss: {total_loss}")
        losses.append(total_loss)
        # break


plt.figure(figsize=(8, 5))
plt.plot([losses[i] for i in range(0, len(losses), 976)], linewidth=2)
plt.plot([losses[i] for i in range(1, len(losses), 976)], linewidth=2)
plt.plot([losses[i] for i in range(2, len(losses), 976)], linewidth=2)


plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.grid()
plt.show()
plt.close()

mymodel.save(f'pipeline/saved_models/retrained_custom_loss_function_diff_coreset.h5')