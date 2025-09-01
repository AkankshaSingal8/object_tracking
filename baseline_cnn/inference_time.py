import cv2
import time
from typing import Iterable, Dict
import tensorflow as tf
import kerasncp as kncp
from kerasncp.tf import LTCCell, WiredCfcCell
from tensorflow.python.keras.models import Functional
from tensorflow import keras
import numpy as np
from matplotlib.image import imread
# from keras_models import generate_ncp_model
from keras_models import generate_ncp_model, generate_cnn_model
import os
import csv
from PIL import Image

DEFAULT_NCP_SEED = 22222
IMAGE_SHAPE = (144, 256, 3)
IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])

batch_size = None
seq_len = 64
augmentation_params = None
no_norm_layer = False
single_step = True
model = generate_cnn_model(seq_len, IMAGE_SHAPE, augmentation_params, batch_size, single_step, no_norm_layer)

root = '/src/drone_causality/object_tracking/baseline_cnn/cnn_mix_goal_0.85_seed22222_lr0.0001_trainloss0.00081_epoch100.h5'

model.load_weights(root)

# image_path = "../quadrant_wise_dataset/mix_goal_heights_diff/1/Image1.png"


def load_image(image_path):
    img = Image.open(image_path)
    img = img.resize(IMAGE_SHAPE_CV)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.convert_to_tensor(img_array)
    return img_array

time_list = []
for i in range(1, 100):
    image_path = f"../quadrant_wise_dataset/mix_goal_heights_diff/1/Image{i}.png"
    img = load_image(image_path)


    start = time.time()
    output = model.predict(img)
    end = time.time()

    print(f"Inference time: {end - start:.4f} seconds")
    time_list.append(end - start)

print(f"Average inference time over 100 runs: {np.mean(time_list[1:]):.4f} seconds")