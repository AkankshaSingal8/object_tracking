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
from keras_models import generate_ncp_model, generate_lstm_model, generate_ctrnn_model
import os
import csv
from PIL import Image

DEFAULT_NCP_SEED = 22222
IMAGE_SHAPE = (144, 256, 3)
IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])

# CTRNNs
batch_size = None
seq_len = 64
augmentation_params = None
no_norm_layer = False
single_step = True
rnn_sizes = [252]
model = generate_ctrnn_model(rnn_sizes, seq_len, IMAGE_SHAPE, ct_network_type='ctrnn', single_step = True)

root = '/src/drone_causality/object_tracking/baseline_comparisons/ctrnn_mix_goal_0.85_seed22222_lr0.0001_trainloss0.00110_epoch100.h5'

model.load_weights(root)

#LSTMs
# batch_size = None
# seq_len = 64
# augmentation_params = None
# no_norm_layer = False
# single_step = True
# rnn_sizes = [193]
# model = generate_lstm_model(rnn_sizes, seq_len, IMAGE_SHAPE, single_step = True)

# root = '/src/drone_causality/object_tracking/baseline_comparisons/lstm_mix_goal_0.85_seed22222_lr0.0001_trainloss0.00103_epoch100.h5'

# model.load_weights(root)

# image_path = "../quadrant_wise_dataset/mix_goal_heights_diff/1/Image1.png"

def generate_hidden_list(model: Functional, return_numpy: bool = True):
    """
    Generates a list of tensors that are used as the hidden state for the argument model when it is used in single-step
    mode. The batch dimension (0th dimension) is assumed to be 1 and any other dimensions (seq len dimensions) are
    assumed to be 0

    :param return_numpy: Whether to return output as numpy array. If false, returns as keras tensor
    :param model: Single step functional model to infer hidden states for
    :return: list of hidden states with 0 as value
    """
    constructor = np.zeros if return_numpy else tf.zeros
    hiddens = []
    print("Length of model input shape: ", len(model.input_shape))
    if len(model.input_shape)==1:
        lool = model.input_shape[0][1:]
    else:
        print("model input shape: ", model.input_shape)
        lool = model.input_shape[1:]
    print("lool: ", lool)
    for input_shape in lool:  # ignore 1st output, as is this control output
        hidden = []
        for i, shape in enumerate(input_shape):
            if shape is None:
                if i == 0:  # batch dim
                    hidden.append(1)
                    continue
                elif i == 1:  # seq len dim
                    hidden.append(0)
                    continue
                else:
                    print("Unable to infer hidden state shape. Leaving as none")
            hidden.append(shape)
        hiddens.append(constructor(hidden))
    return hiddens


hiddens = generate_hidden_list(model= model, return_numpy=True)

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
    output = model.predict([img, *hiddens])
    end = time.time()

    print(f"Inference time: {end - start:.4f} seconds")
    time_list.append(end - start)

print(f"Average inference time over 100 runs: {np.mean(time_list[1:]):.4f} seconds")