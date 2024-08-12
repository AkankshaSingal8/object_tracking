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
from numpy import ndarray

import copy
import json
from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict, Optional, List, Iterable, Union

from tensorflow.python.keras.layers import Conv2D, Dense
from tensorflow.python.keras.models import Functional

from keras_models import generate_ncp_model

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


IMAGE_SHAPE = (144, 256, 3)
IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])

DEFAULT_NCP_SEED = 22222

batch_size = None
seq_len = 64
augmentation_params = None
no_norm_layer = False
single_step = True
single_step_model = generate_ncp_model(seq_len, IMAGE_SHAPE, augmentation_params, batch_size, DEFAULT_NCP_SEED, single_step, no_norm_layer)
single_step_model.load_weights('model_fine_tuned.h5')


def generate_dummy_image() -> ndarray:
    return np.random.rand(1, *IMAGE_SHAPE)

dummy_image = generate_dummy_image()

hiddens = generate_hidden_list(model= single_step_model, return_numpy=True)
print("hiddens shape: ", hiddens[0].shape)

current_image_path = "000000.png"
current_img = Image.open(current_image_path).resize(IMAGE_SHAPE_CV)
current_img_array = np.array(current_img)
im_network = np.expand_dims(current_img_array, 0)

# print("hidden shape", hiddens.shape())
# print(single_step_model.summary())
# print("INput shape :", single_step_model.input_shape)

# hidden_shape = (1, 34)  # Assuming batch size 1 for dummy prediction
# hiddens = [np.zeros(hidden_shape) for _ in range(len(single_step_model.input) - 1)]

# print("hiddens shape: ", hiddens[0].shape)
# print("image shape: ", dummy_image.shape)
print(single_step_model.output_shape)
output = single_step_model.predict([im_network, *hiddens])
print('output 0 shape', output[0][0])
print('output 1 shape', output[1].shape)
# vx,vy,vz, omega_z = output[0][0][0], output[0][0][1], output[0][0][2], output[0][0][3]
l = [[0,0,0,0]]
l.append(output[0][0].tolist())
df = pd.DataFrame(l)
df.to_csv("single_step_trial.csv", index=False)
# for lst in output:
#     print("lst: ", lst)
#     print(lst.shape)
