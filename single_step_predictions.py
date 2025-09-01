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
mymodel = generate_ncp_model(seq_len, IMAGE_SHAPE, augmentation_params, batch_size, DEFAULT_NCP_SEED, single_step, no_norm_layer)

mymodel.load_weights('model_fine_tuned.h5')
hiddens = generate_hidden_list(model= mymodel, return_numpy=True)

print(hiddens)

print("Model loaded")

for directory in range(1, 13):
    root = f"./original_dataset/{directory}"
    image_paths = [path for path in sorted(os.listdir(root)) if 'png' in path]
    # print(image_paths)
    file_ending = 'png'

    predictions = []

    times = []
    for i in range(len(image_paths)):
        current_image_path = os.path.join(root, image_paths[i])
        current_img = Image.open(current_image_path).resize(IMAGE_SHAPE_CV)
        current_img_array = np.array(current_img)
        im_network = np.expand_dims(current_img_array, 0)
        hiddens = np.expand_dims(hiddens, 0)
        print("hidden shape:", hiddens.shape)
        print("im network shape", im_network.shape)
        start = time.time()
        out = mymodel.predict([im_network, *hiddens])
        end= time.time()
        times.append(end - start)
        print("Time taken: ", end - start)
        vel_cmd = out[0]  # shape: 1 x 4
        hiddens = out[1:] 
        predictions.append(vel_cmd)
    
    print("Avg Time: ", sum(times) / len(times))
    print(len(predictions))
    predictions_df = pd.DataFrame(predictions)