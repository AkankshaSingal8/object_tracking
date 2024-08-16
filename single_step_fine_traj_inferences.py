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
    if len(model.input_shape)==1:
        lool = model.input_shape[0][1:]
    else:
        # UPDATED CODE HERE input_shape[2:] -> input_shape[1:]
        lool = model.input_shape[1:]

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

with tf.device('/cpu:0'):
    single_step_model.load_weights('saved_models/fine_tuned_wscheduler_lr0.01_new_data.h5')
print("Model loaded")

hiddens = generate_hidden_list(model= single_step_model, return_numpy=True)
print("hiddens shape: ", hiddens[0].shape)

train_root = "../fly_to_target_dataset/original_dataset"
test_root = "../fly_to_target_dataset/test_data"
output_directory = "SINGLE_STEP_FINE_TUNED_NEW_DATA"
file_ending = 'png'

train_inference_time = []
test_inference_time = []

for directory in range(len(os.listdir(train_root))):
    predictions = []
    times = []
    directory_path = f"{train_root}/{directory + 1}"
    print("Processing directory : ", directory_path)
    n_images = [path for path in os.listdir(directory_path) if file_ending in path]
    print("Number of images :", len(n_images))
    for i in range(len(n_images)):
        current_image_path = f"{directory_path}/Image{i + 1}.png"
        current_img = Image.open(current_image_path).resize(IMAGE_SHAPE_CV)
        current_img_array = np.array(current_img)
        im_network = np.expand_dims(current_img_array, 0)
        start = time.time()
        output = single_step_model.predict([im_network, *hiddens])
        end= time.time()
        times.append(end - start)
        # print(end - start)
        predictions.append(list(output[0][0].tolist()))
        hiddens = output[1:]
    print("Avg Time: ", sum(times) / len(times))
    train_inference_time.append(sum(times) / len(times))
    # print(len(predictions))
    # print(predictions)
    predictions_df = pd.DataFrame(predictions)


    # Save the DataFrame to a CSV file
    predictions_df.to_csv(f"{output_directory}/predictions_train_data{directory + 1}.csv", index=False)
    print("Predictions saved to file: ", directory + 1)

for directory in range(len(os.listdir(test_root))):
    predictions = []
    times = []
    directory_path = f"{test_root}/{directory + 1}"
    print("Processing directory : ", directory_path)
    n_images = [path for path in os.listdir(directory_path) if file_ending in path]
    print("Number of images :", len(n_images))
    for i in range(len(n_images)):
        current_image_path = f"{directory_path}/Image{i + 1}.png"
        current_img = Image.open(current_image_path).resize(IMAGE_SHAPE_CV)
        current_img_array = np.array(current_img)
        im_network = np.expand_dims(current_img_array, 0)
        start = time.time()
        output = single_step_model.predict([im_network, *hiddens])
        end= time.time()
        times.append(end - start)
        # print(end - start)
        predictions.append(list(output[0][0].tolist()))
        hiddens = output[1:]
    print("Avg Time: ", sum(times) / len(times))
    test_inference_time.append(sum(times) / len(times))
    # print(len(predictions))
    # print(predictions)
    predictions_df = pd.DataFrame(predictions)


    # Save the DataFrame to a CSV file
    predictions_df.to_csv(f"{output_directory}/predictions_test_data{directory + 1}.csv", index=False)
    print("Predictions saved to file: ", directory + 1)

for directory in range(len(os.listdir(test_root))):
    print("Processing directory : ", directory + 1)
    csv_file_name = f'{test_root}/{directory + 1}/data_out.csv'
    labels = np.genfromtxt(csv_file_name, delimiter=',', skip_header=1, dtype=np.float32)

    preds_file_name = f'{output_directory}/predictions_test_data{directory + 1}.csv'
    # pred_vals = np.genfromtxt(preds_file_name, delimiter=',', skip_header=1, dtype=np.float32)[:len(labels)]
    pred_vals = pd.read_csv(preds_file_name)
    print(len(labels), len(pred_vals))
    
    error = labels - pred_vals
    
    error_df = pd.DataFrame(error)

    # # Save the DataFrame to a CSV file
    error_df.to_csv(f'{output_directory}/errors_test_data{directory + 1}.csv', index=False)
    print("Errors saved to file: ", directory + 1)

for directory in range(len(os.listdir(train_root))):
    csv_file_name = f'{train_root}/{directory + 1}/data_out.csv'
    labels = np.genfromtxt(csv_file_name, delimiter=',', skip_header=1, dtype=np.float32)

    preds_file_name = f'{output_directory}/predictions_train_data{directory + 1}.csv'
    pred_vals = np.genfromtxt(preds_file_name, delimiter=',', skip_header=1, dtype=np.float32)[:len(labels)]

    print(len(labels), len(pred_vals))
    error = labels - pred_vals
    
    error_df = pd.DataFrame(error)

    # # Save the DataFrame to a CSV file
    error_df.to_csv(f'{output_directory}/errors_train_data{directory + 1}.csv', index=False)
    print("Errors saved to file: ", directory + 1)