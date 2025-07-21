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
from keras_models import generate_ncp_model, generate_lstm_model, generate_ctrnn_model
from train_test_loader import get_dataset_multi, get_val_dataset_multi

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def tlen(dataset):
    for (ix, _) in enumerate(dataset):
        pass
    return ix

training_root = "../quadrant_wise_dataset/mix_goal_heights_diff"
# val_root = "../fly_to_target_dataset/test_data"
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
rnn_sizes = [252]
lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr, decay_steps=500,
                                                            decay_rate=decay_rate, staircase=True)
#Adam optimizer
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)

with strategy.scope():
    mymodel = generate_ctrnn_model(rnn_sizes, seq_len, IMAGE_SHAPE, ct_network_type='ctrnn')
    mymodel.compile(optimizer=optimizer, loss="mean_squared_error", metrics=['mse'])
    mymodel.load_weights('./baseline_comparisons/ctrnn_mix_goal_0.85_seed22222_lr0.0001_trainloss0.01189_epoch1.h5')

    mymodel.summary()