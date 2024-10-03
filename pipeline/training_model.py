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
from train_test_loader import get_dataset_multi, get_val_dataset_multi


def tlen(dataset):
    for (ix, _) in enumerate(dataset):
        pass
    return ix

def train_model():

    training_root = "../fly_to_target_dataset/coreset"
    val_root = "../fly_to_target_dataset/test_data"
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
    lr: float = 0.001
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr, decay_steps=500,
                                                                decay_rate=decay_rate, staircase=True)
    #Adam optimizer
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():
        mymodel = generate_ncp_model(seq_len, IMAGE_SHAPE, augmentation_params, batch_size, DEFAULT_NCP_SEED, single_step, no_norm_layer)
        mymodel.compile(optimizer=optimizer, loss="mean_squared_error", metrics=['mse'])
        mymodel.load_weights('saved_models/fine_tuned_woscheduler_seed22222_lr0.0001_150traj.h5')

        mymodel.summary()

    shift: int = 1
    stride: int = 1
    decay_rate: float = 0.95
    val_split: float = 0.2
    label_scale: float = 1
    seq_len = 64
    val_split: float = 0.1
    label_scale: float = 1

    with tf.device('/cpu:0'):
        training_dataset = get_dataset_multi(training_root, IMAGE_SHAPE, seq_len, shift, stride, val_split, label_scale, extra_data_root=None)
        val_data = get_val_dataset_multi(val_root, IMAGE_SHAPE, seq_len, shift, stride, val_split, label_scale, extra_data_root=None)

    print('\n\nTraining Dataset Size: %d\n\n' % tlen(training_dataset))

    print('load dataset shape', training_dataset.element_spec)
    training_dataset = training_dataset.shuffle(100).batch(64)

    val_dataset = val_data.batch(64)
    print('load val dataset shape', val_dataset.element_spec)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    training_dataset = training_dataset.with_options(options)
    validation_dataset = val_dataset.with_options(options)
    # Have GPU prefetch next training batch while first one runs
    training_dataset = training_dataset.prefetch(tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)


    epochs: int = 100
    callbacks = None
    #setting validation data to None
    history = mymodel.fit(x=training_dataset, validation_data=val_dataset, epochs=epochs,verbose=1, use_multiprocessing=False, workers=1, max_queue_size=5)
    print(history)

    # Extract the final training and validation loss
    train_loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]


    accuracy = mymodel.evaluate(x=training_dataset)
    print('Accuracy:' ,accuracy)

    mymodel.save(f'saved_models/retrain_150traj_wscheduler0.85_seed22222_lr0.001_trainloss{train_loss:.5f}_valloss{val_loss:.5f}_coreset900.h5')
