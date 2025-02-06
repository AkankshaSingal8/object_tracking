import os
from typing import Iterable, Dict

import kerasncp as kncp
from kerasncp.tf import LTCCell, WiredCfcCell
from tensorflow import keras
import tensorflow as tf
# from node_cell import *
# from tf_cfc import CfcCell, MixedCfcCell
# from tf_cfc import LTCCell as CFCLTCCell

IMAGE_SHAPE = (144, 256, 3)
IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])

DROPOUT = 0.1
DEFAULT_CFC_CONFIG = {
    "clipnorm": 1,
    "backbone_activation": "silu",
    "backbone_dr": 0.1,
    "forget_bias": 1.6,
    "backbone_units": 128,
    "backbone_layers": 1,
    "weight_decay": 1e-06
}
DEFAULT_NCP_SEED = 22222
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_ncp_model(seq_len,
                       image_shape,
                       augmentation_params=None,
                       batch_size=None,
                       seed=DEFAULT_NCP_SEED,
                       single_step: bool = False,
                       no_norm_layer: bool = False,
                       ):
    
    inputs_image, trunk_out_image = generate_network_trunk(
        seq_len,
        image_shape,
        augmentation_params=augmentation_params,
        batch_size=batch_size,
        single_step=single_step,
        no_norm_layer=no_norm_layer,
    )

    goal_images, trunk_out_goal = generate_network_trunk(
        seq_len,
        image_shape,
        augmentation_params=augmentation_params,
        batch_size=batch_size,
        single_step=single_step,
        no_norm_layer=no_norm_layer,
    )
    
    diff = keras.layers.Subtract()([trunk_out_image, trunk_out_goal])

    # Setup the network
    wiring = kncp.wirings.NCP(
        inter_neurons=18,  # Number of inter neurons
        command_neurons=12,  # Number of command neurons
        motor_neurons=4,  # Number of motor neurons
        sensory_fanout=6,  # How many outgoing synapses has each sensory neuron
        inter_fanout=4,  # How many outgoing synapses has each inter neuron
        recurrent_command_synapses=4,  # Now many recurrent synapses are in the
        # command neuron layer
        motor_fanin=6,  # How many incoming syanpses has each motor neuron,
        seed=seed,  # random seed to generate connections between nodes
    )

    rnn_cell = LTCCell(wiring)

    # x = keras.layers.RNN(rnn_cell,
    #                          batch_input_shape=(batch_size,
    #                                             seq_len,
    #                                             x.shape[-1]),
    #                          return_sequences=True)(x)

    # ncp_model = keras.Model([inputs_image], [x])

    rnn_out = keras.layers.RNN(
        rnn_cell,
        batch_input_shape=(batch_size, seq_len, diff.shape[-1]),
        return_sequences=True
    )(diff)

    ncp_model = keras.Model(
        inputs=[inputs_image, goal_images],
        outputs=[rnn_out]
    )

    return ncp_model



def generate_augmentation_layers(x, augmentation_params: Dict, single_step: bool):
    # translate -> rotate -> zoom -> noise
    trans = augmentation_params.get('translation', None)
    rot = augmentation_params.get('rotation', None)
    zoom = augmentation_params.get('zoom', None)
    noise = augmentation_params.get('noise', None)

    if trans is not None:
        x = wrap_time(keras.layers.experimental.preprocessing.RandomTranslation(
            height_factor=trans, width_factor=trans), single_step)(x)

    if rot is not None:
        x = wrap_time(keras.layers.experimental.preprocessing.RandomRotation(rot), single_step)(x)

    if zoom is not None:
        x = wrap_time(keras.layers.experimental.preprocessing.RandomZoom(
            height_factor=zoom, width_factor=zoom), single_step)(x)

    if noise:
        x = wrap_time(keras.layers.GaussianNoise(stddev=noise), single_step)(x)

    return x


def generate_normalization_layers(x, single_step: bool):
    rescaling_layer = keras.layers.experimental.preprocessing.Rescaling(1. / 255)

    normalization_layer = keras.layers.experimental.preprocessing.Normalization(
        mean=[0.6042006463205742, 0.6042006463205742, 0.6042006880578502],
        variance=[0.0103, 0.0103, 0.0103])

    x = rescaling_layer(x)
    x = wrap_time(normalization_layer, single_step)(x)
    return x


def wrap_time(layer, single_step: bool):
    """
    Helper function that wraps layer in a timedistributed or not depending on the arguments of this function
    """
    if not single_step:
        return keras.layers.TimeDistributed(layer)
    else:
        return layer


def generate_network_trunk(seq_len,
                           image_shape,
                           augmentation_params: Dict = None,
                           batch_size=None,
                           single_step: bool = False,
                           no_norm_layer: bool = False, ):
    """
    Generates CNN image processing backbone used in all recurrent models. Uses Keras.Functional API

    returns input to be used in Keras.Model and x, a tensor that represents the output of the network that has shape
    (batch [None], seq_len, num_units) if single step is false and (batch [None], num_units) if single step is true.
    Input has shape (batch, h, w, c) if single step is True and (batch, seq, h, w, c) otherwise

    """

    if single_step:
        inputs = keras.Input(shape=image_shape)
    else:
        inputs = keras.Input(batch_input_shape=(batch_size, seq_len, *image_shape))

    x = inputs

    if not no_norm_layer:
        x = generate_normalization_layers(x, single_step)

    if augmentation_params is not None:
        x = generate_augmentation_layers(x, augmentation_params, single_step)

    # Conv Layers
    x = wrap_time(keras.layers.Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu'), single_step)(
        x)
    x = wrap_time(keras.layers.Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='relu'), single_step)(
        x)
    x = wrap_time(keras.layers.Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='relu'), single_step)(
        x)
    x = wrap_time(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu'), single_step)(
        x)
    x = wrap_time(keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation='relu'), single_step)(
        x)

    # fully connected layers
    x = wrap_time(keras.layers.Flatten(), single_step)(x)
    x = wrap_time(keras.layers.Dense(units=128, activation='linear'), single_step)(x)
    x = wrap_time(keras.layers.Dropout(rate=DROPOUT), single_step)(x)

    return inputs, x
