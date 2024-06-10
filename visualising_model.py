import pickle
import visualkeras

import argparse
import copy
import json
import os.path
from enum import Enum
from typing import Dict, Tuple, Union, Optional, Any
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
from typing import List, Iterable, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
from numpy import ndarray
from tensorflow import keras, Tensor
from tensorflow.keras.layers import Conv2D
from tensorflow.python.keras.models import Functional
from keras_models import generate_ncp_model
# from vis_utils import run_visualization, write_video

IMAGE_SHAPE = (144, 256, 3)
IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])

DEFAULT_NCP_SEED = 22222

batch_size = None
seq_len = 64
augmentation_params = None
single_step = True
no_norm_layer = False
mymodel = generate_ncp_model(seq_len, IMAGE_SHAPE, augmentation_params, batch_size, DEFAULT_NCP_SEED, single_step, no_norm_layer)

# pretrained model weights
# mymodel.load_weights('model-ncp-val.hdf5')

# custom model weights
mymodel.load_weights('ncp_model_b64_seq64_lr0.001.h5')

conv_layers = [layer for layer in mymodel.layers if isinstance(layer, Conv2D)]

act_model_inputs = mymodel.inputs[0]  # don't want to take in hidden state, just image
vis_model = keras.models.Model(inputs=act_model_inputs,
                                        outputs=[layer.output for layer in conv_layers])

print(vis_model.summary())
# vis_model.add(visualkeras.SpacingDummyLayer(spacing=100))
visualkeras.layered_view(vis_model,legend=True, spacing = 40, to_file='conv_vis_volume.png')
