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
from math import ceil
from typing import Optional, Sequence, Union

import tensorflow as tf
from numpy import ndarray
from tensorflow import Tensor
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.models import Functional
import numpy as np

import cv2


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

def compute_gradcam(img: Union[Tensor, ndarray], grad_model: Functional, hiddens: Sequence[Tensor],
                    pred_index: Optional[Sequence[Tensor]] = None):
    heatmaps, hiddens = _compute_gradcam(img=img, grad_model=grad_model, hiddens=hiddens, pred_index=pred_index)
    avg_heat = tf.math.add_n(heatmaps)

    map_min = np.min(avg_heat)
    map_max = np.max(avg_heat)
    avg_heat = (avg_heat - map_min) / (map_max - map_min + 1e-6)

    avg_heat = tf.expand_dims(avg_heat, axis=-1)

    return avg_heat, hiddens, avg_heat


# def compute_gradcam_tile(img: Union[Tensor, ndarray], grad_model: Functional, hiddens: Sequence[Tensor],
#                          pred_index: Optional[Sequence[Tensor]] = None):
#     heatmaps, hiddens = _compute_gradcam(img=img, grad_model=grad_model, hiddens=hiddens, pred_index=pred_index)
#     num_rows = ceil(len(heatmaps) / 2)
#     return image_grid(imgs=heatmaps, rows=num_rows, cols=2), hiddens


def _compute_gradcam(img: Union[Tensor, ndarray], grad_model: Functional, hiddens: Sequence[Tensor],
                     pred_index: Optional[Sequence[Tensor]] = None):
    
    if pred_index is None:
        pred_index = range(grad_model.output_shape[1][-1])

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        out = grad_model([img, *hiddens])
        last_conv_layer_output = out[0]
        preds = out[1]
        hiddens = out[2:]

    heatmaps = []
    # for each element of preds, compute gradient of last_conv_out wrt this element of pred, abs and sum these gradients
    # strip batch dim
    # jacobian shape 4x last_conv_layer_output.shape where each element is gradient, preds[:,i] wrt last_conv_layer_out
    grads = tape.jacobian(preds, last_conv_layer_output)[0]
    last_conv_layer_output = last_conv_layer_output[0]
    for pred in pred_index:
        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grad = grads[pred]

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grad, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # patrick edit: absolute value heatmaps to not discount/cancel negative and positive contributions
        heatmap = tf.math.abs(heatmap)

        heatmaps.append(heatmap)

    return heatmaps, hiddens


def display_gradcam(heatmap):
    """
    Display the Grad-CAM heatmap using Matplotlib.

    Args:
    - heatmap: Grad-CAM heatmap (should be 2D).
    """

    # Convert Tensor to NumPy array if needed
    if isinstance(heatmap, tf.Tensor):
        heatmap = heatmap.numpy().squeeze()  # Remove batch dimension

    # Ensure the heatmap has valid values
    heatmap_min, heatmap_max = np.min(heatmap), np.max(heatmap)
    print(f"Heatmap Min: {heatmap_min}, Heatmap Max: {heatmap_max}")

    # Normalize heatmap to range [0, 255]
    heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min + 1e-6)
    heatmap = np.uint8(255 * heatmap)

    # Resize heatmap to ensure visibility
    heatmap = cv2.resize(heatmap, (256, 144))  # Resize to default shape

    # Display heatmap using Matplotlib
    plt.figure(figsize=(6, 4))
    plt.imshow(heatmap, cmap="jet")
    plt.colorbar()
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")
    plt.show()



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
# mymodel.load_weights('./saved_models/retrain_mix_goal_heights_diff_coreset_wscheduler0.85_seed22222_lr0.001_trainloss0.00008_epoch100.h5')
mymodel.load_weights("./saved_models/retrain_difftraj_wscheduler0.85_seed22222_lr0.001_trainloss0.00016_valloss0.13141_diffcoreset900.h5")
conv_layers = [layer for layer in mymodel.layers if isinstance(layer, Conv2D)]
vis_model = tf.keras.models.Model(
        inputs=[mymodel.inputs],
        outputs=[conv_layers[-1].output, *mymodel.output]
    )
print(vis_model.summary())

def load_image(image_path):
    img = Image.open(image_path)
    img = img.resize(IMAGE_SHAPE_CV)
    img_array = np.array(img, dtype=np.float32)  # Convert to float32
    img_array = img_array / 255.0  # Normalize pixel values (0-1)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)  # Convert to Tensor
    return img_array


img = load_image('../fly_to_target_dataset/diff_dataset/1/Image100.png')
vis_hiddens = generate_hidden_list(vis_model, False)
saliency, vis_hiddens, sample_extra = compute_gradcam(img, vis_model, vis_hiddens)

# Display Grad-CAM visualization
display_gradcam(saliency)
# cv2.imshow(saliency)
