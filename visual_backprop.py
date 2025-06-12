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
mymodel.load_weights('./saved_models/retrain_mix_goal_heights_diff_coreset_wscheduler0.85_seed22222_lr0.001_trainloss0.00008_epoch100.h5')

# mymodel.load_weights("./saved_models/retrain_difftraj_wscheduler0.85_seed22222_lr0.001_trainloss0.00016_valloss0.13141_diffcoreset900.h5")
conv_layers = [layer for layer in mymodel.layers if isinstance(layer, Conv2D)]

act_model_inputs = mymodel.inputs[0]  # don't want to take in hidden state, just image
vis_model = keras.models.Model(inputs=act_model_inputs,
                                        outputs=[layer.output for layer in conv_layers])

print(vis_model.summary())

def compute_visualbackprop(img: Union[Tensor, ndarray],
                           activation_model: Functional,
                           hiddens: Optional[List[Tensor]] = None,
                           kernels: Optional[List[Iterable]] = None,
                           strides: Optional[List[Iterable]] = None):
    
    # infer CNN kernels, strides, from layers
    if not (kernels and strides):
        kernels, strides = [], []
        # don't infer form initial input layer so start at 1
        for layer in activation_model.layers[1:]:
            if isinstance(layer, Conv2D):
                kernels.append(layer.kernel_size)
                strides.append(layer.strides)
    activations = activation_model.predict(img)
    average_layer_maps = []
    for layer_activation in activations:  # Only the convolutional layers
        feature_maps = layer_activation[0]
        n_features = feature_maps.shape[-1]
        average_feature_map = np.sum(feature_maps, axis=-1) / n_features

        # normalize map
        map_min = np.min(average_feature_map)
        map_max = np.max(average_feature_map)
        normal_map = (average_feature_map - map_min) / (map_max - map_min + 1e-6)
        # dim: height x width
        average_layer_maps.append(normal_map)

    # add batch and channels dimension to tensor
    average_layer_maps = [fm[np.newaxis, :, :, np.newaxis] for fm in average_layer_maps]  # dim: bhwc
    saliency_mask = tf.convert_to_tensor(average_layer_maps[-1])
    for l in reversed(range(0, len(average_layer_maps))):
        kernel = np.ones((*kernels[l], 1, 1))

        if l > 0:
            output_shape = average_layer_maps[l - 1].shape
        else:
            # therefore, the height and width in the image shape need to be reversed
            output_shape = (1, *(IMAGE_SHAPE[:2]), 1)

        saliency_mask = tf.nn.conv2d_transpose(saliency_mask, kernel, output_shape, strides[l], padding='VALID')
        if l > 0:
            saliency_mask = tf.multiply(saliency_mask, average_layer_maps[l - 1])

    saliency_mask = tf.squeeze(saliency_mask, axis=0)  # remove batch dimension
    return (saliency_mask, hiddens, None) if hiddens else saliency_mask

def load_image(image_path):
    img = Image.open(image_path)
    img = img.resize(IMAGE_SHAPE_CV)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.convert_to_tensor(img_array)
    return img_array

def plot_saliency_map(saliency_map, image_size, root, file_path, title='Saliency Map'):
    # def plot_saliency_map(saliency_map, image_size, title='Saliency Map'):
    # Normalize the saliency map for better visualization
    saliency_map = (saliency_map - tf.reduce_min(saliency_map)) / (tf.reduce_max(saliency_map) - tf.reduce_min(saliency_map) + 1e-6)
    
    # Convert the saliency map to a numpy array if it's not already
    saliency_map = saliency_map.numpy() if isinstance(saliency_map, tf.Tensor) else saliency_map
    
    # Resize the saliency map to match the original image size if necessary
    if saliency_map.shape[:2] != image_size:
        saliency_map = tf.image.resize(saliency_map, image_size)
        saliency_map = saliency_map.numpy()  # Convert to numpy array
    
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.imshow(saliency_map, cmap='hot')
    plt.colorbar()  # Adds a colorbar to interpret values
    plt.title(title)
    plt.axis('off')  # Hide the axis
    # plt.show()
    plt.savefig(os.path.join(root, file_path))
    plt.close()
    print("Saved", file_path)

for index in range(1,2):
    root_directory = f'../gazebo_environment/markerless/diff_images'
    saliency_directory = f'./saliency_maps_markerless_mix_heights'
    if not os.path.exists(saliency_directory):
        os.makedirs(saliency_directory, exist_ok=True)
    sorted_files = os.listdir(root_directory)

    for i in range (len(sorted_files) - 2):
        file = 'Image'+str(i + 1)+'.png'
        print("Processing", file)
        file_path = os.path.join(root_directory, file)
        img = load_image(file_path)
        saliency_map = compute_visualbackprop(img, vis_model)
        plot_saliency_map(saliency_map, (224, 224), saliency_directory, file)
    

# img = load_image('../fly_to_target_dataset/diff_dataset/1/Image500.png')
# img = load_image('../gazebo_environment/markerless/diff_images/Image.png')
# saliency_map = compute_visualbackprop(img, vis_model)
# plot_saliency_map(saliency_map, (224, 224))
# print(saliency_map.shape)



# run_visualization(
#                 vis_model=vis_model,
#                 data=data_path,
#                 vis_func=vis_func,
#                 image_output_path= output_name+"/"+data_model_id,
#                 video_output_path=os.path.join(output_name, f"{data_model_id}.mp4"),
#                 reverse_channels=reverse_channels,
#                 control_source=csv_path if csv_path else control_model,
#                 vis_kwargs=vis_kwargs,
#                 absolute_norm=absolute_norm,
#             )
# print(f"Finished {data_model_id}")