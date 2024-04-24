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
DROPOUT = 0.1

DEFAULT_NCP_SEED = 22222

IMAGE_SHAPE = (144, 256, 3)
IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])

batch_size = None
seq_len = 64
augmentation_params = None
single_step = False
no_norm_layer = False
model = generate_ncp_model(seq_len, IMAGE_SHAPE, augmentation_params, batch_size, DEFAULT_NCP_SEED, single_step, no_norm_layer)

model.load_weights('model-ncp-val.hdf5')
# predictions = model.predict(images)

root = "./dataset/2"
image_paths = [path for path in sorted(os.listdir(root)) if 'png' in path]

predictions = []

image_sequences = []

for i in range(len(image_paths)):
    # Load and process the current and next image in the list
    current_image_path = os.path.join(root, image_paths[i])
    current_img = Image.open(current_image_path).resize(IMAGE_SHAPE_CV)
    current_img_array = np.array(current_img)

    # For the first image, just replicate it 64 times
    if i == 0:
        seq = np.stack([current_img_array] * 64)
        seq = np.expand_dims(seq, axis=0)
        print(seq.shape)
        image_sequences.append(seq)
    else:
        # Append the current image to the last sequence
        last_seq = image_sequences[-1]
        new_seq = np.append(last_seq[0][1:], [current_img_array], axis=0)
        new_seq = np.expand_dims(new_seq, axis=0) 
        # print(new_seq.shape)
        image_sequences.append(new_seq)

# Convert the list of sequences into a numpy array for easier handling
image_sequences_array = np.array(image_sequences)
print(image_sequences_array.shape) 

times = []
for seq in image_sequences_array:
    preds = []
    with tf.device('/cpu:0'):
        start = time.time()
        preds.append(model.predict(seq))
        end= time.time()
        times.append(end - start)
        print(end - start)
    predictions.append(preds[0][0][63])


print("Avg Time: ", sum(times) / len(times))
print(len(predictions))
predictions_df = pd.DataFrame(predictions)

# Save the DataFrame to a CSV file
predictions_df.to_csv("predictions_pretrained_model_data2.csv", index=False)