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

IMAGE_SHAPE = (144, 256, 3)
IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])

with tf.device('/cpu:0'):
    model = tf.keras.models.load_model('my_model_b64.h5')
root = "./dataset/1"
image_paths = os.listdir(root)
image_paths.sort()
image_paths = image_paths[0:64]
file_ending = 'png'

csv_file_name = "dataset/1/data_out.csv" 
labels = np.genfromtxt(csv_file_name, delimiter=',', skip_header=1, dtype=np.float32)
#print("labels", labels)
predictions = []


shift: int = 1
stride: int = 1
decay_rate: float = 0.95
val_split: float = 0.1
label_scale: float = 1
seq_len = 64
val_split: float = 0.1
label_scale: float = 1

array = []

for path in image_paths:
    if file_ending in path:
        img = Image.open(root + '/' + path)
        img = img.resize(IMAGE_SHAPE_CV)
        img_array = np.array(img)
        array.append(img_array)

array = np.array(array)
array = np.expand_dims(array, axis=0) 
print(array.shape)    

with tf.device('/cpu:0'):
    start = time.time()
    predictions = model.predict(array)
    end= time.time()
print("Time: ", end - start)

predictions_df = pd.DataFrame(predictions[0])

# Save the DataFrame to a CSV file
predictions_df.to_csv("predictions.csv", index=False)
