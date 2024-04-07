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
    model = tf.keras.models.load_model('model_ssfalse_b64_lr0.0001wscheduler_seqlen64_new_dataset.h5')
root = "./dataset/1"
image_paths = [path for path in sorted(os.listdir(root)) if 'png' in path][:593]
# print(image_paths)
file_ending = 'png'

# csv_file_name = "dataset/1/data_out.csv" 
#labels = np.genfromtxt(csv_file_name, delimiter=',', skip_header=1, dtype=np.float32)
#print("labels", labels)
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

predictions_df = pd.DataFrame(predictions)

# Save the DataFrame to a CSV file
predictions_df.to_csv("predictions_sliding_window_added_0s_scheduler_593.csv", index=False)
