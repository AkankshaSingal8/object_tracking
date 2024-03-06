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

import seaborn as sns
import pandas as pd

IMAGE_SHAPE = (144, 256, 3)
IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])

def load_dataset_multi(root, image_size, seq_len, shift, stride, label_scale):
    file_ending = 'png'
    IMAGE_SHAPE = (144, 256, 3)
    IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])

    def sub_to_batch(sub_feature, sub_label):
        sfb = sub_feature.batch(seq_len, drop_remainder=True)
        slb = sub_label.batch(seq_len, drop_remainder=True)
        return tf.data.Dataset.zip((sfb, slb))
        # return sub.batch(seq_len, drop_remainder=True)

    
    datasets = []

    #output_means, output_stds = get_output_normalization(root)

    
    for directory in range(1, 2):
        csv_file_name = root + "/" + str(directory) + '/data_out.csv'
        labels = np.genfromtxt(csv_file_name, delimiter=',', skip_header=1, dtype=np.float32)
        print("labels", labels)
        # if labels.shape[1] == 4:
        #     labels = (labels - output_means) / output_stds
        #     # labels = labels * label_scale
        # elif labels.shape[1] == 5:
        #     labels = (labels[:, 1:] - output_means) / output_stds
        #     # labels = labels[:,1:] * label_scale
        # else:
        #     raise Exception('Wrong size of input data (expected 4, got %d' % labels.shape[1])
    
        labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
        # n_images = len(os.listdir(os.path.join(root, d))) - 1
        n_images = len([fn for fn in os.listdir('./' + root + "/" + str(directory)) if file_ending in fn])
        print("no of imgs", n_images)
        # dataset_np = np.empty((n_images, 256, 256, 3), dtype=np.uint8)
        dataset_np = np.empty((n_images, *image_size), dtype=np.uint8)

        for ix in range(n_images):
            # dataset_np[ix] = imread(os.path.join(root, d, '%06d.jpeg' % ix))
            img_file_name = root + "/" + str(directory) +'/Image' + str(ix + 1) + '.'+ file_ending
            img = Image.open(img_file_name)
            img = img.resize(IMAGE_SHAPE_CV)
            # dataset_np[ix] = img[img.height - image_size[0]:, :, :]
            dataset_np[ix] = img

        images_dataset = tf.data.Dataset.from_tensor_slices(dataset_np)
        dataset = tf.data.Dataset.zip((images_dataset, labels_dataset))
        #dataset = tf.data.Dataset.zip((images_dataset))
        dataset = dataset.window(seq_len, shift=shift, stride=stride, drop_remainder=True).flat_map(sub_to_batch)
        datasets.append(dataset)

    return datasets

def get_dataset_multi(root, image_size, seq_len, shift, stride, validation_ratio, label_scale, extra_data_root=None):
    ds = load_dataset_multi(root, image_size, seq_len, shift, stride, label_scale)
    print('n bags: %d' % len(ds))
    cnt = 0

    for d in ds:
        for (ix, _) in enumerate(d):
            pass
            cnt += ix
    print('n windows: %d' % cnt)

    

    val_ix = int(len(ds) * validation_ratio)
    print('\nval_ix: %d\n' % val_ix)
    #validation_datasets = ds[:val_ix]

    training_datasets = ds[val_ix:]

    # if either dataset has length 0, trying to call flat map raises error that return type is wrong
    # assert len(training_datasets) > 0 and len(validation_datasets) > 0, f"Training or validation dataset has no points!" \
    #                                                                     f"Train dataset len: {len(training_datasets)}" \
    #                                                                     f"Val dataset len: {len(validation_datasets)}"
    training = tf.data.Dataset.from_tensor_slices(training_datasets).flat_map(lambda x: x)
    #validation = tf.data.Dataset.from_tensor_slices(validation_datasets).flat_map(lambda x: x)

    #return training, validation
    return training

model = tf.keras.models.load_model('my_model_b64.h5')
# root = "./dataset/1"
# image_paths = os.listdir(root)
# image_paths.sort()
# print(image_paths)
file_ending = 'png'

# csv_file_name = "dataset/1/data_out.csv" 
# labels = np.genfromtxt(csv_file_name, delimiter=',', skip_header=1, dtype=np.float32)
# #print("labels", labels)
# predictions = []


shift: int = 1
stride: int = 1
decay_rate: float = 0.95
val_split: float = 0.1
label_scale: float = 1
seq_len = 64
val_split: float = 0.1
label_scale: float = 1

training_dataset = get_dataset_multi('dataset', IMAGE_SHAPE, seq_len, shift, stride, val_split, label_scale, extra_data_root=None)
print('load dataset shape', training_dataset.element_spec)
training_dataset = training_dataset.batch(64)
print('load dataset shape', training_dataset.element_spec)

predictions = model.predict(training_dataset)
print(predictions)

# for path in image_paths:
#     if file_ending in path:
#         img = Image.open(root + '/' + path)
#         img = img.resize(IMAGE_SHAPE_CV)
#         img_array = np.array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array = np.expand_dims(img_array, axis=1)
#         prediction = model.predict(img_array)
#         print(prediction)
#         print(path)
        

#predictions = model.predict(preprocessed_images)
# df_predictions = pd.DataFrame(predictions, columns=['Image Path'] + [f'Prediction {i+1}' for i in range(prediction.shape[1])])
# df_predictions.to_csv(csv_file_name, index=False)

# print(f"Predictions saved to {csv_file_name}")