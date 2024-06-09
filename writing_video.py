import json
import os
from pathlib import Path
from typing import Optional, Callable, Sequence, Union, Dict, Any, Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from numpy import ndarray
from pandas import DataFrame
from tensorflow import Tensor
from tensorflow.python.keras.models import Functional
from tqdm import tqdm

def write_video(img_seq: Sequence[ndarray], output_path: str, fps: int = 10):
    Path(os.path.dirname(output_path)).mkdir(exist_ok=True, parents=True)
    seq_shapes = [img.shape for img in img_seq]
    assert seq_shapes.count(seq_shapes[0]) == len(seq_shapes), "Not all shapes in img_seq are the same"

    image_shape = img_seq[0].shape
    cv_shape = (image_shape[1], image_shape[0])  # videowriter takes width, height, image_shape is height, width
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, cv_shape,
                             True)  # true means write color frames

    for img in img_seq:
        writer.write(img)

    writer.release()
    

def load_image(image_path, target_size):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    return img_array

def image_array(root_directory, output_path):
    images = []
    
    for i in range (len(os.listdir(root_directory)) - 1):
        file = 'Image'+str(i + 1)+'.png'
        print("Processing", file)
        file_path = os.path.join(root_directory, file)
        img = load_image(file_path, (224, 224))
        images.append(img)
    return images


root = './saliency_maps/1'
output_path = './saliency_maps/saliency_traj1.mp4'
images = image_array(root, output_path)
write_video(images, output_path)