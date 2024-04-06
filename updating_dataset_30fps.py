import os
from typing import Iterable, Dict

import csv

import numpy as np
from matplotlib.image import imread
from tqdm import tqdm
from PIL import Image

import seaborn as sns
import pandas as pd

root = "dataset"
file_ending = 'png'

for directory in range(1, 12):
    #print("directory", directory)
    csv_file_name = root + "/" + str(directory) + '/data_out.csv'
    labels = np.genfromtxt(csv_file_name, delimiter=',', skip_header=1, dtype=np.float32)
    #print("labels len", labels.shape)
    d = labels.shape[0] % 30
    if d == 0:
        diff_labels = 0
    else:
        diff_labels = 30 - d
 
    print("labels", labels[-1])
    
    # with open(csv_file_name, 'a', newline='') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     for i in range(diff_labels):
    #         csvwriter.writerow(labels[-1])
   

   
    n_images = len([fn for fn in os.listdir('./' + root + "/" + str(directory)) if file_ending in fn])
    print(n_images)
  
    d_img = n_images % 30
    
    if d_img == 0:
        diff = 0
    else:
        diff = 30 - d_img
    print("diff_labels", diff_labels, " diff", diff, " diff_labels%64", (labels.shape[0] + diff_labels)%64, " diff%64", (n_images + diff)%64)
    img_file_name = root + "/" + str(directory) +'/Image' + str(n_images) + '.'+ file_ending
    img = Image.open(img_file_name)
    for i in range(1, diff + 1):
       
        new_filename = os.path.join(root,str(directory))
        fname =  'Image' + str(i + n_images) + '.'+ file_ending
        file_path = os.path.join(new_filename, fname)
        
        img.save(file_path)
        