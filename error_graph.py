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
import matplotlib.pyplot as plt
import cv2 as cv

result_directory = "SINGLE_STEP_FINE_TUNED"
# for directory in range(os.listdir(result_directory)):
    # csv_file_name = f'original_dataset/{directory}/data_out.csv'
    # labels = np.genfromtxt(csv_file_name, delimiter=',', skip_header=1, dtype=np.float32)

    # preds_file_name = f'FINE_TUNED/predictions_test_data{directory}.csv'
    # pred_vals = np.genfromtxt(preds_file_name, delimiter=',', skip_header=1, dtype=np.float32)[:len(labels)]

    # print(len(labels), len(pred_vals))
    # error = labels - pred_vals

error = pd.read_csv(f"{result_directory}/errors_test_data1.csv")
print(error)
# fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

# for i in range(4):
#     axs[i].plot(error[:, i])
#     axs[i].set_title(f'Error in Dimension {i+1}')
#     axs[i].set_ylabel('Error')
#     axs[i].grid(True)

# axs[3].set_xlabel('Sequence Length')

# plt.tight_layout()
# plt.show()

# error_df = pd.DataFrame(error)

    # # # Save the DataFrame to a CSV file
    # error_df.to_csv(f'FINE_TUNED/errors_test_data{directory}.csv', index=False)