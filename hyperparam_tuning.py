import copy
import time
import warnings
from typing import Dict, Any, Callable, Optional, Tuple

import numpy as np
import optuna
from optuna import Trial
from optuna.integration import TFKerasPruningCallback
from tensorflow.python.keras.callbacks import History

from tf_data_training import train_model
from utils.model_utils import NCPParams, CTRNNParams, LSTMParams, TCNParams