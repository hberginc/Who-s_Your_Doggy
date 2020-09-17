from tensorflow import keras
from tensorflow.keras.preprocessing.image import (ImageDataGenerator, 
                                                  array_to_img, img_to_array, 
                                                  load_img) 
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras import backend as K
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os

import warnings
warnings.filterwarnings("ignore")

import matplotlib.image as mpimg
import pathlib
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import VGG16, Xception, InceptionV3
from tensorflow.keras.optimizers import Nadam, Adam
from tensorflow.keras.utils import plot_model


from helper_functions import *
from build_models import *

from datetime import datetime
import datetime
import time


if __name__ == '__main__':
    train_generator, validation_generator, holdout_generator = create_data_gens(target_size=(229,229),train_dir = "../../images/Images/train",  val_dir = '../../images/Images/val', holdout_dir = '../../images/Images/test', batch_size = 30)

    X_model = load_model('../../Xception_mod_2.h5')


    Xception_history = X_model.fit(train_generator,
                steps_per_epoch=3034 // 15,
                epochs=10,
                validation_data=validation_generator,
                validation_steps=650 // 16,
                verbose = 1)

    X_model.save('../../Xception_mod3.h5')
