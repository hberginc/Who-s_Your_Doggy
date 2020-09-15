# Transfer Learning use the VGG16 architecture, pre-trained on the ImageNet dataset

from tensorflow import keras
from tensorflow.keras.preprocessing.image import (ImageDataGenerator, 
                                                  array_to_img, img_to_array, 
                                                  load_img) 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
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
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping





def create_data_gens(train_dir = '../../images/Images/train', val_dir = '../../images/Images/val', batch_size = 16):
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
            train_dir,  # this is the target directory
            target_size=(150, 150),  # all images will be resized to 150x150
            batch_size=batch_size,
            class_mode='categorical')  # since we use CategoricalCrossentropy loss, we need categorical labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
            val_dir,
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode='categorical')
    return train_generator, validation_generator

def basic_cnn():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5))
    model.add(Activation('softmax'))

    loss_fn = keras.losses.CategoricalCrossentropy(
        from_logits=False,
        label_smoothing=0,
        reduction="auto",
        name="categorical_crossentropy",
    )

    model.compile(loss=loss_fn, optimizer='adam', metrics=['accuracy'])
    return model


def transer_model():





if __name__ == '__main__':


    train_generator, validation_generator = create_data_gens()

    #first basic model saved weights
    model = basic_cnn()
        #  Added for Tensorboard
    tensorboard = TensorBoard(log_dir='./logs',
                            histogram_freq=2,
                            batch_size=batch_size,
                            write_graph=True,
                            write_grads=True,
                            write_images=True,

                            ) 
    early_stopping = EarlyStopping(patience = 2)
    # modified for Tensorboard

    epochs = 10
    history = model.fit(
        train_generator,
        steps_per_epoch=2000 // 16,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=800 // 16,
        verbose = 1,
        callbacks = [tensorboard, early_stopping])

    # !tensorboard --logdir=logs/ --port=8889

    