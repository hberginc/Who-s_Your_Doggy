from tensorflow import keras
from tensorflow.keras.preprocessing.image import (ImageDataGenerator, 
                                                  array_to_img, img_to_array, 
                                                  load_img) 
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib.image as mpimg
import pathlib
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import VGG16, Xception
from tensorflow.keras.optimizer import Nadam, adam
from datetime import datetime
import datetime
import time



def create_data_gens(target_size = (229,229), train_dir = '../../images/Images/train', val_dir = '../../images/Images/val', holdout_dir =  '../../images/Images/test',  batch_size = 16):
    '''
    this is the augmentation configuration we will use for training
    PARAMS: train, val, holdout dirs are directories geared toward the storage of such data

    all images are resized to (229,229)

    RETURNS:
    each image generator in order train, val, holdout
   '''
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
            train_dir, 
            shuffle=True, # this is the target directory
            target_size=target_size,  # all images will be resized to 150x150
            batch_size=batch_size,
            class_mode='categorical',
            seed = 42)  # since we use CategoricalCrossentropy loss, we need categorical labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
            val_dir,
            target_size=target_size,
            batch_size=185,
            class_mode='categorical',
            shuffle = False,
            seed = 42)


    return train_generator, validation_generator


def basic_cnn(n_categs = 5):
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
    model.add(Dense(n_categs))
    model.add(Activation('softmax'))
    model.compile(optimizer='adam', loss=['categorical_crossentropy'], metrics=['accuracy', 'top_k_categorical_accuracy'])

    return model


def basic_transfer_model(input_size, n_categories, weights = 'imagenet', trans_model = VGG16):
    # note that the "top" is not included in the weights below
    base_model = trans_model(weights=weights,
                        include_top=False,
                        input_shape=input_size)
    model = base_model.output
    # add new head
    model = GlobalAveragePooling2D()(model)
    predictions = Dense(n_categories, activation='softmax')(model)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def create_new_transfer_model(input_size, n_categories, weights = 'imagenet', trans_model = Xception):
    # note that the "top" is not included in the weights below
    base_model = trans_model(weights=weights,
                        include_top=False,
                        input_shape=input_size)
    
    model = base_model.output

    model = GlobalAveragePooling2D()(model)
    model = Dropout(0.3)(model)
    model = Dense(500, activation='relu')(model)
    model = Dense(500, activation='relu')(model)

    output = Dense(n_categories, activation='softmax')(model)
    model = Model(inputs=base_model.input, outputs=output)
    return model


def print_model_properties(model, indices = 0):
     for i, layer in enumerate(model.layers[indices:]):
        print("Layer {} | Name: {} | Trainable: {}".format(i+indices, layer.name, layer.trainable))

def change_trainable_layers(model, trainable_index):
    for layer in model.layers[:trainable_index]:
        layer.trainable = False
    for layer in model.layers[trainable_index:]:
        layer.trainable = True
        

def create_callbacks(file_path = "./../logs/", patience = 3):
    tensorboard = TensorBoard(log_dir= file_path+datetime.datetime.now().strftime("%m%d%Y%H%M%S"),
                                histogram_freq=0,
                                write_graph=False,
                                update_freq='epoch')
    early_stopping = EarlyStopping(restore_best_weights = True, patience = patience, monitor='val_loss')
    
    return tensorboard, early_stopping


def evaluate_model(model, holdout_generator):
    """
    evaluates model on holdout data
    params: model to evaluate
    Returns: [loss, accuracy]
        """

    metrics = model.evaluate(holdout_generator,
                                        use_multiprocessing=True,
                                        verbose=1)
    return metrics

def load_final_model(mod_file_path = "../models_and_weights/Xception_mod3_run2.h5", weights = '../models_and_weights/weights_Xception_mod3_run2' ):
    model = load_model(mod_file_path)
    model.load_weights(weights)
    return model



if __name__ == '__main__':
    model = load_final_model()

    train_generator, validation_generator, holdout_generator = create_data_gens()

        #  Added for Tensorboard
    tensorboard = TensorBoard(log_dir='./logs',
                            histogram_freq=2,
                            batch_size=16,
                            write_graph=True,
                            write_grads=True,
                            write_images=True,
                            ) 

    early_stopping = EarlyStopping(patience = 2)
    # modified for Tensorboard


    change_trainable_layers(model, 60)
    model.compile(optimizer=Nadam(lr=0.001), loss=['categorical_crossentropy'], metrics=['accuracy', 'top_k_categorical_accuracy'])

    Xception_history1 = model.fit(train_generator,
                steps_per_epoch=3000 // 16,
                epochs=10,
                validation_data=validation_generator,
                validation_steps=600 // 16,
                verbose = 1)


    # !tensorboard --logdir=logs/ --port=8889

    