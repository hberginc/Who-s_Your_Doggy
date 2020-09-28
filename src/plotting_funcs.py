import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib.image as mpimg
import pathlib
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
from tensorflow.keras.optimizers import Nadam
from datetime import datetime
import datetime
import time

from PIL import Image, ImageSequence
import imageio
import glob
import build_models as bld





def get_preds(model, holdout_generator):
    pred = model.predict(holdout_generator,verbose=1)
    return pred

def get_real_pred(predictions, holdout_generator):
    predicted_class_indices = np.argmax(predictions,axis=1)
    labels = holdout_generator.class_indices
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]
    real_classes = holdout_generator.classes
    real_labels = [labels[k] for k in real_classes]
    predictions = ['-'.join(lab.split('-')[1:]) for lab in predictions]
    real_labels = ['-'.join(lab.split('-')[1:]) for lab in real_labels]
    return predictions, real_labels

def find_missclass_indx(real_labels, prediction_labels):
    incorrect_index = []
    for ind, (i, j) in enumerate(zip(real_labels, prediction_labels)):
      if i != j:
        print('True- {}, \nPred- {}\n'.format(i,j))
        incorrect_index.append(ind)
    return incorrect_index

def plot_missclass(holdout_generator, predictions, axs = 'ax'):
    x,y = holdout_generator.next()
    prediction_labels, real_labels = get_real_pred(predictions, holdout_generator)
    
    incorrect_index = find_missclass_indx(real_labels, prediction_labels)
    ax = axs.flatten()
    for i, ind in enumerate(incorrect_index):
        rl = real_labels[ind]
        pl = prediction_labels[ind]
        image = x[ind]
        ax[i].set_title('Actual: {}, \n Predicted: {}'.format(rl, pl))
        ax[i].imshow(image)
        ax[i].axis('off')


def plot_one_img(img_path, title, save_path=None):
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.title(title, fontsize = 15)
    plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show();
    
def show_imgs(direct, num_imgs=20):
    images = os.listdir(direct)[:num_imgs]
    plt.figure(figsize=(18,18))
    for i in range(num_imgs):
        #connect directory to selected breed path and image number
        img = mpimg.imread(direct + '/'+ images[i])
        plt.subplot(num_imgs/5+1, 5, i+1)
        plt.imshow(img)
        plt.axis('off')


def plot_acc_loss_per_epoch(fit_model, epochs=10, file_name = 'train_acc_loss_basic.png'):
    acc = fit_model.history['accuracy']
    val_acc = fit_model.history['val_accuracy']
    top_k_acc = fit_model.history['top_k_categorical_accuracy']

    loss=fit_model.history['loss']
    val_loss=fit_model.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.plot(epochs_range, top_k_acc, label = 'Top 5 Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(file_name)
    plt.show()


def get_activations(model, validation_generator, n = 5):
    preds = model.predict(validation_generator)
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    X_train, y_train = validation_generator.next()
    activations = activation_model.predict(X_train[n].reshape(1,229, 229, 3))
    return activations

def display_activation_layer(activations, n_imgs, act_index, ax = 'ax', figsize=(8,4), title = None): 
    fig, ax = plt.subplots(2, n_imgs//2, figsize= figsize)
    ax = ax.flatten()
    activation = activations[act_index]
    activation_index=0
    for im in range(n_imgs):
        ax[im].imshow(activation[0, :, :, activation_index])
        activation_index += 1
        ax[im].axis('off')
    fig.suptitle(title, fontsize = 20)

def show_preview_imgs(direct, num_imgs=20):
    images = os.listdir(direct)[:num_imgs]
    for i in range(num_imgs):
        #connect directory to selected breed path and image number
        img = mpimg.imread(direct + '/'+ images[i])
        plt.subplot(2, 4, i+1)
        plt.imshow(img)
        plt.axis('off')


if __name__ == '__main__': 
    #print image augmentations
    plt.figure(figsize=(12,5))
    plt.title('Boston Bull Augmentation', loc = 'center')
    show_preview_imgs('../preview', num_imgs=8)
    plt.tight_layout()
    plt.show()
    # plt.savefig('to_image_process_or_to_timeseries/visuals/chow.png')
    
    #print activation Layers
    train_generator, validation_generator = bld.create_data_gens(target_size = (229,229), train_dir = "../../images/Images/train", 
                                                val_dir = '../../images/Images/test', batch_size = 16)

    model = load_model('../models_and_weights/models/Xception_mod3_run2.h5')
    activations = get_activations(model, validation_generator, n=114)
    display_activation_layer(activations, 6, 17, figsize=(9,6), title = 'Activation Layer 3')
    plt.show();

    png_dir = '../../previews/all/'
    images = []
    for file_name in os.listdir(png_dir):
        if file_name.endswith('.jpeg'):
            file_path = os.path.join(png_dir, file_name)
            images.append(imageio.imread(file_path))
    kargs = { 'duration': 1.5 }
    imageio.mimsave('../visuals/animal_imgs/full.gif', images, **kargs)

