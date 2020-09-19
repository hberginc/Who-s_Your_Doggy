import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib.image as mpimg
import pathlib
from predictions import *
from build_models import *

def find_missclass_indx(real_labels, prediction_labels):
    incorrect_index = []
    for ind, (i, j) in enumerate(zip(real_labels, prediction_labels)):
      if i != j:
        print('True- {}, \nPred- {}\n'.format(i,j))
        incorrect_index.append(ind)
    return incorrect_index


def plot_missclass(holdout_generator, predictions):
    x,y = holdout_generator.next()
    prediction_labels, real_labels = get_real_pred(predictions, holdout_generator)
    
    incorrect_index = find_missclass_indx(real_labels, prediction_labels)
    fig, axs = plt.subplots(4,2, figsize = (20,20))
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

