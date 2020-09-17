import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib.image as mpimg
import pathlib


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


def plot_pred_actual(y_test, target_labels, predictions, x_test, test_predictions):
    import random
    plt.figure(figsize=(30,40))
    for counter, i in enumerate(random.sample(range(0, len(y_test)), 30)): # random 30 images
        plt.subplot(6, 5, counter+1)
        plt.subplots_adjust(hspace=0.6)
        actual = str(target_labels[i])
        predicted = str(predictions[i])
        conf = str(max(test_predictions[i]))
        plt.imshow(x_test[i]/255.0)
        plt.axis('off')
        plt.title('Actual: ' + actual + '\nPredict: ' + predicted + '\nConf: ' + conf, fontsize=18)
        
    plt.show()
