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


