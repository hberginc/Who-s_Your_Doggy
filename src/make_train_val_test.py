# # Creating Train / Val / Test folders (One time use)
import os
import shutil
import numpy as np
import pandas as pd

root_dir = '../images/Images'
Chihuahua = '/n02085620-Chihuahua/'
Japanese_spaniel = '/n02085782-Japanese_spaniel'
Maltese = '/n02085936-Maltese_dog'
Pekinese = '/n02086079-Pekinese'
Shih_Tzu = '/n02086240-Shih-Tzu'

os.makedirs(root_dir +'/train' + Chihuahua)
os.makedirs(root_dir +'/train' + Shih_Tzu)
os.makedirs(root_dir +'/train' + Japanese_spaniel)
os.makedirs(root_dir +'/train' + Maltese)
os.makedirs(root_dir +'/train' + Pekinese)

os.makedirs(root_dir +'/val' + Chihuahua)
os.makedirs(root_dir +'/val' + Shih_Tzu)
os.makedirs(root_dir +'/val' + Japanese_spaniel)
os.makedirs(root_dir +'/val' + Maltese)
os.makedirs(root_dir +'/val' + Pekinese)

os.makedirs(root_dir +'/test' + Chihuahua)
os.makedirs(root_dir +'/test' + Shih_Tzu)
os.makedirs(root_dir +'/test' + Japanese_spaniel)
os.makedirs(root_dir +'/test' + Maltese)
os.makedirs(root_dir +'/test' + Pekinese)
# Creating partitions of the data after shuffeling
currentCls = Shih_Tzu
src = root_dir+currentCls # Folder to copy images from
allFileNames = os.listdir(src)
np.random.shuffle(allFileNames)
train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                          [int(len(allFileNames)*0.7), int(len(allFileNames)*0.85)])
train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]
print('Total images: ', len(allFileNames))
print('Training: ', len(train_FileNames))
print('Validation: ', len(val_FileNames))
print('Testing: ', len(test_FileNames))
# Copy-pasting images
for name in train_FileNames:
    shutil.copy(name,  "images/Images/train"+currentCls)
for name in val_FileNames:
    shutil.copy(name,  "images/Images/val"+currentCls)
for name in test_FileNames:
    shutil.copy(name, "images/Images/test"+currentCls)