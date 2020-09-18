# # Creating Train / Val folders (One time use)
import os
import shutil
import numpy as np
import pandas as pd

root_dir = '../../images/Images'
Chihuahua = '/n02085620-Chihuahua/'
Japanese_spaniel = '/n02085782-Japanese_spaniel'
Maltese = '/n02085936-Maltese_dog'
Pekinese = '/n02086079-Pekinese'
Shih_Tzu = '/n02086240-Shih-Tzu'

Blenheim_spaniel = '/n02086646-Blenheim_spaniel'
Papillon = '/n02086910-papillon'
Rhodesian_ridgeback = '/n02087394-Rhodesian_ridgeback'
Basset = '/n02088238-basset'
Bloodhound = '/n02088466-bloodhound'
Whippet = '/n02091134-whippet'
Boston_bull = '/n02096585-Boston_bull'
Standard_schnauzer = '/n02097209-standard_schnauzer'
Scotch_terrier = '/n02097298-Scotch_terrier'
Lab = '/n02099712-Labrador_retriever'
Golden = '/n02099601-golden_retriever'
English_setter = '/n02100735-English_setter'
Kuvasz = '/n02104029-kuvasz'
Rottweiler = '/n02106550-Rottweiler'
German_Shepherd = '/n02106662-German_shepherd'
Doberman = '/n02107142-Doberman'
Great_Dane = '/n02109047-Great_Dane'
pug = '/n02110958-pug'
standard_poodle = '/n02113799-standard_poodle'
Chow = '/n02112137-chow'


lst = [Chihuahua,Japanese_spaniel, Maltese, Pekinese, Shih_Tzu, Blenheim_spaniel,  Papillon, 
Rhodesian_ridgeback, Basset, Bloodhound, Whippet, Boston_bull, Standard_schnauzer, Scotch_terrier, 
 Lab, Golden, English_setter, Kuvasz, Rottweiler, German_Shepherd, Doberman, Great_Dane, pug, standard_poodle,
 Chow]


os.makedirs(root_dir +'/train' + Chihuahua)
os.makedirs(root_dir +'/val' + Chihuahua)
os.makedirs(root_dir +'/test' + Chihuahua)


os.makedirs(root_dir +'/train' + Pekinese)
os.makedirs(root_dir +'/val' + Pekinese)
os.makedirs(root_dir +'/test' + Pekinese)

os.makedirs(root_dir +'/train' + Maltese)
os.makedirs(root_dir +'/val' + Maltese)
os.makedirs(root_dir +'/test' + Maltese)

os.makedirs(root_dir +'/train' + Japanese_spaniel)
os.makedirs(root_dir +'/val' + Japanese_spaniel)
os.makedirs(root_dir +'/test' + Japanese_spaniel)

os.makedirs(root_dir +'/train' + Shih_Tzu)
os.makedirs(root_dir +'/val' + Shih_Tzu)
os.makedirs(root_dir +'/test' + Shih_Tzu)

os.makedirs(root_dir +'/train' + Blenheim_spaniel)
os.makedirs(root_dir +'/val' + Blenheim_spaniel)
os.makedirs(root_dir +'/test' + Blenheim_spaniel)

os.makedirs(root_dir +'/train' + Papillon)
os.makedirs(root_dir +'/val' + Papillon)
os.makedirs(root_dir +'/test' + Papillon)

os.makedirs(root_dir +'/train' + Rhodesian_ridgeback)
os.makedirs(root_dir +'/val' + Rhodesian_ridgeback)
os.makedirs(root_dir +'/test' + Rhodesian_ridgeback)

os.makedirs(root_dir +'/train' + Basset)
os.makedirs(root_dir +'/val' + Basset)
os.makedirs(root_dir +'/test' + Basset)

os.makedirs(root_dir +'/train' + Bloodhound)
os.makedirs(root_dir +'/val' + Bloodhound)
os.makedirs(root_dir +'/test' + Bloodhound)

os.makedirs(root_dir +'/train' + Whippet)
os.makedirs(root_dir +'/val' + Whippet)
os.makedirs(root_dir +'/test' + Whippet)

os.makedirs(root_dir +'/train' + Boston_bull)
os.makedirs(root_dir +'/val' + Boston_bull)
os.makedirs(root_dir +'/test' + Boston_bull)

os.makedirs(root_dir +'/train' + Standard_schnauzer)
os.makedirs(root_dir +'/val' + Standard_schnauzer)
os.makedirs(root_dir +'/test' + Standard_schnauzer)

os.makedirs(root_dir +'/train' + Scotch_terrier)
os.makedirs(root_dir +'/val' + Scotch_terrier)
os.makedirs(root_dir +'/test' + Scotch_terrier)

os.makedirs(root_dir +'/train' + Lab)
os.makedirs(root_dir +'/val' + Lab)
os.makedirs(root_dir +'/test' + Lab)

os.makedirs(root_dir +'/train' + Golden)
os.makedirs(root_dir +'/val' + Golden)
os.makedirs(root_dir +'/test' + Golden)

os.makedirs(root_dir +'/train' + English_setter)
os.makedirs(root_dir +'/val' + English_setter)
os.makedirs(root_dir +'/test' + English_setter)

os.makedirs(root_dir +'/train' + Kuvasz)
os.makedirs(root_dir +'/val' + Kuvasz)
os.makedirs(root_dir +'/test' + Kuvasz)

os.makedirs(root_dir +'/train' + Rottweiler)
os.makedirs(root_dir +'/val' + Rottweiler)
os.makedirs(root_dir +'/test' + Rottweiler)

os.makedirs(root_dir +'/train' + German_Shepherd)
os.makedirs(root_dir +'/val' + German_Shepherd)
os.makedirs(root_dir +'/test' + German_Shepherd)

os.makedirs(root_dir +'/train' + Doberman)
os.makedirs(root_dir +'/val' + Doberman)
os.makedirs(root_dir +'/test' + Doberman)

os.makedirs(root_dir +'/train' + Great_Dane)
os.makedirs(root_dir +'/val' + Great_Dane)
os.makedirs(root_dir +'/test' + Great_Dane)

os.makedirs(root_dir +'/train' + pug)
os.makedirs(root_dir +'/val' + pug)
os.makedirs(root_dir +'/test' + pug)

os.makedirs(root_dir +'/train' + standard_poodle)
os.makedirs(root_dir +'/val' + standard_poodle)
os.makedirs(root_dir +'/test' + standard_poodle)

os.makedirs(root_dir +'/train' + Chow)
os.makedirs(root_dir +'/val' + Chow)
os.makedirs(root_dir +'/test' + Chow)



# Creating partitions of the data after shuffeling
for dog in lst:
    currentCls = dog
    src = root_dir+currentCls # Folder to copy images from
    allFileNames = os.listdir(src)
    np.random.shuffle(allFileNames)

    train_FileNames, val_FileNames, test_FileNames = np.array_split((allFileNames),
                                                            [int(len(allFileNames)*0.96), int(len(allFileNames)*0.96)])
    train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
    val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
    test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]
    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    print('Validation: ', len(val_FileNames))
    print('Testing: ', len(test_FileNames))
    # Copy-pasting images
    for name in train_FileNames:
        shutil.copy(name,  "../../images/Images/train"+currentCls)
    for name in val_FileNames:
        shutil.copy(name,  "../../images/Images/val"+currentCls)
    for name in test_FileNames:
        shutil.copy(name, "../../images/Images/test"+currentCls)



