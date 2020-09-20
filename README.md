# Who's Your Doggy
![picture](visuals/dogs.jpeg)


## Motivation

Dogs have been artificially bred by humans over the years to highlight desirable work and pet qualities. Due to the multitude of purposes, dogs have become the most diverse species on the planet.

The day you bring home your furry ball of love from the animal shelter is typically exciting and full of joy, however, the return rate back to the shelter can vary between 5 and 20%. The most common reasons why a dog may be returned are unexpected costs, human health issues, and a variety unwanted behaviors. Doing some reasearch about the personality and behaviors you are looking for can help prepare you for your shelter visit, however, coming home with a miss classified mixed breed is highly likely due to dog breed diverstiy. 

Personally, classifying my own rez dog with more accurate breeds than the generic, "Australian Shepherd Mix", is my personal motivation for the beginning of the project. 

![picture](visuals/rees_unknown.png)

## About the data:

This set of images comes from the well known <a href = "http://vision.stanford.edu/aditya86/ImageNetDogs/">Standford Dog dataset</a>



The data contains the following:

    120 dog breeds
    ~150 images per breed
    Total images: 20,580
   
Due to imense ammount of dog classifications, I started initially with only five random breeds and eventually utilized a total of 25 breeds with wich to work. 

The final 25 breeds I chose to classify are: 

![picture](visuals/breeds.png)

## Machine Learning Modeling with CNN

### Image Data Generator for Training

Before training a model, we have to determine what information we want it to learn. Utilizing an ImageDataGenerator, I generated batches of augmented images that look similar to the following:

![picture](visuals/chow.png)

Changing the aspects of the image gives the machine learning model more of an opportunity to find dynamic features within the dog that are not particular to the area that feature occures within the picture itself. 

### Baseline CNN
Layer 0 | Name: conv2d | Trainable: True
Layer 1 | Name: activation | Trainable: True
Layer 2 | Name: max_pooling2d | Trainable: True
Layer 3 | Name: conv2d_1 | Trainable: True
Layer 4 | Name: activation_1 | Trainable: True
Layer 5 | Name: max_pooling2d_1 | Trainable: True
Layer 6 | Name: conv2d_2 | Trainable: True
Layer 7 | Name: activation_2 | Trainable: True
Layer 8 | Name: max_pooling2d_2 | Trainable: True
Layer 9 | Name: flatten | Trainable: True
Layer 10 | Name: dense | Trainable: True
Layer 11 | Name: activation_3 | Trainable: True
Layer 12 | Name: dropout | Trainable: True
Layer 13 | Name: dense_1 | Trainable: True
Layer 14 | Name: activation_4 | Trainable: True



### Transfer Learning
Xception:


![picture](visuals/cnn_vis.gif)

![picture](visuals/activation_one.png)

![picture](visuals/activation_ten.png)

![picture](visuals/activation_hund_thirty.png)


#### Final Optimized Xception
![picture](visuals/model_improvements.png)



Model Arcitecture 

![picture](visuals/wrong_pred.png)



#### Determine Rees' Breeds

!['picture'](visuals/rees.png)

