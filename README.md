# Who's Your Doggy


## Motivation

Dogs have been artificially bred by humans over the years to highlight desirable work and pet qualities. Due to the multitude of purposes, dogs have become the most diverse species on the planet.

The day you bring home your furry ball of love from the animal shelter is typically exciting and full of joy, however, the return rate back to the shelter can vary between 5 and 20%. The most common reasons why a dog may be returned are unexpected costs, human health issues, and a variety unwanted behaviors. Doing some reasearch about the personality and behaviors you are looking for can help prepare you for your shelter visit, however, coming home with a miss classified mixed breed is highly likely due to dog breed diverstiy. 

Personally, classifying my own rez dog with more accurate breeds than the generic, "Australian Shepherd Mix", is my personal motivation for the beginning of the project. 

![picture](visuals/animal_imgs/rees_unknown.png)

## About the data:

This set of images comes from the well known <a href = "http://vision.stanford.edu/aditya86/ImageNetDogs/">Standford Dog dataset</a>



The data contains the following:

    120 dog breeds
    ~150 images per breed
    Total images: 20,580
   
Due to imense ammount of dog classifications, I started initially with only five random breeds and eventually utilized a total of 25 breeds with wich to work. 

The final 25 breeds I chose to classify are: 

![picture](visuals/animal_imgs/breeds.png)

## Machine Learning Modeling with CNN

### Image Data Generator for Training

Before training a model, we have to determine what information we want it to learn. Utilizing an ImageDataGenerator, I generated batches of augmented images that look similar to the following:

![picture](visuals/animal_imgs/chow.png)

Changing the aspects of the image gives the machine learning model more of an opportunity to find dynamic features within the dog that are not particular to the area that feature occures within the picture itself. 

### Baseline CNN

I began the project with a very basic Convolutional Nerual Network. 
The Architecture of this basic CNN involved a series of r convolutional, pooling and activation layers. To end the process, a flatten, dense, dropout and final activation layer to show the model to produce 25 different predictions, one for each breed. 

![picture](visuals/NN_viz/cnn_basic_.jpeg)

The Image below gives a quick exampl of how the modle reivews the input vectorized image to learn important features

![picture](visuals/NN_viz/cnn_vis.gif)

This model was never optimized and started off producing the following metrics. 

![picture](visuals/model_progress/basic_cnn_compare.png)

With focused optimization techniques, I can image this model eventually producing great metrics when visualizing the original 20 epochs. However, a quicker method to gaining better accuracy scores quicker is the move toward transfer learning. 


### Transfer Learning
Utilizing a mixtue of Transfer Learning Techniques, I settled on optimizing the model Xception.

![picture](visuals/model_progress/plus100_Xception290.png)
Xception:

Working further with the Xception model, here are some images give the layer. 

![picture](visuals/activation_one.png)

![picture](visuals/activation_ten.png)

![picture](visuals/activation_hund_thirty.png)




Noting that the validation data flattens out around 75-80% accuracy it was time to optimize. The following chart shows that the best model was trained with 53 Tranable features and a learning rate of 0.001. 

![picture](visuals/model_progress/model_improvements.png)



### Final Predictions

![picture](visuals/model_progress/confusion_mtx.png)




![picture](visuals/.png)



#### Determine Rees' Breeds

!['picture'](visuals/rees.png)

