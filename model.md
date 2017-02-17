#**Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[samplingImage]: ./images/sampling.png "Sampling"
[balancingImage]: ./images/balancing.png "Balancing"
[flippingImage]: ./images/flipping.png "Flipping"
[translationImage]: ./images/translation.png "Translation"
[flippingImage1]: ./images/flipping1.png "Flipped Image"
[flippingImage2]: ./images/flipping2.png "Flipped Image"
[translationImage1]: ./images/translation1.png "Translated Image"
[translationImage2]: ./images/translation2.png "Translated Image"
[translationImage3]: ./images/translation3.png "Translated Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I have adopted the NVIDIA model with minor changes. It consists of 5 convolution layers followed by 4 fully-connected layers. 

The first 3 convolution layers use 5x5 filters with stride value = 2, and the other 2 layers use 3x3 filters with regular stride value = 1. All convolution layers are activated by a RELU function to introduce non-linearity (model.py line XXX).

The first 3 dense layers are activated with RELU functions, while the last output layer is activated by a linear function. 

The input data is normalised and cropped using initial Keras layers (model.py line XXX).


####2. Attempts to reduce overfitting in the model

The convolution layers are followed by dropout layers in order to reduce overfitting (model.py line XXX). 

The dataset was balanced and augmented to ensure that the model was not biased towards low steering angles and also generalised to extreme positions (model.py line XXX). The model was tested by running it through the simulator and ensuring that the vehicle could stay on both tracks.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line XXX).

####4. Appropriate training data

I did not explicitly train using the simulator. Instead, I chose to use the udacity dataset as a starting point and heavily augmented it to ensure a rich and varied dataset.

The training data augmentation is explained in detail below:

###Model Architecture and Training Strategy

####1. Solution Design Approach

I started out with a simple convolutional neural network model because I thought it would be appropriate for a regression problem based on visual data. The initial results were discouraging as the predictions were heavily biased towards zero steering angle.

After initial balancing, the mse metrics were encouraging, but it was clear that even if the mean-square errors were small, they could accumulate and eventually push the car to a position the model would not know about and therefore be unable to recover.

The next major part was augmenting a well-balanced dataset that would allow the model to see enough angles of the road so that it could recover from error accumulation.

The simple network model was not sufficient to accommodate the increased data complexity and resorted to constant predictions. Therefore, I switched to the well-known nvidia model with minor modifications.

The mse on training data was low but validation suffered so I added dropouts after the convolution layers to prevent overfitting. It was difficult to see how the training was improving with changes in parameters, so I plotted the training labels against training predictions to see how thin I could make the shape to be. Also, this gave a sense of the extreme values the model is capable of predicting.

I intermittently drove the simulator to ensure that the theoretical results had practical benefits. At one point, I decided to use all the color channels when the car mistook dirt for road.

At the end of the process, the vehicle was able to drive successfully on both tracks (although, on the second track, it needed to go with low throttle)

####2. Final Model Architecture

Here is a summary of the final model architecture:
```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
cropping2d_10 (Cropping2D)       (None, 90, 320, 3)    0           cropping2d_input_10[0][0]        
____________________________________________________________________________________________________
lambda_7 (Lambda)                (None, 90, 320, 3)    0           cropping2d_10[0][0]              
____________________________________________________________________________________________________
c1 (Convolution2D)               (None, 43, 158, 24)   1824        lambda_7[0][0]                   
____________________________________________________________________________________________________
dropout_48 (Dropout)             (None, 43, 158, 24)   0           c1[0][0]                         
____________________________________________________________________________________________________
c2 (Convolution2D)               (None, 20, 77, 36)    21636       dropout_48[0][0]                 
____________________________________________________________________________________________________
dropout_49 (Dropout)             (None, 20, 77, 36)    0           c2[0][0]                         
____________________________________________________________________________________________________
c3 (Convolution2D)               (None, 8, 37, 48)     43248       dropout_49[0][0]                 
____________________________________________________________________________________________________
dropout_50 (Dropout)             (None, 8, 37, 48)     0           c3[0][0]                         
____________________________________________________________________________________________________
c4 (Convolution2D)               (None, 6, 35, 64)     27712       dropout_50[0][0]                 
____________________________________________________________________________________________________
dropout_51 (Dropout)             (None, 6, 35, 64)     0           c4[0][0]                         
____________________________________________________________________________________________________
c5 (Convolution2D)               (None, 4, 33, 64)     36928       dropout_51[0][0]                 
____________________________________________________________________________________________________
dropout_52 (Dropout)             (None, 4, 33, 64)     0           c5[0][0]                         
____________________________________________________________________________________________________
flatten_10 (Flatten)             (None, 8448)          0           dropout_52[0][0]                 
____________________________________________________________________________________________________
d1 (Dense)                       (None, 100)           844900      flatten_10[0][0]                 
____________________________________________________________________________________________________
d2 (Dense)                       (None, 50)            5050        d1[0][0]                         
____________________________________________________________________________________________________
d3 (Dense)                       (None, 10)            510         d2[0][0]                         
____________________________________________________________________________________________________
out (Dense)                      (None, 1)             11          d3[0][0]                         
====================================================================================================
Total params: 981,819
Trainable params: 981,819
Non-trainable params: 0
____________________________________________________________________________________________________
Train on 29788 samples, validate on 1568 samples
Epoch 1/5
958s - loss: 0.0423 - val_loss: 0.0355
Epoch 2/5
736s - loss: 0.0361 - val_loss: 0.0345
Epoch 3/5
572s - loss: 0.0349 - val_loss: 0.0340
Epoch 4/5
575s - loss: 0.0342 - val_loss: 0.0330
Epoch 5/5
608s - loss: 0.0337 - val_loss: 0.0334
```

####3. Creation of the Training Set & Training Process

Sampling - picking a subset of the low steering angle data

![alt text][samplingImage]

Balancing - including the left camera sample (add offset to steering angle) and right camera sample (subtract offset to steering angle)

![alt text][balancingImage]

Flipping - for a selection of extreme angles, flip the image and use the negative of the corresponding steering angle

![alt text][flippingImage]

Translation - variety of x and y translation to account for extreme road positions and vertical climbs

![alt text][translationImage]

Flipping can provide examples of driving on both sides of the road

![alt text][flippingImage1]

Translation can provide interesting examples from pre-existing data

![alt text][translationImage1]

![alt text][translationImage2]

![alt text][translationImage3]

After augmentation, I had 31356 samples, which were converted to YUV colorspace. These would get cropped and normalised from within the eras layers.

I finally randomly shuffled the data set and put 5% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3-5 because the mse metric did not seem to improve after that. I used an adam optimizer so that manually training the learning rate wasn't necessary.
