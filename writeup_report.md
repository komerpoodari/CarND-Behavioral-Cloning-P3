# **Behavioral Cloning Project Report** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.jpg "Model Visualization"
[image2]: ./examples/center-image.jpg "Center Lane Driving: Raw & Cropped"
[image3]: ./examples/recover_to_left.jpg "Recovery Image"
[image4]: ./examples/recover_to_right.jpg "Recovery Image"
[image5]: ./examples/normal.jpg "Normal Image"
[image6]: ./examples/flipped.jpg "Flipped Image"
[image7]: https://www.youtube.com/watch?v=8DjwK2Dzh5o

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.ipynb containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.ipynb file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network similar to the one suggested in NVDIA paper '**End to End Learning for Self-Driving Cars** and blog (https://devblogs.nvidia.com/deep-learning-self-driving-cars/). The initial stages are convolution layers with 'elu' activation functions, followed by Dense layers wih linear activation functions.

The input data were cropped in the beginning to avoid irrelevant data such as trees and sky at the top and steering wheel and hood at the bottom. 

Then the data were normalized in the model using a Keras lambda layer.  
By using Keras primitives to crop and normalize input, **drive.py** does not need any changes for image normalization and cropping.

I employed 'elu' activation instead of 'relu' to ensure negative values since steering angle values range is -1.0  to 1.0. 

The model architecture is implemented in section **17. Define Model Architecture** of model.ipynb. 


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting as implemented in section **17. Define Model Architecture** of model.ipynb.
Also, the number of epochs used are 7 to avoid overfitting. 
I used data over multiple runs with augmentation by using weights learned in previous training session as baseline.

The model was trained and validated on different data sets to ensure that the model was not overfitting (section **4. Load Image Paths and Angle Data in the lists for visualization & Analysis**). 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (**17. Define Model Architecture**).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I also trained the model with segments containing extreme turnings /bends before the bridge and after the bridge.

The raw data collected includes significant number of images with steering angle closer to zero. I ensured to put a threshold on the images closer to zero angle (section **7. Functions to Select and Load Training Data**, function **select_training_data**).

Then I augmented the data with flipped images so I can avoid, left leaning bias in the training data (section **11. Generate Augmented Training Data with Flipped Images**).

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to to reduce the mean sqaure error (mse), as this is regresson-like problem.

My first step was to use a convolution neural network model similar to the one suggested by NVDIA. I thought this model might be appropriate because early stage convolution layers handle the localized and translated features to a reasonable extent. The later dense layers derive the steering angle with linear activation functions.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I updated the model to include dropout and reduced the number of epochs.  


Then I used multiple sets of data including Udacity supplied data and data that I collected over multiple laps at different times. I saved the weights and used as starting point in subsequent training sessions.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track especially right before the bridge and right after the bridge. I collected some recovery data starting before the left turn before the bridge and until after the crossing the bridge.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.ipynb, section **17. Define Model Architecture** and section **18. Compile, Train and Save the model**) consisted of a convolution neural network with the following layers and layer sizes.

Here is a visualization of the architecture information in detail.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving, with raw image collected by simulator and the cropped image equivalent to cropping operation done in the model.

![alt text][image2]

I then recorded the vehicle recovering from the right side of the road back to center so that the vehicle would learn to bring the vehicle back to the center. Here is an example.
![alt text][image3]




Similarly an example of recovering from left to the center.
![alt text][image4]


Then I repeated this process on track two in order to get more data points.  It was a bit difficult to collect the data this way.

To augment the data sat, I also flipped images and angles thinking that this would avoid left side bias induction, because the nature of the track. For example, here is an image and the flipped version:

![alt text][image5]
![alt text][image6]


After the collection process, I had 20,000 to 25,000 number of data points, for each training session. I then preprocessed this data by in Keras model by cropping to avoid the irrelevant data feed at the top (60 rows, containing tries and horizon) and at the bottom (20 rows, containing steering wheel or hood).  

Then I normalized the image pixel data with mean 0 and range -0.5 to 0.5, using lambda primitive.


I finally randomly shuffled the data set and put 20% of the data into a validation set, as part of model.fit (in model.ipynb, section **18. Compile, Train and Save the model**). 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as evidenced by multiple runs. I used an adam optimizer. I did not specifically tune learning rate.

The following is the video link of the autonomous run (https://www.youtube.com/watch?v=8DjwK2Dzh5o)
![alt text][image7]

The following are my takeaways.
1. It's a good exercise, training was a bit time consuming and quite a few parameters / aspects to tune and select.
2. The speed element is missing in the exercise, whereas in the realworld, speed is an input on how steering wheel need to be turned.