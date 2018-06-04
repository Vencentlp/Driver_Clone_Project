# **Behavioral Cloning** 

## Writeup Report

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

 

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 64 (model.py lines 89-101) 

The model includes RELU layers to introduce nonlinearity (code line 92-96), and the data is normalized in the model using a Keras lambda layer (code line 90). 

#### 2. Attempts to reduce overfitting in the model

The model did not consist of dropout layers because the overfitting can be improved by data processing.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 49-77). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 105).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving counter clockwise, driving around curves and driving on both tracks.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use multipul layer of convolutional layers.

My first step was to use a convolution neural network model similar to the achitecture from NVIDIA. I thought this model might be appropriate because it has more convolutional layers and more powerful.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I use more data (including data augment and drive on both tracks) and shuffle the data each step of training. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track especially when it needs to turn to improve the driving behavior in these cases, I collect more data with recovering driving from left and right side and I augmented the data by flipping the images and turning angles.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 89-101) consisted of a convolution neural network with the following layers and layer sizes 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   							| 
| Lambda         		| Normalize data    							| 
| Cropping         		| Delete unused data to spped up the training process|   							|
| Convolution 5x5     	| 2x2 stride, 24 filters, relu activation 	|
| Convolution 5x5	    | 2x2 stride, 36filters, relu activation 	|
| Convolution 5x5	    | 2x2 stride, 48filters, relu activation 	|
| Convolution 5x5	    | 1x1 stride, 64filters, relu activation 	|
| Convolution 5x5	    | 1x1 stride, 64filters, relu activation 	|										|
| Fully connected		| outputs 100  									|
| Fully connected		| outputs 50								|
| Fully connected		| outputs 10  									|
| Fully connected		| outputs 1  									|



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][https://github.com/Vencentlp/Driver_Clone_Project/raw/master/center_2016_12_01_13_30_48_287.jpg]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn how to drive back to the center of the road when the car is on the side of the road. These images show what a recovery looks like starting from left side to the center of the road :

![alt text][https://github.com/Vencentlp/Driver_Clone_Project/raw/master/recovery/center_2018_06_01_09_58_00_587.jpg]
![alt text][https://github.com/Vencentlp/Driver_Clone_Project/raw/master/recovery/center_2018_06_01_09_58_00_656.jpg]
![alt text][https://github.com/Vencentlp/Driver_Clone_Project/raw/master/recovery/center_2018_06_01_09_58_00_725.jpg]
![alt text][https://github.com/Vencentlp/Driver_Clone_Project/raw/master/recovery/center_2018_06_01_09_58_00_793.jpg]
![alt text][https://github.com/Vencentlp/Driver_Clone_Project/raw/master/recovery/center_2018_06_01_09_58_00_863.jpg]
![alt text][https://github.com/Vencentlp/Driver_Clone_Project/raw/master/recovery/center_2018_06_01_09_58_01_070.jpg]
![alt text][https://github.com/Vencentlp/Driver_Clone_Project/raw/master/recovery/center_2018_06_01_09_58_01_210.jpg]
![alt text][https://github.com/Vencentlp/Driver_Clone_Project/raw/master/recovery/center_2018_06_01_09_58_01_348.jpg]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would add more data to train the model turn in the opposite direction and improve the overfitting.  For example, here is an image that has then been flipped:

![alt text][https://github.com/Vencentlp/Driver_Clone_Project/raw/master/flip_images/flip_image.jpg]
![alt text][https://github.com/Vencentlp/Driver_Clone_Project/raw/master/flip_images/flip_origin.jpg]

Etc ....

After the collection process, I had 32758 number of data points. I then preprocessed this data by normalization and generating batches of data.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by rising of validation error when epochs is over 3. I used an adam optimizer so that manually training the learning rate wasn't necessary.
