#**Behavioral Cloning** 

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model_summary.png "Model Visualization"
[image2]: ./examples/training_data_historgram.png "Data Visualization"
[image3]: ./examples/center_2016_12_01_13_31_13_279.jpg "center Image"
[image4]: ./examples/left_2017_02_11_17_23_18_497.jpg "Left Recovery Image"
[image5]: ./examples/right_2017_02_11_14_55_30_805.jpg "Right Recovery Image"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* model.json containing json file for the model configuration
* writeup_report.md summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.json
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model consists of a convolution neural network with 3x3 filter and depth of 32 currently. Even though my original Design had depths of 64, 32, 16 and 8. I could not train the network
on my CPU (Windows i7 2.6GHZ machine), as it was taking a lot of time to train. You can see that I have currently commented out lines of code 
The convolution layers with filter size of 64, 16 and 8 are commented as noted in lines (116, 129 and 133)

The model includes RELU layers to introduce nonlinearity (code line 123), and the data is normalized in the model using batch normalization (code line 111). 
Also, the model has 4 fully connected layers with density of 128, 64, 32 and finally 1 for the steering angle prediction
####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 127). 

The model was trained and validated on different datasets to ensure that the model was not overfitting (code line 86-91). The model was tested by running it through the simulator and the vehicle wears off the road after a while. I guess, this is because
I do not have enough training data
####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 160).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Inspite of this I feel I do not have
enough training data.
For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ensure that vehicle stays with lanes and not wear off the road.

My first step was to use a convolution neural network model similar to a previous keras lab that I had done. I thought this model might be appropriate because we are essentially a image classification problem but predicting a final steering angle 
instead of classifying images.
In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model add dropout layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. I am still improving the driving behavior in these cases by collecting
more recovery data.
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road for about 60% of the time.

####2. Final Model Architecture

The final model architecture (model.py lines 110-157) consisted of a convolution neural network with the following layers and layer sizes .
The convolutional layers have filters of 32 layer. The fully connected layers have output sizes of 128, 64, 32, and 1.

Here's the summary of my model

![model summary][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![center image][image3]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![left image][image4]
![right image][image5]

Also, here's the distribution of steering angle data vs count:
![training_data][image2]


Then I repeated this process on track two in order to get more data points.


After the collection process, I had X number of data points. I then preprocessed this data by reducing the image size (original size 320 x160) to 80x40.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
