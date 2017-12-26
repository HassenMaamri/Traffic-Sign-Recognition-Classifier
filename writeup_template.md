# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Training_Set_Distribution.png "Training_Set"
[image2]: ./examples/random.png "randomFive"
[image3]: ./examples/predictions.JPG "predictions"
[image4]: ./examples/softmax.JPG "softmax"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the training set. It is a bar chart showing how the data is distributed

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

For Multi-Scale Convolutional Networks, converting images to YUV color space, in this cas the traffic signs, helped considerably.
Experimentally, performing YUV conversion performed very well.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I used the LeNet architecture for my final model, so this is how my layers are:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| k=2, 2x2 stride, SAME padding, outputs 14x14x16 				|
| ReLU	    |       									|
| Convolution 5x5		|1x1 stride, VALID padding, outputs 10x10x16         									|
| ReLU				|         									|
| Max pooling	      	| k=2, 2x2 stride, SAME padding, outputs 5x5x16 				|
| Flatten	      	| outputs 400 				|
| Fully connected	      	| outputs 120 				|
| ReLU				|         									|
| Fully connected	      	| outputs 84 				|
| ReLU				|         									|
| Fully connected	      	| outputs 43 				|


This ended me up with these values:

  Training set accuracy = 0.997
Validation set accuracy = 0.943
      Test set accuracy = 0.937


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I started of an epochs value of 10, it didn't work well so I tried 50. 50 was good but not enough to get the accuracy needed so I made it to 80.

To train the model, I used AdamOptimizer with a learning rate of 0.001. For training I used dropout on each layer with a keep_prob value of 0.9.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

LeNet architecture was adopted here. Since we are working with images for traffic sign classification, LeNet worked perfectly.
However, we are using images with color here so I had to change the depth from 1 to 3.
One more little change I did was modifying the output length from 10 to 43 to suit the number of our classes.

LeNet with its convolutional network architecture is a great choice for an image classification and recognition problem.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image2]

My classifier found it easy to get a 100% accuracy on these images.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image3]

The model was able to correctly guess 5 out of the 5 traffic signs, which gives an accuracy of 100%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

![alt text][image4]
