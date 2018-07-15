# **Traffic Sign Recognition** 

## Writeup

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

[image1]: ./content/untitled00.png "Distribution of train data"
[image2]: ./content/untitled01.png "Examples of train data"
[image3]: ./content/untitled02.png "Distribution of validation data"
[image4]: ./content/untitled03.png "Examples of validation data"
[image5]: ./content/untitled04.png "Distribution of test data"
[image6]: ./content/untitled05.png "Examples of test data"
[image7]: ./content/untitled06.png "Image augmentation"
[image8]: ./content/untitled07.png "Distribution after augmentation"
[image9]: ./content/untitled08.png "Preprocessing"
[image10]: ./content/untitled09.png "Training curve"
[image11]: ./content/untitled10.png "Internet images"
[image12]: ./content/untitled11.png "Internet images preprocessed"
[image13]: ./content/untitled12.png "Top5: 20km/h speed limit"
[image14]: ./content/untitled13.png "Top5: yield"
[image15]: ./content/untitled14.png "Top5: pedestrians"
[image16]: ./content/untitled15.png "Top5: end of 80km/h speed limit"
[image17]: ./content/untitled16.png "Top5: 120km/h speed limit"
[image18]: ./content/untitled17.png "Layer 1 visualization"
[image19]: ./content/untitled18.png "Layer 2 visualization"
[image20]: ./content/untitled19.png "Layer 3 visualization"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### 1. Files submitted

#### 1.1 Submission files

* Ipython notebook with code
[Ipython notebook](https://github.com/rdhelli/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)
* HTML output of the code
[HTML output](https://github.com/rdhelli/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html)
* A writeup report (either pdf or markdown)
[this writeup report](https://github.com/rdhelli/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup.md)

### 2. Dataset Exploration

#### 2.1 Dataset Summary

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2.2 Exploratory Visualization

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed among the 43 image classes.

![alt text][image1]

What I have learned is that the underrepresented classes with training data lower than about 1000 images need special attention or else the model will predict images of those classes with a fairly low accuracy. 

Here are example images from the training data.

![alt text][image2]

The validation set and the test set distributions follow a similar pattern and similar examples.

### 3 Design and Test a Model Architecture

#### 3.1. Preprocessing

Since the above mentioned distribution leads to problems, I decided to augment the dataset. The augmented data was generated from only those classes that were underrepresented, by rotating, shifting, shearing and zooming into the images. For this process I used the built-in data generator of keras.

![alt text][image7]

As seen on the bar graph, the distribution of images is much more even after my augmentation.

* The size of the augmentation set is 28738
* The size of the augmented training set is thus 63537

![alt text][image8]

As for preprocessing, I decided to apply histogram equalization. It converts the image into grayscale, but the classification depends mostly on the shapes of the images. The histogram equalization unfolds the local differences in intensity, helping the network to identify the edges and shapes more easily.

![alt text][image9]

As a last step, I normalized the image data so that at any point, multiple sources and types of input information could be easily added to the model. Training with a higher range of variables is also discouraged because it increases the significance of computational errors.

#### 3.2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3.3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 3.4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


