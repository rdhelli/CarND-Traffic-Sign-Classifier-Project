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

* Ipython notebook with code: 
[Ipython notebook](https://github.com/rdhelli/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)
* HTML output of the code: 
[HTML output](https://github.com/rdhelli/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html)
* A writeup report (either pdf or markdown): 
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

#### 3.2. Model Architecture

My final model consisted of the following layers, with extending the LeNet architecture with some elements of the VGGNet architecture.

| Layer         		      |     Description	        					                 | 
|:---------------------:|:---------------------------------------------:| 
| Input         		      | 32×32×1 Grayscale image   							             |
| **Layer 1**           |                                               |
| Convolution 3×3       | 1×1 stride, same padding, outputs 32×32×64    |
| RELU                  |                                               |
| Max pooling           | 2×2, same padding, outputs 16×16×64           |
| **Layer 2**           |                                               |
| Convolution 3×3       | 1×1 stride, same padding, outputs 16×16×128   |
| RELU                  |                                               |
| Max pooling           | 2×2, same padding, outputs 8×8×128            |
| **Layer 3**           |                                               |
| Convolution 3×3       | 1×1 stride, same padding, outputs 8×8×256     |
| RELU                  |                                               |
| Dropout               |                                               |
| **Layer 4**           |                                               |
| Convolution 3×3       | 1×1 stride, same padding, outputs 8×8×256     |
| RELU                  |                                               |
| Max pooling           | 2×2, same padding, outputs 4×4×256            |
| Dropout               |                                               |
| **Layer 5**           |                                               |
| Flatten               | outputs 4096                                  |
| Fully connected	      | outputs 400                                   |
| RELU                  |                                               |
| Dropout               |                                               |
| **Layer 6**           |                                               |
| Fully connected       | outputs 200                                   |
| RELU                  |                                               |
| Dropout               |                                               |
| **Layer 7** (output)  |                                               |
| Fully connected       | outputs 43                                    |

#### 3.3. Model Training

To train the model, I used the AdamOptimizer, as a good alternative to the gradient descent optimizer. The default 0.001 learning rate was sufficient. Due to the additional layers and layer depths of the architecture overfitting needed to be avoided. I used L2 regularization with a regularization gain of 0.015 to avoid unnecessarily high weights and I have inserted several dropout layers with a 0.5 probability to keep calculated features. This has proved to be useful, as shown by the training and validation accuracies that had a much lower difference during training.

A note on the number of parameters to train:

| Layer        | number of parameters |
|:------------:|:--------------------:|
CONV1          | (3 * 3 * 1 + 1) * 64 = 640 parameters (0.02%)
CONV2          | (3 * 3 * 64 + 1) * 128 = 73856 parameters (2.75%)
CONV3          | (3 * 3 * 128 + 1) * 256 = 295168 parameters (10.98%)
CONV4          | (3 * 3 * 256 + 1) * 256 = 590080 parameters (21.96%)
FC1            | (4096 + 1) * 400 = 1638800 parameters (60.98%)
FC2            | (400 + 1) * 200 = 80200 parameters (2.98%)
FC3            | (200 + 1) * 43 = 8643 parameters (0.32%)
**Altogether** | 2687387 parameters

It can be concluded that the typical multilayered 3×3 structure of the VGGNet helped to keep the number of parameters to a sustainable amount, while providing enough depth on various feature levels. A critical choice in a convolutional model is often the connection of the last convolutional and the first fully connected layer. Without the maxpooling in between, the complexity would have increased to 3-4 times the current complexity.

The batch size of 256 made the training process fast enough with a gpu. I have trained the model for 20 epochs, in which the plateauing started around the 15th. I have saved the model with the highest validation accuracy of them.

![alt text][image10]

#### 3.4. Solution Approach

My final model results were:
* training set accuracy of 0.972
* validation set accuracy of 0.974
* test set accuracy of 0.933

An iterative approach was chosen:
* The first architecture was the LeNet-5 type architecture, which is a good starting point for classifying images containing shapes, symbols and letters.
* No matter the hyperparameters, the LeNet-t architecture just did not suffice for me. My understanding is that the complexity of traffic sign shapes requires greater depths, but that did not seem to improve the model enough, more layers were needed.
* Instead of just throwing in layer after layer and turning the knobs blindly, I did some research of various available architectures. I ended up with an architecture corresponding to VGGnet's structure, due to its similar building blocks, simplicity and its proven success in classifying images. The changing to uniform 3×3 filters in all convolutional layers brought a breakthrough in accuracy.
* I scaled the original model down to match the current problem (lower resolution pictures), by removing layers. At first, the architecture showed signs of overfitting (high training accuracy and low validation accuracy), but I have managed to address the problem by inserting several dropout layers and adding L2 regularization.
* I have experimented with the hyperparameters over the iterations, but the default values did well themselves. Tried more epochs but mostly it only caused more overfitting. For the data augmentation, I have ended up with the parameters according to the visual feedback - these amounts seemed natural and justified in comparison with the original dataset.
* Regarding the preprocessing, histogram equalization turned out to be fruitful for accuracy and for computational efficiency as well.
* Plotting a classification report gave a lot of insight as to what was going on under the hood. At this point future improvements would require focused problem solving. For example, much of the error was due to misclassification of classes "Dangerous Curve To The Right", , "Double Curves", "Traffic Signals", "Pedestrians" and "Beware of ice/snow". Their similarity is eye-catching: a vertical blob in a triangular frame. Colored channels might help the recognition of "Traffic Signals", sharpening the edges might help the rest. 

### 4. Test a Model on New Images

#### 4.1. Acquiring New Images

I went beyond and downloaded 1 additional example from the web for all the defined classes, 43 images altogether. I tried to include a variety of different qualities, although the process of finding exactly the same type of traffic sign images turned out to be a gruesome task. Many times the sizes and style of symbols are inconsistent with our dataset, or are just presented out of context.

Here are a few examples:

![alt text][image11]

And here are the examples after preprocessing:

![alt text][image12]

#### 4.2. Performance on New Images

Here are the results of the prediction:

| Id	| Result  |              Label               |         Prediction      |
|:--:|:-------:|:--------------------------------:|:-----------------------:| 
 0 |  True |                Speed limit (20km/h)  |  Speed limit (20km/h)
 1 | False |                Speed limit (30km/h)  |  Speed limit (20km/h)
 2 |  True |    No passing for vehicles over 3.5  |  No passing for vehicles over 3.5
 3 |  True |    Right-of-way at the next interse  |  Right-of-way at the next interse
 4 |  True |                       Priority road  |  Priority road
 5 |  True |                               Yield  |  Yield
 6 |  True |                                Stop  |  Stop
 7 |  True |                         No vehicles  |  No vehicles
 8 |  True |    Vehicles over 3.5 metric tons pr  |  Vehicles over 3.5 metric tons pr
 9 |  True |                            No entry  |  No entry
10 |  True |                     General caution  |  General caution
11 |  True |         Dangerous curve to the left  |  Dangerous curve to the left
12 |  True |                Speed limit (50km/h)  |  Speed limit (50km/h)
13 |  True |        Dangerous curve to the right  |  Dangerous curve to the right
14 |  True |                        Double curve  |  Double curve
15 |  True |                          Bumpy road  |  Bumpy road
16 |  True |                       Slippery road  |  Slippery road
17 |  True |           Road narrows on the right  |  Road narrows on the right
18 |  True |                           Road work  |  Road work
19 |  True |                     Traffic signals  |  Traffic signals
20 | False |                         Pedestrians  |  Road narrows on the right
21 |  True |                   Children crossing  |  Children crossing
22 | False |                   Bicycles crossing  |  Slippery road
23 |  True |                Speed limit (60km/h)  |  Speed limit (60km/h)
24 |  True |                  Beware of ice/snow  |  Beware of ice/snow
25 | False |               Wild animals crossing  |  Slippery road
26 | False |    End of all speed and passing lim  |  End of speed limit (80km/h)
27 |  True |                    Turn right ahead  |  Turn right ahead
28 | False |                     Turn left ahead  |  Speed limit (20km/h)
29 |  True |                          Ahead only  |  Ahead only
30 |  True |                Go straight or right  |  Go straight or right
31 |  True |                 Go straight or left  |  Go straight or left
32 |  True |                          Keep right  |  Keep right
33 |  True |                           Keep left  |  Keep left
34 |  True |                Speed limit (70km/h)  |  Speed limit (70km/h)
35 | False |                Roundabout mandatory  |  Keep left
36 |  True |                   End of no passing  |  End of no passing
37 | False |    End of no passing by vehicles ov  |  Roundabout mandatory
38 |  True |                Speed limit (80km/h)  |  Speed limit (80km/h)
39 | False |         End of speed limit (80km/h)  |  Children crossing
40 |  True |               Speed limit (100km/h)  |  Speed limit (100km/h)
41 | False |               Speed limit (120km/h)  |  Speed limit (80km/h)
42 | False |                          No passing  |  Vehicles over 3.5 metric tons pr

The accuracy on the new images is 0.744 which falls short compared to the 0.933 accuracy on the test set. The difference might be caused by the manual selection of the web images, as some of them might not live up to the german traffic sign standards. Moreover many of the images appear in front of a white background, which is also something not encountered in the training set.

#### 4.3. Model Certainty - Softmax Probabilities

Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

It is worthwhile to look at the specific softmax probabilities behind each prediction. Here are some interesting examples:

![alt text][image13]

12% probability on the matching label is a fairly good result here. Competing are some reasonably similar looking alternatives.

![alt text][image14]

12% for the yield sign, and all the other probabilities drop to 3%, the yield sign is easily recognizable due to its spectacular shape.

![alt text][image15]

12% is a failed prediction for pedestrian type, the competitors are almost all vertical blobs in a triangle.

![alt text][image16]

This image failed to trigger a significant response from any classes. 

![alt text][image17]

The speed limit predictions are almost accurate here, but the number 120 is missing. 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

If we plot the response of different layers to an example image, we might get more insight about how is a class recognized. Here is my model's response to the "End of all limits" sign.

![alt text][image18]

On the first layer responses, we can see that some neurons focused on the white areas of the sign, and some focused on the diagonal edges of the black band.

![alt text][image19]

On the second layer responses, the same aspects can be seen, more specifically zoomed in on some details.

![alt text][image20]

On the third layer, the same tendency continues with a lower and lower resolution, but greater depth.
