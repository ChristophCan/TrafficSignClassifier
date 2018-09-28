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

[visualisation01]: ./readmePics/visualisation01.jpg
[visualisation02]: ./readmePics/visualisation02.jpg
[grayscale]: ./readmePics/grayscale.jpg
[traffic7]: ./RealSigns/Small/AheadOnly01_32.jpg
[traffic6]: ./RealSigns/Small/AnimalCrossing01_32.jpg
[traffic9]: ./RealSigns/Small/AnimalCrossing02_32.jpg
[traffic5]: ./RealSigns/Small/NoEntry01_32.jpg
[traffic8]: ./RealSigns/Small/NoEntry02_32.jpg
[traffic1]: ./RealSigns/Small/NoPassing01_32.jpg
[traffic2]: ./RealSigns/Small/PriorityRoad01_32.jpg
[traffic10]: ./RealSigns/Small/RoadWork01_32.jpg
[traffic3]: ./RealSigns/Small/Stop01_32.jpg
[traffic4]: ./RealSigns/Small/Yield01_32.jpg


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used simple `len`, `shape` and `numpy` functions to get information about the dataset.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. I first printed a random sample of 20 training pictures with it´s corresponding labels. I just wanted to get a visual feeling for the data I am working with.

![sampleImages][visualisation01]

I then generated a histogram over the amounts of images per traffic sign class. You can see, that there are up to twice as many pictures for the lower class numbers, than there are for the higher ones. I propose this means, that these kind of signs will get a much higher recognition in the end. You also can see, that the relative distribution of all the classes in the training, the validation and the testing set are comparable.

![histogram][visualisation02]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

For the preprocessing I only did two steps, which are a simple gray scaling, followed by a normalization. The gray scaling had a huge effect on the final accuracy of the network. It seems, that the color information isn´t really helpful for the CNN and is so better neglected.

Here is an example of a raw-image, followed by the gray scaled version. You can also see, that the contrast of the image got enhanced, which will also help training the CNN.

![grayscale][grayscale]

At first glance, the normalization step `(gray_image - 128.) / 128.` doesn´t have any impact on the final image. But looking at the minimum and maximum color values of the image show, that the original has a maximum of 120.87 / minimum of 17.11, while the normalized version has a maximum of -0.06 / minimum of -0.87.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input      		| 32x32x3 RGB image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Activation: tanh					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Normalization   |  bias=1.0, alpha=0.001 / 9.0, beta=0.75 |
| Convolution 5x5	    |       1x1 stride, valid padding, outputs 10x10x16		|
| Activation: tanh					|												|
| Max pooling   |  2x2 stride,  outputs 5x5x16 |
| Normalization   |  bias=1.0, alpha=0.001 / 9.0, beta=0.75 |
| Flatten   | outputs 400   |
| Fully connected   | outputs 120  |
| Activation: sigmoid   |   |
| Fully connected   | outputs 84  |
| Activation: sigmoid   |   |
| Fully connected   | outputs 43  |
| Output   | 43  |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used the following parameters for training:
- Epochs = 12
- Batch Size = 128
- Learning Rate = 0.001

I slightly increased the amount of Epochs and left the remaining parameters like they were in the LeNet of the previous lessons. A further increase of Epochs doesn´t lead to higher results, it leads to overfitting.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.996
* validation set accuracy of 0.943
* test set accuracy of 0.915

I had to be very careful to not overfit my CNN even more. It already has a very high training accuracy, compared to the lower validation and test accuracies. I stayed with it for now, as the results for the new images, as will be discussed below, was satisfying.

I mainly stayed with the already discussed LeNet architecture. My main adjustment was the addition of two normalisation layers with `tf.nn.lrn`. I read this approach in the Tensor Flow beginners tutorial and used the hyperparameters given, as the lead to a good result. For further improving I might still play a little with them.

I also changed the activation functions for the first two 5x5 convolution layers to `tanh`. I tried all the different functions mentioned on the Tensor Flow documentation, this one showed the best results for me. I guess it´s often always a little try and error in optimizing a CNN?!

For activation of the fully connected layers in the end, I chose the `sigmoid` functions. Same approach as for the `tanh`, it just lead to the best results!

I also tried to implement bypassing the first convolution and pooling layer beside the second ones, and merging them back together before the flattening. I understood this was the approach done in the paper you mentioned as a possible resource. I have to admit, that I didn´t fully understand how to implement this and will have to try doing that in the future. Am I on the right way, or totally mistaken?

Another improvment that I want to try is adding a dropout layer. I guess it might prevent my model from being overfitted, as it makes it more robust and less dependent on special features of my training set.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![AheadOnly][traffic1] ![AheadOnly][traffic2] ![AheadOnly][traffic3] ![AheadOnly][traffic4] ![AheadOnly][traffic5]
![AheadOnly][traffic6] ![AheadOnly][traffic7] ![AheadOnly][traffic8] ![AheadOnly][traffic9] ![AheadOnly][traffic10]

It was surprisingly hard to find traffic sign images in real world scenarios, so most of the ones I found are pretty clearly visible and I don´t expect them to be hard classifying. The second (Animal crossing) and the sixth (No passing) are the only ones, which are slightly darker and less contrasty. So I am curious how they will be handled.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|Result|
|:---------------------:|:---------------------------------------------:|:------:|
| No Passing      		| No Passing   									| Correct|
| Priority Road     			| Priority Road 										| Correct|
| Stop					| Stop											|Correct|
| Yield	      		| Yield					 				|Correct|
| No Entry			| No Entry      							|Correct|
|Animal Crossing   |  Double curve |Not Correct|
|Ahead Only   | Ahead Only  |Correct|
|No Entry   |  Turn Left Ahead |Not Correct|
|Animal Crossing   |  Animal Crossing  |Correct|
|Road Work   |  Road Work |Correct|


The model was able to correctly guess 8 of the 10 traffic signs, which gives an accuracy of 80%. This is about 10% lower than the test set and 15% to 20% lower than the training and validation set. This also might be a sign of CNN overfitting. Another possibility is, that the signs I chose are the ones with have fewer training images and so are not trained and recognized as good as the other ones.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The following table shows the top three probabilities for each traffic sign.

|Image|Prediction|Probability|
|:-----:|:-----:|:-----:|
|No Passing   | No Passing  | 0.99   |
|   | Children Crossing  | 0.0005  |
|   | Priority Road  | 0.0003  |
|Priority Road   | Priority Road   | 0.99  |
|   | No Passing  | 0.0014  |
|   | Roundabout mandatory  | 0.0003  |
| Stop   | Stop  | 0.86  |
|   |  Bumpy road |0.036   |
|   |  Go straight or right |0.036   |
| Yield   | Yield  |0.99   |
|   |No vehicles   |0.0009   |
|   | Priority road  |  0.0008 |
| No Entry   | No Entry   | 0.96  |
|   | Stop  | 0.03  |
|   | Keep right  | 0.003   |
|Animal Crossing   | Double curve  |  0.33 |
|   |  Right-of-way at the next intersection | 0.25  |
|   |  Road work | 0.22  |
|Ahead Only   |Ahead only   |  0.99 |
|   |Bicycles crossing   |  0.0007 |
|   |No passing   |  0.0003 |
|No Entry   | Turn left ahead   |  0.49 |
|   | Priority road  | 0.14  |
|   | Turn right ahead  | 0.09  |
|Animal Crossing   |Animal crossing   | 0.87  |
|   |Double curve   |  0.08 |
|   |Dangerous curve to the left   |  0.01 |
| Road Work  | Road Work  | 0.99  |
|   |Dangerous curve to the right   | 0.0005  |
|   |Bicycles crossing   |  0.0004 |

It´s very interesting to see, that the correctly recognized images always have a very high probability bigger than 85% for the first image, followed by relatively low probabilities for the second and third. The wrongly recognized images already start with very low probabilites for the first image. Maybe that´s another sign, that some traffic sings have to be better trained, either by more examples or a better CNN structure.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

I am sorry, I didn´t have the chance to finish this exercise! I hope I will do it some time later in the class!
