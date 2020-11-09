# Content-Based-Image-Retrievel-using-VGG-in-Keras
The script to get similar images provided the reference image

In CBIR we are trying to fetch similar images from are database based on a query image.
This is a less qccurate demo version of what google have implemented in there google image
search. Here the user provides an image and the related result is given as output.
We can send the number of outputs we want. 
In this technique we are using VGG16 pretrained model to implement CBIR. The specify the 
search according to color, texture or shape we can use different techniques in the
place of VGG16 to do that.
This script was made to work on a server so no GUI is here. You have to run the script using IDE.
This model was trained on caltex256 dataset which can be downloaded from the given link.

# Requirements
- h5py
- OpenCv
- Keras
# Weights
You can download mine feature weights for Caltech256 dataset. If you want to train your own you can do it by using index.php

https://drive.google.com/file/d/1fJFvxJQx6fkN5Z6KR8TZmbp7lWLMPRtW/view?usp=sharing
# Dataset
Download it from the link
https://www.kaggle.com/jessicali9530/caltech256

Unzip the dataset as
- caltex256
- - 1.jpg
- - 2.jpg
- - 3.jpg


# How does it work
extract_cnn_vgg16_keras.py contains the class which initiate the VGG16 model and predicts using the model.
THe first file to run is index.php. This file uses the VGG16 model to predict the feature vector of all the caltex256 dataset images.
The file than saves the feature vector of each image with its name in a H5 file for use while quering.
Place the download .h5 file and place it with the index.php file.


query.php is used to query the database with respect to an image. We can change the path of image we want to query inside query.php.

```python
queryDir = 'database/caltex256/006_0001.jpg'
```

What this file do is to first retrieve the H5 file we have made. Than we pass the query image to our VGG16 model and gets a feature vector out of it.
Now we have the feature vector of our query image and all the image in our database. The next step is just to calculate a similaraity score and outputs the 
top n results. We can customize the number of search results.

```python
maxres = 10
```
We can use different similarity score link cosine similarity, logarithimic and others here. It depends on the requirement and the dataset we have.

# Input

# Output


