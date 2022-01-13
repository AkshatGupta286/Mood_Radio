# Mood Radio

## Overview
In this work, we have created a system which recommends songs to people based on their mood. The system is divided into two parts.  
First, the emotions or mood of the person is recognized from the facial expressions. For that, a deep learning model is trained on FER2013 dataset of Microsoft. For now, the model is trained to predict only 3 emotions i.e. Happy, Angry, and Sad.  
Second, a dataset of songs is created and the mood of the songs is predicted using the audio features. For that, the Spotify API of python is used to find the audio features of the song and then a deep learning model is trained to predict emotion of the song. For now, the model is trained to predict only 2 emotions i.e. Happy, and Sad.

## Installation
To use this repo, create a conda environment using ```environment.yml```

```
# from environment.yml
conda env create -f environment.yml
```

## Facial Emotion Recgonition

### Preprocessing and Augmentation

From FER 2013 dataset, 12832 images of 3 emotions angry, happy, and sad are taken. For these images, the augmentation is done so that any image which might be in any form can be converted to a standard size of 48x48 pixel, rescaling the images up to ± 20% of its original scale, horizontally and vertically shifting the image by up to ± 10% of its size, and rotating range up to ± 180 degrees. 20% of the images are reserved for the validation purpose.  

### Training

The data is compiled over the modified VGG 16 architecture. In here, the test accuracy was approximately 0.5%, whereas the loss was approximately 0.80 for 60 epochs. The compilation is done using the Adam optimizer with learning rate of 0.0001.

### Testing

The remaining 20% of dataset, test-set, was used for testing. The evaluation metric used for Facial Emotion Recognition was categorical cross entropy loss and accuracy. Since the model categorizes multiple labels, it is ideal to use a metric which evaluates appropriately and categorical cross entropy loss. Apart from this, accuracy was also calculated as it gives a clearer understanding of the model performance.

## Music Mood Detection

### Preprocessing

By executing feature learning, we dropped the features from 30 to 7 namely ‘danceability’, ‘energy’, ‘loudness’, ‘valence’, ‘acousticness’, ‘instrumentalness’, ‘tempo’ to avoid any overfitting.  
The Spotify Dataset is not labelled, so labelling is done using the K-mean unsupervised learning method. This KMeans algorithm is used only once for this dataset. Future song inputs for mood detection will not pass through the KMeans model. KMeans is applied so that we can quantify the results from our mood classifier, the shallow neural network. While testing the results from the KMeans algorithm, no such outliers were found. The songs were accurately labelled into ‘happy’ and ‘sad’.  
Since we are categorizing the data into 2 labels, the number of clusters given as input to the model was 2. KMeans model used the ‘k-means++’ algorithm to initialize the centroids. The algorithm uses the tolerance level and maximum iterations as parameters to stop calculating. The tolerance level is set at 0.00001 and the maximum iteration is set at 300.  
Once labeled, the count was happy and sad was found to be heavily imbalanced. Passing this data into the neural network will instigate a bias towards the larger count data. To prevent this, random over sampling was implemented on the dataset. This algorithm identifies the difference in count between the two labels and makes random copies of data from the lower count set to ensure equal number of datapoints in both labels.

### Training

For Music Mood Detection, 4316 songs for training and, 1080 test songs were used. We have trained and tested the dataset on a double hidden layer Neural Network, i.e., a shallow Neural Network. The model has an input size of 7 and passes through 2 hidden layers which has a size of 16 and 8 neurons in each layer. Th output layer has 2 neurons. ReLU activation function is used for the hidden layer and Sigmoid activation function for the output layer neural network.

### Testing

The 1080 songs were used for testing the classification results from the shallow neural network. To test the network, precision and recall was used. These metrics helps us accurately measure how well the model is predicting the “happy” and “sad” labels.

## Demo

A final script (Create Playlist.ipynb) was written to combine FER and MMD to help any user play a mood appropriate song from his playlist. Given a folder of songs, each song is taken as input to our Spotify API and the required features are extracted. These features are passed into our MMD model and classified accordingly. The classified songs are then added into accordingly into newly created folders named “Happy Songs” and “Sad Songs”. This process is done just once or every time a new set of songs are passed.  
A separate script (Mood Radio.ipynb) is used to detect the mood of the user. An image of the user is passed. The image is processed in such a way that the model is accepts it. The processed image is then passed into the MMD model and predicted. With the label, a song from the appropriate folder is chosen at random and played. 
