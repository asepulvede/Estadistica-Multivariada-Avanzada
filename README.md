# Identification of bird sounds in soundscapes with Convolutional Neuronal Network
This is a Convolutional Neural Network (CNN) based audio classification model to detect and classify bird species. The classification model consists of stages:
- birdEDA.py: Where we do an Exploratory Data Analysis, which was obtained from the Kaggle's competition: https://www.kaggle.com/c/birdclef-2022. For the Exploratory Data Analysis, we choose a random audio file to perform an exploratory visualization, where we calculate and plot the Mel spectrogram, spectral centroid, spectral bandwidth, and the spectral roll-off. On the other hand, to have a global image of the regions that provide the recording, a heat map was made. Finally, a bar graph and a mapping of bird species with most and least audio files were made.
- birdPrediction.py: This file consists of the creation of the neural network and the training and testing phase of the model: Firstly we have the data preparation, were we sample the training data by using only high quality (rating) audios, that is, audio with a rating greater than 4. Secondly we extract the spectrograms of each audio file, for this, we only calculate the mel spectogram of the first 15 seconds of each audio and save each spectrogram as PNG image in a working directory for later access. Furthermore samples are loaded from hard drive and combined into a large NumPy array to make it easy to use the data for training. The next step is build the model: we use a design similar to AlexNet with four convolutional layers and three dense layers. Each layer has the sequence: conv, relu, bnorm and maxpool, so we perform global average pooling and add 2 dense layers. The last layer is the classification layer and is softmax activated. Once the CNN is built we specify the optimizer: Adam optimization, loss function: Cross entropy and metric: Accuracy, F1 Score and ROC curve. 
