#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import pickle

# Audio
import librosa
import librosa.display

# Plotting
import matplotlib.pyplot as plt 

# Model
from sklearn.utils import shuffle
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from tensorflow_addons.metrics import F1Score
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import seaborn as sns


#%%
# Hyperparameters
seed = 8685
sampleRt = 32000
sLength = 5
shapeSpec = (48, 128)
minFreq = 500
maxFreq = 12500
nEpochs = 20
audiosRequired = 175
timeLimit = 15

# Paths
trainingPath ='/Users/LuisaToro/Desktop/Bird prediction/train_metadata.csv'
inputPath = '/Users/LuisaToro/Desktop/Bird prediction/train_audio/'
outputPath = '/Users/LuisaToro/Desktop/Bird prediction/melspectrogram_dataset/'
soundscapePath = '/Users/LuisaToro/Desktop/Bird prediction/test_soundscapes/soundscape_453028782.ogg'
testPath = '/Users/LuisaToro/Desktop/Bird prediction/test.csv'
soundscapesPaths = '/Users/LuisaToro/Desktop/Bird prediction/test_soundscapes'
samplePath = '/Users/LuisaToro/Desktop/Bird prediction/sample_submission.csv'
#%%
def sampling(trainingPath):
    
    train = pd.read_csv(trainingPath)
    
    # Sample the train data by using only high quality (rating) audios
    train = train.query('rating>=4')
    
    # Count the number of audio samples per specie 
    speciesCount = {}
    for birdSpecies, count in zip(train.primary_label.unique(), 
                                   train.groupby('primary_label')['primary_label'].count().values):
        speciesCount[birdSpecies] = count
    
    # Use the birds that meet the number required of audio samples
    mostBirds = [key for key,value in speciesCount.items() if value >= audiosRequired ] 
    
    # Sample train according to that
    Train = train.query('primary_label in @mostBirds')
    Labels = sorted(Train.primary_label.unique())
    
    # Check the sample
    print('Number of species left:', len(Labels))
    print('Number of samples  left:', len(Train))
    print('Labels:', mostBirds)
    Train = shuffle(Train, random_state=seed)
    return Train, Labels, mostBirds

Train, Labels, mostBirds= sampling(trainingPath)
#%%
with open('Labels.pkl','wb') as f:
    pickle.dump(Labels,f)
    
#%%
# Take audio samples and process them into mel spectograms
# Saves the images into a directory
def getSpectrograms(filepath, primary_label, outputPath):
    
    # Uses the librosa library, and cuts the audios into an established time limit.
    sig, rate = librosa.load(filepath, sr=sampleRt, offset=None, duration=timeLimit)
    
    # Split signal into five second parts
    signalPartitions = []
    for i in range(0, len(sig), int(sLength * sampleRt)):
        split = sig[i:i + int(sLength * sampleRt)]
        if len(split) < int(sLength * sampleRt):
            break
        signalPartitions.append(split)
        
    # Get mel spectrograms for each audio part
    s = 0
    savedSamples = []
    for part in signalPartitions:
        hopLen = int(sLength * sampleRt / (shapeSpec[1] - 1))
        melSpectro = librosa.feature.melspectrogram(y=part, 
                                                  sr=sampleRt, 
                                                  n_fft=1024, 
                                                  hop_length=hopLen, 
                                                  n_mels=shapeSpec[0], 
                                                  fmin=minFreq, 
                                                  fmax=maxFreq)
        melSpectro = librosa.power_to_db(melSpectro, ref=np.max) 
        
        # Normalize
        melSpectro -= melSpectro.min()
        melSpectro /= melSpectro.max()
        
        # Save images
        save_dir = os.path.join(outputPath, primary_label)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, filepath.rsplit(os.sep, 1)[-1].rsplit('.', 1)[0] + 
                                 '_' + str(s) + '.png')
        im = Image.fromarray(melSpectro * 255.0).convert("L")
        im.save(save_path)
        savedSamples.append(save_path)
        s += 1
    return savedSamples
#%%
# Goes through the audio samples and generates their spectrograms

def generateSpecs(inputPath,outputPath,Train,mostBirds):
    samples = []
    with tqdm(total=len(Train)) as pbar:
        for idx, row in Train.iterrows():
            pbar.update(1)
            if row.primary_label in mostBirds:
                audioPath = os.path.join(inputPath, row.filename)
                samples += getSpectrograms(audioPath, row.primary_label, outputPath)
                
    trainSpecs = shuffle(samples, random_state=seed)
    print('Extracted {} spectrograms'.format(len(trainSpecs)))
    return trainSpecs
    
trainSpecs = generateSpecs(inputPath,outputPath,Train,mostBirds)
#%%
def plotSpecs(trainSpecs):
    plt.figure(figsize=(15, 7))
    for i in range(12):
        spec = Image.open(trainSpecs[i])
        plt.subplot(3, 4, i + 1)
        plt.title(trainSpecs[i].split(os.sep)[-1])
        plt.imshow(spec, origin='lower')
    
plotSpecs(trainSpecs)
#%%
def dataPrep(trainSpecs,Labels):
    TrainSpecs, TrainLabels = [], []
    with tqdm(total=len(trainSpecs)) as pbar:
        for path in trainSpecs:
            pbar.update(1)

            spec = Image.open(path)
            spec = np.array(spec, dtype='float32')
            
            # Normalize
            spec = (spec-spec.min())/spec.max()
            if not spec.max() == 1.0 or not spec.min() == 0.0:
                continue
    
            # Add new axis
            spec = np.expand_dims(spec, -1)
            spec = np.expand_dims(spec, 0)
    
            # Add to train data
            if len(TrainSpecs) == 0:
                TrainSpecs = spec
            else:
                TrainSpecs = np.vstack((TrainSpecs, spec))
    
            # Add to label data
            target = np.zeros((len(Labels)), dtype='float32')
            bird = path.split(os.sep)[-2]
            target[Labels.index(bird)] = 1.0
            if len(TrainLabels) == 0:
                TrainLabels = target
            else:
                TrainLabels = np.vstack((TrainLabels, target))
    return TrainSpecs, TrainLabels
TrainSpecs, TrainLabels = dataPrep(trainSpecs,Labels)
#%%
def designModel():
    tf.random.set_seed(seed)
    model2 = tf.keras.Sequential([
        
        # Convolutional layer 1
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', 
                               input_shape=(shapeSpec[0], shapeSpec[1], 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Convolutional layer 2
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)), 
        
        # Convolutional layer 3
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)), 
        
        # Convolutional layer 4
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Global pooling
        tf.keras.layers.GlobalAveragePooling2D(), 
        
        # Dense block
        tf.keras.layers.Dense(256, activation='relu'),  
        tf.keras.layers.Dropout(0.5),
        
        # Classification layer
        tf.keras.layers.Dense(len(Labels), activation='softmax')
    ])
    print('Model has {} parameters.'.format(model2.count_params()))
    f1Score = F1Score(num_classes=len(Labels),average='macro',name='f1_score')
    return model2, f1Score


model2,f1Score = designModel()
#%%
def refineModel(model2):
    
    # Set optimizer, loss and metric
    model2.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.01),
                  metrics=['accuracy',f1Score])
    
    # Add callbacks: learning rate, early stopping, and checkpoint saving
    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1_score',
                                                      mode='max',
                                                      patience=2, 
                                                      verbose=1, 
                                                      factor=0.5),
                 tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                  verbose=1,
                                                  patience=20),
                 tf.keras.callbacks.ModelCheckpoint(filepath='best_model2.h5', 
                                                    mode='max',
                                                    monitor='val_f1_score',
                                                    verbose=0,
                                                    save_best_only=True)]
    return model2, callbacks

CNNmodel, callbacks = refineModel(model2)

#%%
def plot(history):
    his = pd.DataFrame(history.history)
    
    # Plot loss:
    plt.figure()
    plt.plot(range(len(his)),his['loss'],label='Training')
    plt.plot(range(len(his)),his['val_loss'],label='Validation')
    plt.legend()
    plt.ylim([0,1.1])
    plt.title('Loss')
    
    # Plot accuracy
    plt.figure()
    plt.plot(range(len(his)),his['accuracy'],label='Training')
    plt.plot(range(len(his)),his['val_accuracy'],label='Validation')
    plt.legend()
    plt.ylim([0,1.1])
    plt.title('Accuracy')
    
    # Plot f1
    plt.figure()
    plt.plot(range(len(his)),his['f1_score'],label='Training')
    plt.plot(range(len(his)),his['val_f1_score'],label='Validation')
    plt.legend()
    plt.ylim([0,1.1])
    plt.title('F1 Score')
    
    plt.show()  

#%%
his = CNNmodel.fit(TrainSpecs, 
          TrainLabels,
          batch_size=32,
          validation_split=0.25,
          callbacks=callbacks,
          verbose=1,
          epochs=nEpochs)

#%%
plot(his)

#%%

XTrain, XTest, yTrain, yTest = train_test_split(TrainSpecs, TrainLabels, test_size=0.25, shuffle=False)

nEpochs = 20

his2 = CNNmodel.fit(XTrain, 
          yTrain,
          batch_size=32,
          validation_data = (XTest,yTest),
          callbacks=callbacks,
          verbose=1,
          epochs=nEpochs)

plot(his2)



#%%
def ROCcurve(yTest,predictions):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(predictions[0])):
        fpr[i],tpr[i],_ = roc_curve(yTest[:,i], predictions[:,i])
        roc_auc[i] = auc(fpr[i],tpr[i])
        
    plt.figure()
    lw =2
    
    for i in range(len(predictions[0])):
        plt.figure()
        Class = i
        plt.plot(
            fpr[Class],
            tpr[Class],
            lw=lw,
            label = 'ROC curve for class %0.0f (area = %0.2f)' %(i , roc_auc[Class]),
            
            )
        plt.plot([0,1],[0,1],lw=lw, linestyle = '--')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend()
predictions = CNNmodel.predict(XTest)   
ROCcurve(yTest,predictions)
#%%
yTrue = np.argmax(yTest, axis=-1)

yPredict = np.argmax(predictions, axis=-1)
cm = confusion_matrix(yTrue, yPredict)

fig = plt.figure()
ax = plt.subplot()
sns.heatmap(cm)

ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.xticks(rotation =90)
ax.xaxis.set_ticklabels(Labels)
plt.yticks(rotation =0)
ax.yaxis.set_ticklabels(Labels)

plt.title('Multilabel confusion matrix')

#%%
print(his.history.keys()) 
def plot(history):
    his = pd.DataFrame(history.history)
    
    # Plot loss:
    plt.figure()
    plt.plot(range(len(his)),his['loss'],label='Training')
    plt.legend()
    plt.title('Loss')
    
    # Plot accuracy
    plt.figure()
    plt.plot(range(len(his)),his['accuracy'],label='Training')
    plt.legend()
    plt.title('Accuracy')
    
    plt.show()  
plot(his)
    
#%%
# Prediction
testPath = '/Users/LuisaToro/Desktop/Bird prediction/test2.csv'
sampleSubmission = pd.read_csv(samplePath)
testFile = pd.read_csv(testPath)

with open('LABELS.pkl','wb') as f:
    pickle.dump(Labels,f)

def load_pickle(path):
    with open(path,'rb') as f:
        file = pickle.load(f)
        
    return file

LABELS = load_pickle('LABELS.pkl')
                   
#%%

def listOfFiles(path):
    '''get test sound files'''
    return [os.path.join(path, f) for f in os.listdir(path) if f.rsplit('.', 1)[-1] in ['ogg']]

testAudio=listOfFiles(soundscapesPaths)

# test files are hidden,  hence checking on train_soundscapes
if len(testAudio) == 0:
    testAudio = listOfFiles('../input/birdclef-2021/train_soundscapes')
    
print('{} FILES IN TEST SET.'.format(len(testAudio)))

#%%

testAudio=listOfFiles(soundscapesPaths)

def predictSpecies(threshold, testAudio, testFile):
    birds = testFile['bird']
    row_id = []
    prediction = []
    for file_path in testAudio[:2]:
        sig, rate = librosa.load(file_path, sr=sampleRt)
        sig_splits = []
        for i in range(0, len(sig), int(sLength * sampleRt)):
            split = sig[i:i + int(sLength * sampleRt)]
            if len(split) < int(sLength * sampleRt):
                break
            sig_splits.append(split)
        seconds= 0
        cont = 0
        for chunk in sig_splits:
            bird = birds[cont]
            cont+=1
            seconds += 5
            hopLen = int(sLength * sampleRt / (shapeSpec[1] - 1))
            melSpectro = librosa.feature.melspectrogram(y=chunk, 
                                                      sr=sampleRt, 
                                                      n_fft=1024, 
                                                      hop_length=hopLen, 
                                                      n_mels=shapeSpec[0], 
                                                      fmin=minFreq, 
                                                      fmax=maxFreq)

            melSpectro = librosa.power_to_db(melSpectro, ref=np.max) 
            melSpectro = (melSpectro - melSpectro.min())/melSpectro.max()
            melSpectro = np.expand_dims(melSpectro, -1)
            melSpectro = np.expand_dims(melSpectro, 0)
            p = model2.predict(melSpectro)[0]
            idx = p.argmax()
            species = LABELS[idx]
            score = p[idx]
            print(score)
            print(species, bird)
            row_id.append(file_path.split(os.sep)[-1].rsplit('.', 1)[0]+'_' + str(bird)+'_'+ str(seconds))  
            if species == bird and score > threshold:
                prediction.append('True')
            else:
                prediction.append( 'False')    
        result = pd.DataFrame({'row_id': row_id, 'target': prediction})
    return result

testFile = pd.read_csv(testPath)
result = predictSpecies(0.5, testAudio, testFile)

print(result)
