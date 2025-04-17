# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 17:51:44 2024

@author: ludej
"""

 
# importing the library
from torch.profiler import profile, record_function, ProfilerActivity #For Profiling
 


from time import time

import json
import numpy as np
from sklearn.model_selection import train_test_split #Used for data spliting
import tensorflow.keras as keras
import matplotlib.pyplot as plt #For plots

'''
#Inference Imports
from IPython import display
import os
import pathlib
import tensorflow as tf
'''


DATASET_PATH = "data.json" #The path to the dataset


def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)
        
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y

def prepare_datasets(test_size, validation_size):
    #Load data
    X, y = load_data(DATASET_PATH)
    
    #Create train/test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size)
    
    
    #Create the train/Validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)
    
    #CNNs expect 3D arrays (dim, dim, channel) --> eg. (64, 64, 3)
    X_train = X_train[..., np.newaxis] #This is a 4D array --> (num_samples, 64, 64, 3)
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    
    return X_train, X_validation, X_test, y_train, y_validation, y_test



def build_model(input_shape):
    #Create model
    model = keras.Sequential()
    
    #1st Conv Layer
    model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2,2), padding='same')) #zero padding, same all round
    model.add(keras.layers.BatchNormalization()) #Helps the models converge faster
    
    #2nd Conv Layer
    model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2,2), padding='same')) #zero padding, same all round
    model.add(keras.layers.BatchNormalization()) #Helps the models converge faster
    
    #3rd Conv Layer
    model.add(keras.layers.Conv2D(32, (2,2), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2,2), padding='same')) #zero padding, same all round
    model.add(keras.layers.BatchNormalization()) #Helps the models converge faster
    
    
    #Flatten the output and feed to Dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3)) #To avoid Overfitting issues we use dropout
    
    
    #Output Layer - Softmax function (classifier)
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    
    return model


def predict(model, X, y):
    X = X[np.newaxis, ...]
    
    #Prediction is a 2D array
    prediction = model.predict(X) #X - -> (1, 130, 13, 1)
    
    #Extract the index with the max value
    predicted_index = np.argmax(prediction, axis=1)
    
    print("Expected index: {}, Predicted index: {}".format(y, predicted_index ))


if __name__ == "__main__":
    
    start = time() #Start timing
    
    #Create train, Validation and Test Sets
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2) #0.2 means 20percent will be for validation
    
    #Build the CNN net
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape)
    
    '''
    #Profiler Block
    with profile(activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        with record_function("model_inference"):
            model()
    '''
    
    #Compile the CNN Network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])
    
    #Train the Network
    #model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=10) #Initially 30 epoch
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=100) #Initially 30 epoch, try epoch 10 too
    
    #Evaluate the CNN on the dataset
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))
    
    
    #Make Predictions on a sample
    X = X_test[100]
    y = y_test[100]
    
    
    predict(model, X, y)
    
    
    
    end = time() #Stop timing
    print(f'It took {end - start} seconds!')
    
    
    '''
    #Print Profiler Data
    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
    '''
    
    
    #Save Model
    
    model.save('saved_conv_model/my_model')#This directory creates itself. You don't have to create one b4 running
    
    # Save the entire model to a HDF5 file.
    # The '.h5' extension indicates that the model should be saved to HDF5.
    model.save('saved_conv_model/my_model.h5')
    
    
    
    #Plot
    metrics = history.history
    plt.figure(figsize=(16,6))
    fig = plt.figure(figsize=(16,6)) #To save figure (loss and accuracy)
    plt.subplot(1,2,1)
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'], marker='*')
    plt.legend(['loss', 'val_loss'])
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel('Epoch')
    plt.ylabel('Loss [Cross-Entropy]')
    
    plt.subplot(1,2,2)
    plt.plot(history.epoch, 100*np.array(metrics['accuracy']), 100*np.array(metrics['val_accuracy']), marker='*')
    plt.legend(['accuracy', 'val_accuracy'])
    plt.ylim([0, 100])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy - Percent [%]')
    
    plt.savefig('Accuracy_loss.png')
    
    
    '''
    #Leo - I kept getting a size based error here
    
    #Perform Inference
    DATASET_PATH = 'Data/'    
    data_dir = pathlib.Path(DATASET_PATH)


    def get_spectrogram(waveform):
      # Convert the waveform to a spectrogram via a STFT.
      spectrogram = tf.signal.stft(
          waveform, frame_length=255, frame_step=128)
      # Obtain the magnitude of the STFT.
      spectrogram = tf.abs(spectrogram)
      # Add a `channels` dimension, so that the spectrogram can be used
      # as image-like input data with convolution layers (which expect
      # shape (`batch_size`, `height`, `width`, `channels`).
      spectrogram = spectrogram[..., tf.newaxis]
      return spectrogram


    #model = build_model()

    #Run inference on Audio file
    xl = data_dir/'genres_original/blues/blues.00000.wav'
    xl = tf.io.read_file(str(xl))
    xl, sample_rate = tf.audio.decode_wav(xl, desired_channels=1, desired_samples=16000,)
    xl = tf.squeeze(xl, axis=-1)
    waveform = xl
    xl = get_spectrogram(xl)
    xl = xl[tf.newaxis,...]

    prediction = model(xl)
    xl_labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae','rock' ]
    plt.bar(xl_labels, tf.nn.softmax(prediction[0]))
    plt.title('No')
    plt.show()

    display.display(display.Audio(waveform, rate=16000))
    
    '''
    
 
    
 
"""

Got Error: ModuleNotFoundError: No module named 'tensorflow.keras'
Solution:
conda update --force conda
pip install --upgrade pip
python.exe -m pip install --upgrade pip
pip uninstall tensorflow
    
All the above didn't work. I finally had to uninstall and reinstall Anaconda, and then install tensorflow'


"""