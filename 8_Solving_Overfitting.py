# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 05:45:06 2024

@author: ludej
"""


import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

#import tensorflow as tf
#import onnx
#import tf2onnx

#import os


DATASET_PATH = "data.json" #The path to the dataset


def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)
        
        #convert lists ino numpy arrays
        inputs = np.array(data["mfcc"]) #parse mfcc
        targets = np.array(data["labels"])
        
        return inputs, targets
    

def plot_history(history):
    fig, axs = plt.subplots(2) #returns 2 subplots
    
    #Accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy") #Must start with small letter
    axs[0].plot(history.history["val_accuracy"], label="test accuracy") #Must start with small letter
    axs[0].set_ylabel("Epoch")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right") #Must start with small letter
    axs[0].set_title("Accuracy eval")
    
    #Error subplot
    axs[1].plot(history.history["loss"], label="train error") #Must start with small letter
    axs[1].plot(history.history["val_loss"], label="test error") #Must start with small letter
    axs[1].set_ylabel("Epoch")
    axs[1].set_ylabel("Error")
    axs[1].legend(loc="upper right") #Must start with small letter
    axs[1].set_title("Error eval")
    
    plt.show()
    
    
    
if __name__ == "__main__":
    inputs, targets = load_data(DATASET_PATH) #Parse in dataset path
    
    #Split the datatset itno training and testing datasets
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.3) #se 30% for testing
        
    
    #Build Network Architecture
    model = keras.Sequential([
        #input layer
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),
        
        #1st hidden layer
        keras.layers.Dense(512,activation="relu", kernel_regularizer= keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3), #We use a dropout value very similar to the difference between the training and validation accuracy 
        
        #2nd hidden layer
        keras.layers.Dense(256,activation="relu", kernel_regularizer= keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        
        #3rd hidden layer
        keras.layers.Dense(64,activation="relu", kernel_regularizer= keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        
        #Output layer
        keras.layers.Dense(10, activation="softmax") #Note that the output activation differs from the hidden processing layer activation
        
        
        ])
        
    
    #Compile Our Network\
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    model.summary() #Prints out model details
    
    #Train the Network
    history = model.fit(inputs_train, targets_train, 
              validation_data=(inputs_test, targets_test),
              epochs=70,
              batch_size=32)
    
    #Plot Model accuracy and error over time
    plot_history(history)
    
    
    #!mkdir -p saved_model
    model.save('saved_model/my_model') #This directory creates itself. You don't have to create one b4 running
    
    # Save the entire model to a HDF5 file.
    # The '.h5' extension indicates that the model should be saved to HDF5.
    model.save('saved_model_HDF5/my_model.h5')
    
    '''
    #Save Model as ONNX file
    input_signature = [tf.TensorSpec([3, 3], tf.float32, name='x')]
    # Use from_function for tf functions
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
    onnx.save(onnx_model, "saved_model_ONNX/model.onnx")
    '''