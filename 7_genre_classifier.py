# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 03:40:08 2024

@author: ludej
"""

#load data
#Split the data into train and test sets
#build the network  architecture
#compile network
#train network

import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras


DATASET_PATH = "data.json" #The path to the dataset


def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)
        
        #convert lists ino numpy arrays
        inputs = np.array(data["mfcc"]) #parse mfcc
        targets = np.array(data["labels"])
        
        return inputs, targets
    
    
if __name__ == "__main__":
    inputs, targets = load_data(DATASET_PATH) #Parse in dataset path
    
    #Split the datatset itno training and testing datasets
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.3) #se 30% for testing
        
    
    #Build Network Architecture
    model = keras.Sequential([
        #input layer
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),
        
        #1st hidden layer
        keras.layers.Dense(512,activation="relu"),
        
        #2nd hidden layer
        keras.layers.Dense(256,activation="relu"),
        
        #3rd hidden layer
        keras.layers.Dense(64,activation="relu"),
        
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
    model.fit(inputs_train, targets_train, 
              validation_data=(inputs_test, targets_test),
              epochs=50,
              batch_size=32)
    
    
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        
        
        