# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 17:41:26 2024

@author: ludej
"""
import os
import librosa
import math
import json #This is to store values

DATASET_PATH = "Data/genres_original/" #Leo: sIf I dont't add "genres_original" it tries to read from "Images_original" folder fater it's done with "genres_original" folder
#JSON_PATH = "Data/data.json"
JSON_PATH = "data.json"
SAMPLE_RATE = 22050
DURATION = 30 #measured in secondss
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    
    #dictionary to store data
    data = {
        "mapping":[], #maps genre labels to numbers
        "mfcc": [],  #stores mfcc vectors for each segment. it's the training data/input
        "labels": [] #zeros and ones. This is the target
        }
    
    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length) #1.2 -> 2    
    
    
    #Loop through all the genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        #ensure that we're not at the root level
        if dirpath is not dataset_path:
            
            #save the semantic label
            dirpath_components = dirpath.split("/") #genre/blues => ["genre", "blues"]
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))
            
            
            #Now, we go through all the files in the genre folder
            #process files for a specific genre
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                
                
                #Process Segments extracting mfcc and storing data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s  #s=0 -> 0 #s is the current segement that we are in
                    finish_sample = start_sample + num_samples_per_segment #s=0 -> num_samples_per_segment
                    
                    
                    mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample], #Note that this value must be assigned to why to avoid this error: "TypeError: mfcc() takes 0 positional arguments but 1 positional argument (and 2 keyword-only arguments) were given"
                                                sr=sr,
                                                n_fft=n_fft,
                                                n_mfcc=n_mfcc,
                                                hop_length=hop_length
                                                #Note that these values come from the argument of the function
                                                )
                    
                    mfcc = mfcc.T
                    
                    #Store mfcc for segment if it has the expected length
                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        #In the next two lines below we are storing the mfcc and labels of each iteration
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        
                        print("{}, segment:{}".format(file_path,s+1))
                        
        
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        
        
if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
                    
        
                    
                    
        
        
        
        
        
        
        
        
        
        
        
        
        
        