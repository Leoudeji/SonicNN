# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:56:39 2024

@author: ludej
"""

import librosa, librosa.display #librosa display is built on Matplotlib
import matplotlib.pyplot as plt
import numpy as np #We need this for Fourier Transform

#Waveform
#fft -> spectrum
#stft -> spectrogram
#MFCCs

file = "blues.00000.wav"

#Waveform
signal, sr = librosa.load(file, sr=22050) #sr = sample rate; sr * T -> 22050 * 30
#librosa.display.waveplot(signal, sr=sr) #leo: I got this error here: "AttributeError: module 'librosa.display' has no attribute 'waveplot'"
librosa.display.waveshow(signal, sr=sr)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()


#fft -> spectrum
fft = np.fft.fft(signal)

magnitude = np.abs(fft)

frequency = np.linspace(0, sr, len(magnitude))#Linespace is a line function that gives us evenly spaced values in a dimension
#We want to know how much each frequency contributes to the original sound
plt.plot(frequency, magnitude)
plt.xlabel("Frequency") 
plt.ylabel("Magnitude") #Also called Energy
plt.show()


left_frequency = frequency[:int(len(frequency)/2)]
left_magnitude = magnitude[:int(len(frequency)/2)]

plt.plot(left_frequency, left_magnitude)
plt.xlabel("Frequency") 
plt.ylabel("Magnitude") #Also called Energy
plt.show()


#stft -> spectrogram
n_fft = 2048
hop_length = 512

stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
spectrogram = np.abs(stft)


librosa.display.specshow(spectrogram, sr=sr, hop_length=hop_length)
plt.xlabel("Time") 
plt.ylabel("Frequency") #Also called Energy
plt.colorbar()
plt.show()

log_spectrogram = librosa.amplitude_to_db(spectrogram) #Converts to DB
librosa.display.specshow(log_spectrogram , sr=sr, hop_length=hop_length)
plt.xlabel("Time") 
plt.ylabel("Frequency") #Also called Energy
plt.colorbar()
plt.show()

#MFCCs - 
MFFCs = librosa.feature.mfcc(y=signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13) #Leo : I got an error here. MFCC always need value assign to 'y' not the value directly as in the Tut. I found the solution here: "https://blog.csdn.net/qq_57438473/article/details/130394336"
#MFFCs = librosa.feature.mfcc(signal)
librosa.display.specshow(MFFCs, sr=sr, hop_length=hop_length)
plt.xlabel("Time") 
plt.ylabel("MFFC") 
plt.colorbar()
plt.show()








