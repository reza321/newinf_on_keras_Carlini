## setup_mnist.py -- mnist data and model loading code
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
import os
import pickle
import gzip
import urllib.request

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model

def generate_data(data, samples, start=0):
    
    inputs = []
    targets = []
    for i in range(samples):
        seq = range(data.test.labels.shape[1])
        for j in seq:                    
            if (j == np.argmax(data.test.labels[start+i])):
                continue
            inputs.append(data.test.x[start+i])
            targets.append(np.eye(data.test.labels.shape[1])[j])

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets




class All_CNN_C_Attack:
    
    def __init__(self, input_side, num_classes,num_channels, conv_patch_size, h1_units, h2_units, h3_units,d1_units,d2_units,restore):
        self.image_size = input_side
        self.num_channels = num_channels
        self.input_dim = self.image_size * self.image_size * self.num_channels    
        self.conv_patch_size = conv_patch_size
        self.h1_units = h1_units
        self.h2_units = h2_units
        self.h3_units = h3_units        
        self.d1_units = d1_units
        self.d2_units = d2_units
        self.num_labels=num_classes
        self.restore=restore
        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10
        
        model = Sequential()
        model.add(Conv2D(self.h1_units, (self.conv_patch_size, self.conv_patch_size),
                               input_shape=(self.image_size, self.image_size, 1),name="conv1"))
        model.add(Activation('relu'))
        model.add(Conv2D(self.h1_units, (self.conv_patch_size, self.conv_patch_size),name="conv2"))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),name="pool1"))
        
        model.add(Conv2D(self.h2_units, (self.conv_patch_size, self.conv_patch_size),name="conv3"))
        model.add(Activation('relu'))
        model.add(Conv2D(self.h3_units, (self.conv_patch_size, self.conv_patch_size),name="conv4"))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),name="pool2"))
        
        model.add(Flatten())
        model.add(Dense(self.d1_units,name="dense1"))
        model.add(Activation('relu'))
        model.add(Dense(self.d2_units,name="dense2"))
        model.add(Activation('relu'))
        model.add(Dense(10,name="logits"))

        model.load_weights(restore)

        self.model = model

    def predict(self, data):
        return self.model(data)
