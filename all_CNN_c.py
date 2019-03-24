from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import abc
import sys
from six.moves import xrange  # pylint: disable=redefined-builtin
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD,Adam
import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing, cluster
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.linalg as slin
import scipy.sparse.linalg as sparselin
import scipy.sparse as sparse 

import os.path
import time
import IPython
import tensorflow as tf
import math

from genericNeuralNet_with_noflatvar import GenericNeuralNet, variable, variable_with_weight_decay
from dataset import DataSet

tf.random.set_random_seed(10)    
def conv2d(x, W, r):
    return tf.nn.conv2d(x, W, strides=[1, r, r, 1], padding='VALID')

def softplus(x):
    return tf.log(tf.exp(x) + 1)


class All_CNN_C(GenericNeuralNet):

    def __init__(self, input_side, input_channels, conv_patch_size, h1_units, h2_units, h3_units,d1_units,d2_units, weight_decay, **kwargs):
        self.weight_decay = weight_decay
        self.input_side = input_side
        self.input_channels = input_channels
        self.input_dim = self.input_side * self.input_side * self.input_channels
        self.conv_patch_size = conv_patch_size
        self.h1_units = h1_units
        self.h2_units = h2_units
        self.h3_units = h3_units        
        self.d1_units = d1_units
        self.d2_units = d2_units


        super(All_CNN_C, self).__init__(**kwargs)


    def conv2d_softplus(self, input_x, conv_patch_size, input_channels, output_channels, stride):
        weights = variable_with_weight_decay(
            'weights', 
            [conv_patch_size * conv_patch_size * input_channels * output_channels],
            stddev=2.0 / math.sqrt(float(conv_patch_size * conv_patch_size * input_channels)),
            wd=self.weight_decay)
        biases = variable(
            'biases',
            [output_channels],
            tf.constant_initializer(0.0))
        weights_reshaped = tf.reshape(weights, [conv_patch_size, conv_patch_size, input_channels, output_channels])
        hidden = tf.nn.tanh(conv2d(input_x, weights_reshaped, stride) + biases)

        return hidden



    def get_all_params(self):
        # names=[n.name for n in tf.get_default_graph().as_graph_def().node]
        all_params = []
        
        for layer in ['conv1', 'conv2', 'conv3', 'conv4','dense1','dense2','logits']:        
            for var_name in ['kernel', 'bias']:
                temp_tensor = tf.get_default_graph().get_tensor_by_name("%s/%s:0" % (layer, var_name))            
                all_params.append(temp_tensor)               

        return all_params        
        

    def retrain(self, num_steps, feed_dict):        

        retrain_dataset = DataSet(feed_dict[self.input_placeholder], feed_dict[self.labels_placeholder])

        for step in xrange(num_steps):   
            iter_feed_dict = self.fill_feed_dict_with_batch(retrain_dataset)
            self.sess.run(self.train_op, feed_dict=iter_feed_dict)


    def placeholder_inputs(self):
        input_placeholder = tf.placeholder(
            tf.float32, 
            shape=(None, self.input_dim),
            name='input_placeholder')
        labels_placeholder = tf.placeholder(
            tf.int32,             
            shape=(None),
            name='labels_placeholder')
        return input_placeholder, labels_placeholder


    def load_model(self,inputX):        

        keras_model = Sequential()
        keras_model.add(Conv2D(self.h1_units, (self.conv_patch_size, self.conv_patch_size),
                               input_shape=(self.input_side, self.input_side, 1),name="conv1"))
        keras_model.add(Activation('relu'))
        keras_model.add(Conv2D(self.h1_units, (self.conv_patch_size, self.conv_patch_size),name="conv2"))
        keras_model.add(Activation('relu'))
        keras_model.add(MaxPooling2D(pool_size=(2, 2),name="pool1"))
        
        keras_model.add(Conv2D(self.h2_units, (self.conv_patch_size, self.conv_patch_size),name="conv3"))
        keras_model.add(Activation('relu'))
        keras_model.add(Conv2D(self.h3_units, (self.conv_patch_size, self.conv_patch_size),name="conv4"))
        keras_model.add(Activation('relu'))
        keras_model.add(MaxPooling2D(pool_size=(2, 2),name="pool2"))
        
        keras_model.add(Flatten())
        keras_model.add(Dense(self.d1_units,name="dense1"))
        keras_model.add(Activation('relu'))
        keras_model.add(Dense(self.d2_units,name="dense2"))
        keras_model.add(Activation('relu'))
        keras_model.add(Dense(10,name="logits"))
        iputX_reshaped=tf.reshape(inputX,[-1,self.input_side,self.input_side,1])
        logits=keras_model(iputX_reshaped) 
        return logits,keras_model


    def predict(self, data):
        return self.keras_model(data)
