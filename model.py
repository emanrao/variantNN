"""
Model for variantNN
"""
import logging
import numpy as np
import intervaltree
import random
import tensorflow as tf

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_model(features, FLAGS):
    
    # Model Parameters
    kernel_size1 = (2, 4) # Convolution layer 1
    kernel_size2 = (3, 4) # Convolution layer 2
    pool_size1 = (7, 1) # MaxPool layer 1
    pool_size2 = (3, 1) # MaxPool layer 2
    filter_num = 48 # Conv layers
    hidden_layer_unit_number = 48 # First fully connected layer
    
    # Concolution + MaxPool Layer 1
    with tf.name_scope('Conv1'):
        conv1 = tf.layers.conv2d(
            inputs=features, # Alignment data in 15x4x3 matrices
            filters=filter_num,
            kernel_size=kernel_size1,
            padding="same",
            activation=tf.nn.elu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=pool_size1, strides=1)
        
    # Concolution + MaxPool Layer 2
    with tf.name_scope('Conv2'):
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=filter_num,
            kernel_size=kernel_size2,
            padding="same",
            activation=tf.nn.elu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=pool_size2, strides=1)
        
        flat_size = ( 15 - (pool_size1[0] - 1) - (pool_size2[0] - 1))
        flat_size *= ( 4 - (pool_size1[1] - 1) - (pool_size2[1] - 1))
        flat_size *= filter_num
        
        conv2_flat =  tf.reshape(pool2, [-1,  flat_size])
        
    # Fully Connected + Dropout Layer 1
    with tf.name_scope('FC1'):
        unit_num = hidden_layer_unit_number
        h1 = tf.layers.dense(inputs=conv2_flat, units=unit_num, activation=tf.nn.elu)
        dropout1 = tf.layers.dropout(inputs=h1, rate=0.50)
        
    # Fully Connected + Dropout Layer 2
    with tf.name_scope('FC2'):
        h2 = tf.layers.dense(inputs=dropout1, units=unit_num, activation=tf.nn.elu)
        dropout2 = tf.layers.dropout(inputs=h2, rate=0.50)
        
    # Fully Connected + Dropout Layer 3
    with tf.name_scope('FC3'):
        h3 = tf.layers.dense(inputs=dropout2, units=unit_num, activation=tf.nn.elu)
        dropout3 = tf.layers.dropout(inputs=h3, rate=0.50)
        
    #Configure Outputs
    with tf.name_scope('Dense'):
        Y1 = tf.layers.dense(inputs=dropout3, units=4, activation=tf.nn.sigmoid)  # Base ID [A(0) C(1) G(2) T(3)]
        Y2 = tf.layers.dense(inputs=dropout3, units=4, activation=tf.nn.elu)  # Variant Type [het(0) hom(1) non-variant(2) non-SNP(3)]
        
        Y_out = tf.concat([Y1, Y2], -1)
        
    return Y_out


def get_loss(predictions,labels):
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.square(predictions-labels)) 
        return loss