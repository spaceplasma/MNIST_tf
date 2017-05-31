# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data # <-- get the tensorflow MNSIT data

import tensorflow as tf # <-- import Tensorflow here

FLAGS = None

def weight_value(shape):
    # We need to generate the weight values for different layers.
    # Add a bit of noise - helps to prevent 0 gradients
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
  
def bias_value(shape):
    # Bias values - always positive to avoid dead neurons in the layer
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
    
def conv2d(x,W):
    # Uses a stride of 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
   
def max_pool2x2(x):
    # Pools the matrix in 2x2 fashion
    # ksize -> kernal size
    # stride -> 2 step of course!
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def main(_):
    
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    ### ---------------------------------------------------
    #
    # Create the model
    #
    x = tf.placeholder(tf.float32, [None, 784]) # <-- create a tf palceholder for the data
    y_ = tf.placeholder(tf.float32, [None, 10])

    x_image = tf.reshape(x, [-1,28,28,1])
    
    #Start with a 5x5 patch and compute 32 features for each patch
    W_conv1 = weight_value([5, 5, 1, 32])
    b_conv1 = bias_value([32])
    
    #Lets use ReLU rather than sigmoid or other such
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool2x2(h_conv1)

    #Add a second layer -> take the previous features to pass on
    W_conv2 = weight_value([5, 5, 32, 64])
    b_conv2 = bias_value([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool2x2(h_conv2)
    
    # Add a fully connected layer to process the entire image
    W_fc1 = weight_value([7 * 7 * 64, 1024])
    b_fc1 = bias_value([1024])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    # Add a dropout feature - reduces overfitting in largeg datasets
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) 

    # Output layer
    W_fc2 = weight_value([1024, 10])
    b_fc2 = bias_value([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    ### ----------------------------------------------

    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.

    cross_entropy = tf.reduce_mean(
                                  tf.nn.softmax_cross_entropy_with_logits
                                  (labels=y_, logits=y_conv))
  
    # Try AdamOptimizer                             
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        batch = mnist.train.next_batch(100)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)