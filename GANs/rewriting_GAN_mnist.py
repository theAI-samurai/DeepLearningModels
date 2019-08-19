import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

#mnist = keras.datasets.mnist()  # Loading MNIST dataset from Keras
mnist = input_data.read_data_sets('MNIST_data')
train_images = mnist.train._images # training has 55000 sample images in total
train_labels = mnist.train._labels # labels for 50000 images
plt.imshow(train_images[0,:].reshape(28,28))    # plotting 1st image, resizing 784 into 28,28==> 28*28=784

tf.reset_default_graph() # resets default global graph
tf.get_default_graph()

'''
a = tf.constant(5)
b = tf.constant(10)
c = tf.placeholder(tf.int32)
z = a+b+c

result = tf.Session().run(z, {c: 2})
print(result)
'''

real_images = tf.placeholder(tf.float32, shape=[None, 784])  # size of image 28*28=784
z = tf.placeholder(tf.float32, shape=[None, 100])            # planning to keep batch size = 100


def generator(z, reuse=None):
    with tf.variable_scope('gen', reuse=reuse):  # creates namespace for variable and operator in DEFAULT GRAPH
        hidden1 = tf.layers.dense(inputs=z, units=128, activation=tf.nn.leaky_relu)
        hidden2 = tf.layers.dense(inputs=hidden1, units=128, activation=tf.nn.leaky_relu)
        output = tf.layers.dense(inputs=hidden2, units=784, activation=tf.nn.tanh)
        return output

def discriminator()


G = generator(z)
D_output_real, D_logits_real = discriminator(real_images)
D_output_fake, D_logits_fake = discriminator(G, reuse=True)