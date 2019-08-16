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





