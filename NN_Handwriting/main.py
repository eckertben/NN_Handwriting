import os
import cv2
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf

# Imports the handwriting dataset
mnist = tf.keras.datasets.mnist

# Dataset is already split into the training data and tested data, normally 80/20
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizes the pixel 0-255 -> 0-1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()

"These are all the layers"
# Flattens the 2D array into a 1D array that is 28*28 len
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# relu is Rectify Linear Unit, 0 if negative then straight up linearlu
#Need more research
model.add(tf.keras.layers.Dense(128, activations='relu'))

model.add(tf.keras.layers.Dense(128, activations='relu'))

# Acts as a confidence layers, makes sure all 10 neurons add to 1 (or at least close)
model.add(tf.keras.layers.Dense(128, activations='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3) #Trains it 3 times
model.save("handwritten.model")