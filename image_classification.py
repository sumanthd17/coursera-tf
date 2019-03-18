import tensorflow as tf
import numpy as np
from tensorflow import keras

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255;
test_images = test_images / 255;

model = tf.keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)),
                            keras.layers.Dense(128, activation=tf.nn.relu),
                            keras.layers.Dense(10, activation=tf.nn.softmax)])
model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)
model.evaluate(test_images, test_labels)
