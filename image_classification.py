import tensorflow as tf
import numpy as np
from tensorflow import keras

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc') > 0.6):
            print("Reached 60% accuracy!! Stopping training.")
            self.model.stop_training = True

callbacks = myCallback()

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255;
test_images = test_images / 255;

model = tf.keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)),
                            keras.layers.Dense(512, activation=tf.nn.relu),
                            keras.layers.Dense(10, activation=tf.nn.softmax)])
model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, callbacks=[callbacks])
model.evaluate(test_images, test_labels)
