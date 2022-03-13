import tensorflow as tf
import numpy as np
import tensorflow as keras
import matplotlib.pyplot as plt

fashion_mnist = keras.dataset.fashion_mnist

(train_img, train_labels), (test_img, test_labels) = fashion_mnist.load_data()

#print(train_labels[0])
#print(train_img[0])
plt.imshow(train_img[0], cmap='gray', vmin=0, vmax=0)
plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(units=128, activation=tf.nn.relu),
    keras.layers.Dense(units=10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.optimizer.Adam(),loss='sparse_categorical_crossentropy')
model.fit(train_img,train_labels,epochs=5)
