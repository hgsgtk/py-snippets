from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

class_names = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]

print(train_images.shape)  # (60000, 28, 28) 60000枚 28px 28px
print(train_labels)  # [9 0 0 ... 3 0 5]
print(len(train_labels))  # 60000

print(test_images.shape)  # (10000, 28, 28)
print(len(test_labels))
