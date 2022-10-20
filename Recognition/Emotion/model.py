from matplotlib import pyplot as plt

from Recognition.Emotion.helpFunctions import *

import tensorflow as tf
from tensorflow.keras import datasets, layers, models


def training_data():
    # load train
    train_images, train_labels = load_dataset("Recognition/archive/train/", 224)
    test_images, test_labels = load_dataset("Recognition/archive/test/", 224)
    print(train_images[0].shape)
    #train_new_iamges = resize_images(train_images, 224)
    # test images if they exist
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(True)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(train_labels[i])
    plt.show()

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(True)
        plt.imshow(test_images[i], cmap=plt.cm.binary)
        plt.xlabel(test_labels[i])
    plt.show()

