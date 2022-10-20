import tensorflow as tf
from matplotlib import pyplot as plt

from Recognition.Emotion.helpFunctions import *


def training_data():
    epochs_index = 10
    # load train
    train_images, train_labels = load_dataset("Recognition/archive/train/")
    test_images, test_labels = load_dataset("Recognition/archive/test/")
    print(train_images[0].shape)
    # train_new_iamges = resize_images(train_images, 224)
    # test images if they exist
    # plt.figure(figsize=(10, 10))
    # for i in range(25):
    #     plt.subplot(5, 5, i + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(True)
    #     plt.imshow(train_images[i], cmap=plt.cm.binary)
    #     plt.xlabel(train_labels[i])
    # plt.show()
    #
    # plt.figure(figsize=(10, 10))
    # for i in range(25):
    #     plt.subplot(5, 5, i + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(True)
    #     plt.imshow(test_images[i], cmap=plt.cm.binary)
    #     plt.xlabel(test_labels[i])
    # plt.show()

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.MaxPool2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPool2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(10))

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=epochs_index, validation_data=(test_images, test_labels))

    plt.plot(history.history['accuracy'], label="accuracy")
    plt.plot(history.history['val_accuracy'], label="val_accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
    print("test loss: " + test_loss)
    print("test accuracy: " + test_accuracy)

