import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

from Recognition.Emotion.helpFunctions import *

'''
    Conv -> BN -> Activation -> Conv -> BN -> Activation -> MaxPooling
    Conv -> BN -> Activation -> Conv -> BN -> Activation -> MaxPooling
    Conv -> BN -> Activation -> Conv -> BN -> Activation -> MaxPooling
    Flatten
    Dense -> BN -> Activation
    Dense -> BN -> Activation
    Dense -> BN -> Activation
    Output layer
'''


def training_data():
    class_numbers = 7
    epochs_index = 10
    batch_size = 64
    num_features = 64
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

    print(model.summary())

    data_generator = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=.1,
        horizontal_flip=True)

    es = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)

    history = model.fit_generator(data_generator.flow(train_images, train_labels, batch_size),
                                  epochs=epochs_index, callbacks=[es], verbose=2,
                                  validation_data=(test_images, test_labels))

    plt.plot(history.history['accuracy'], label="accuracy")
    plt.plot(history.history['val_accuracy'], label="val_accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
    print("test loss: " + test_loss)
    print("test accuracy: " + test_accuracy)
