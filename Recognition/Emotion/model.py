import datetime

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scipy.ndimage import label

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

class Model:

    index_train_images = 28709
    index_validation_images = 7178

    # this is for the last training, I deleted the images which coulde't be processed
    # index_train_images = 20349
    # index_validation_images = 5074
    epoches = 50
    batch_size = 64
    image_size = 48
    file_name_train = "/home/simon/BA/archive/train"
    file_name_test = "/home/simon/BA/archive/test"
    def __init__(self, index_train_images=28709, index_validatiyon_images=7178, epoches=50, batch_size=64, image_size=48):
        self.index_train_images = index_train_images
        self.index_validation_images = index_validatiyon_images
        self.epoches = epoches
        self.batch_size = batch_size
        self.image_size = image_size


    def plot_model_history(self, model_history):
        # dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
        # epoches = range(1, self.epoches)

        # Loss Curves
        plt.figure(figsize=[8, 6])
        plt.plot(model_history.history['loss'], 'r', linewidth=3.0)
        plt.plot(model_history.history['val_loss'], 'b', linewidth=3.0)
        plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
        plt.xlabel('Epochs ', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.title('Loss Curves', fontsize=16)
        plt.savefig('T&V-loss.png')
        plt.show()

        # Accuracy Curves
        plt.figure(figsize=[8, 6])
        plt.plot(model_history.history['accuracy'], 'r', linewidth=3.0)
        plt.plot(model_history.history['val_accuracy'], 'b', linewidth=3.0)
        plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
        plt.xlabel('Epochs ', fontsize=16)
        plt.ylabel('Accuracy', fontsize=16)
        plt.title('Accuracy Curves', fontsize=16)
        plt.savefig('T&V-accuracy.png')
        plt.show()



    def data_generate(self):
        image_size = self.image_size
        batch_size = self.batch_size
        file_name_train = self.file_name_train
        file_name_test = self.file_name_test
        train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
        validation_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

        train_generator = train_data_generator.flow_from_directory(
            file_name_train,
            target_size=(image_size, image_size),
            batch_size=batch_size,
            color_mode="grayscale",
            class_mode="categorical"
        )

        validation_generator = validation_data_generator.flow_from_directory(
            file_name_test,
            target_size=(image_size, image_size),
            batch_size=batch_size,
            color_mode="grayscale",
            class_mode="categorical"
        )
        return train_generator, validation_generator

    def create_model(self):
        train_generator, validation_generator = self.data_generate()
        index_train_images = self.index_train_images
        batch_size = self.batch_size
        image_size = self.image_size
        epoches = self.epoches
        index_validation_images = self.index_validation_images
        # create Model
        model = tf.keras.Sequential()

        # relu = f(x) = max(0,x)
        model.add(tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(image_size, image_size, 1)))  # 48 x 48
        model.add(tf.keras.layers.Conv2D(64, kernel_size=(5, 5), activation='relu')) # 48 x 48
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2))) # 48 x 48
        model.add(tf.keras.layers.Dropout(0.25)) # 24 x 24

        model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu')) # 24 x 24
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2))) # 24 x 24
        model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))  # 12 x 12
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2))) # 12 x 12
        model.add(tf.keras.layers.Dropout(0.25)) # 6x 6

        # new layers
        model.add(tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation="relu")) # 6 x 6
        # model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2))) # 6 x 6
        model.add(tf.keras.layers.Dropout(0.25)) # 3 x 3

        model.add(tf.keras.layers.Flatten()) # 3 x 3
        # model.add(tf.keras.layers.Dense(1024, activation='relu')) # 9
        model.add(tf.keras.layers.Dense(512, activation='relu')) # 9
        # model.add(tf.keras.layers.Dropout(0.5)) # 9
        model.add(tf.keras.layers.Dropout(0.25)) # 9
        model.add(tf.keras.layers.Dense(7, activation='softmax'))

        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        model_info = model.fit_generator(
            train_generator,
            steps_per_epoch=index_train_images // batch_size,
            epochs=epoches,
            validation_data=validation_generator,
            validation_steps=index_validation_images // batch_size,
            callbacks=[tensorboard_callback]
        )
        print(model_info.history.keys())
        self.plot_model_history(model_info)
        return model

#save infos
    def save_model_info(self):
        model = self.create_model()
        # open path
        # Camera.open_path(r"Model infos")
        #
        # save_model = model.to_json()
        # with open("model_2.json", "w") as json_file:
        #     json_file.write(save_model)
        #
        # model.save_weights("model_3.h5")

        # save model structure in jason file
        save_model = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(save_model)

        # save trained model weight in .h5 file
        model.save_weights("model.h5")