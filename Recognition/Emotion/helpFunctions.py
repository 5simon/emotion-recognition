import os

import cv2 as cv
import matplotlib.pyplot as plt


'''
    loading alle images for testing and training
'''
def load_dataset():
    path_name = "Recognition/archive/train/"
    # Emotion types
    classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    for emotion_type in classes:
        path = os.path.join(path_name, emotion_type)
        for image in os.listdir(path):
            images = cv.imread(os.path.join(path, image))
            plt.imshow(cv.cvtColor(images, cv.COLOR_BGR2RGB))
            plt.show()
            print("old: ", images.shape)
            new_images = resize_images(images, 224)
            plt.imshow(cv.cvtColor(new_images, cv.COLOR_BGR2RGB))
            plt.show()
            break
        break


'''
    resize_images received 
        * images as array
        * new_size as integer
'''
def resize_images(images, new_size):
    image_size = new_size
    new_images = cv.resize(images, (image_size, image_size))

    return new_images
