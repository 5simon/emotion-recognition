import os

import cv2 as cv
import matplotlib.pyplot as plt


'''
    loading alle images for testing and training
'''
def load_dataset(path_name):
    path_name = path_name
    data = []
    images_train = []
    label_train = []
    images_test = []
    label_test = []
    # Emotion types
    classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    for emotion_type in classes:
        path = os.path.join(path_name, emotion_type)
        for image in os.listdir(path):
            try:
                images = cv.imread(os.path.join(path, image))
                #just to test if the images were loading

                # plt.imshow(cv.cvtColor(images, cv.COLOR_BGR2RGB))
                # plt.show()
                # print("old: ", images.shape)
                new_images = resize_images(images, 224)

                # just to test function

                # plt.imshow(cv.cvtColor(new_images, cv.COLOR_BGR2RGB))
                # plt.show()
                # print("new: ", new_images.shape)
                data.append([new_images, emotion_type])
            except Exception as error:
                print(error)
                pass

    if path_name == "Recognition/archive/train/":
        for images, labels in data:
            images_train.append(images)
            label_train.append(labels)
            print(labels)
        print("the index of Train data: ", len(data))
    elif path_name == "Recognition/archive/test/":
        for images, labels in data:
            images_test.append(images)
            label_test.append(labels)
            label_train.append(labels)
        print("the index of Test data: ", len(data))



'''
    resize_images received 
        * images as array
        * new_size as integer
'''
def resize_images(images, new_size):
    image_size = new_size
    new_images = cv.resize(images, (image_size, image_size))

    return new_images
