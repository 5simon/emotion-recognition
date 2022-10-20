import os

import cv2 as cv

'''
    loading alle images for testing and training
'''


def load_dataset(path_name, image_size=48):
    path_name = path_name
    # check if path exist
    if not os.path.exists(path_name):
        raise IOError("no such path like: " + path_name)
    # training and test data
    data = []

    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    # Emotion types
    classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    for emotion_type in classes:
        path = os.path.join(path_name, emotion_type)
        # check if path exist
        if not os.path.exists(path):
            raise IOError("no such path like: " + path)
        for image in os.listdir(path):
            try:
                images = cv.imread(os.path.join(path, image))
                # just to test if the images were loading

                # plt.imshow(cv.cvtColor(images, cv.COLOR_BGR2RGB))
                # plt.show()
                # print("old: ", images.shape)
                new_images = resize_images(images, image_size)

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
            train_images.append(images)
            train_labels.append(labels)
        print("the index of Train data: ", len(data))
        return train_images, train_labels
    elif path_name == "Recognition/archive/test/":
        for images, labels in data:
            test_images.append(images)
            test_labels.append(labels)
        print("the index of Test data: ", len(data))
        return test_images, test_labels


'''
    resize_images received 
        * images as array
        * new_size as integer
'''


def resize_images(images, new_size):
    image_size = new_size
    new_images = cv.resize(images, (image_size, image_size))

    return new_images


