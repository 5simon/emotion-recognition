import tensorflow as tf
import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

class TrainingData:
    def loadDataSet(self):
        pathName = "Recognition/archive/train/"
        # Emotion types
        classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

        for emotionType in classes:
            path = os.path.join(pathName, emotionType)
            for image in os.listdir(path):
                images = cv.imread(os.path.join(path, image))
                plt.imshow(cv.cvtColor(images, cv.COLOR_BGR2RGB))
                plt.show()
                break
            break