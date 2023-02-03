import matplotlib.pyplot as plt

from Recognition.Emotion.model import Model
from Recognition.face.camera import Camera
import tensorflow as tf
import cv2
import argparse

# make it able to choose the path for test data(so let give the path of .h5 and json file)
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--testCam", type=str, help="test model by webcam")
parser.add_argument("-i", "--path", type=str, nargs='+', help="test model by image")
parser.add_argument("-l", "--train", type=str,  nargs='+', help="train model")
mode = parser.parse_args()

if __name__ == "__main__":
    if mode.testCam:
        camera_window = Camera(which_camera=0, which_data= mode.testCam)
        camera_window.open_camera(full_screen=True)
    if mode.path:
        # camera_window.determine_emotion_by_image(frame_path="pictures_for_test/natural.jpg", size=0)
        camera_window = Camera(which_camera=2, which_data= mode.path[1])
        camera_window.determine_emotion_by_image(frame_path=mode.path[0], size=0)
    if mode.train:
        model1 = Model(epoches=75)
        model1.save_model_info(which_model=mode.train[0], which_strategy=mode.train[1])