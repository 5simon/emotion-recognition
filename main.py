from Recognition.Emotion.model import Model
from Recognition.face.camera import Camera
import tensorflow as tf
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--test", type=str, help="test model")
parser.add_argument("-l", "--train", type=str, help="train model")
mode = parser.parse_args()

if __name__ == "__main__":
    if mode.test:
        camera_window = Camera(which_camera=0)
        camera_window.save_image_from_camera()

    if mode.train:
        model1 = Model()
        model1.save_model_info()