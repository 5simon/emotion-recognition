import tensorflow as tf
import cv2 as cv
from Recognition.openCamera import openCamera

print("tensorflow is installed and has the version: ", tf.__version__)
print("\ntensorflow is installed and has the version: ", cv.__version__)

openCamera()
