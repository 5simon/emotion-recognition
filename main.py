import tensorflow as tf
import cv2 as cv
from Recognition.saveImageAndVideo import saveVideoAndImageFromCamera

print("tensorflow is installed and has the version: ", tf.__version__)
print("\ntensorflow is installed and has the version: ", cv.__version__)
print("test")
saveVideoAndImageFromCamera()
