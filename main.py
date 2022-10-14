import tensorflow as tf

from Recognition.camera import *

print("tensorflow is installed and has the version: ", tf.__version__)
print("\nopencv is installed and has the version: ", cv.__version__)

# saveVideoAndImageFromCamera()
# faceRecognition()
# openCamere()

cameraWindow = Camera()
#cameraWindow.openCamere()
cameraWindow.saveVideoAndImageFromCamera()
#cameraWindow.closeCamera()
