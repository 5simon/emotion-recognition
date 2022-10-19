from Recognition.face.camera import *
from Recognition.Emotion.dataTraining import *

print("tensorflow is installed and has the version: ", tf.__version__)
print("\nopencv is installed and has the version: ", cv.__version__)

# saveVideoAndImageFromCamera()
# faceRecognition()
# openCamere()

#cameraWindow = Camera()
#cameraWindow.saveImageFromCamera()
#cameraWindow.closeCamera()

data = TrainingData()
data.loadDataSet()

