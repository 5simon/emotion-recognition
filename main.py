import matplotlib.pyplot as plt

from Recognition.face.camera import *
from Recognition.Emotion.helpFunctions import *
import tensorflow as tf

print("tensorflow is installed and has the version: ", tf.__version__)
print("\nopencv is installed and has the version: ", cv.__version__)

# camera_window = Camera()
# camera_window.save_image_from_camera()


images_train, labels_train = load_dataset("Recognition/archive/train/")
images_test, labels_test = load_dataset("Recognition/archive/test/")

# test function
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(True)
    plt.imshow(images_train[i], cmap=plt.cm.binary)
    plt.xlabel(labels_train[i])
plt.show()
