from Recognition.Emotion.test_model import TestModel
from Recognition.face.camera import Camera
import cv2
if __name__ == "__main__":
    camera_window = Camera(which_camera=0)
    camera_window.save_image_from_camera()

    # test_1 = TestModel("Recognition/Emotion/model_1/model.json", "Recognition/Emotion/model_1/model.h5")
    # test_1.emotion_recognition()
