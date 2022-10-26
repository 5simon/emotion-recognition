#from Recognition.face.camera import *
import cv2
import numpy as np
from Recognition.Emotion.help_functions import *


class TestModel:
    emotion_classes = {
        0: "Angry", 1: "Disgusted", 2: "Fear",
        3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
    }
    filename_json = ""
    filename_h5 = ""

    def __init__(self, filename_json, filename_h5):
        self.filename_json = filename_json
        self.filename_h5 = filename_h5
        print("I am testing your state emotion :) always be happy")

    # Calling  open_emotion_model be like
    # emotion_model = open_emotion_model("model_1/model.json", "model_1/model.h5")
    #
    def open_emotion_model(self, filename_json, filename_h5):
        # filename_json can be like : 'model_1/model.json'
        # filename_h5 can be like "model_1/model.h5"

        file = open(filename_json, 'r')
        model_as_json = file.read()
        file.close()
        model = tf.keras.models.model_from_json(model_as_json)
        model.load_weights(filename_h5)

        return model

    def emotion_recognition(self, frame,gray_image,check_camera, x, y,image_size=48):
        emotion_model = self.open_emotion_model(self.filename_json, self.filename_h5)

        # open camera with face detection
        # window = Camera(which_camera=0)
        # window.open_camera()


        #roi_gray_frame = gray_image[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(resize_images(gray_image, image_size), -1), 0)
        emotion_prediction = emotion_model.predict(cropped_img)
        max_index = int(np.argmax(emotion_prediction))
        cv2.putText(frame, self.emotion_classes[max_index], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
