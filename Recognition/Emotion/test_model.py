#from Recognition.face.camera import *
import cv2
import matplotlib.pyplot as plt
import numpy as np
import datetime
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
    # emotion_model = open_emotion_model("model_1/model_2.json", "model_1/model_3.h5")
    #
    def open_emotion_model(self, filename_json, filename_h5):
        # filename_json can be like : 'model_1/model_2.json'
        # filename_h5 can be like "model_1/model_3.h5"

        file = open(filename_json, 'r')
        model_as_json = file.read()
        file.close()
        model = tf.keras.models.model_from_json(model_as_json)
        model.load_weights(filename_h5)

        return model

    def emotion_recognition(self, frame, gray_image, check_camera, x, y, h, w,face_detect, image_size=48):
        emotion_model = self.open_emotion_model(self.filename_json, self.filename_h5)
        for (x, y, h, w) in face_detect:

            #                               y: y + h, x: x + w
            roi_gray_frame = gray_image[int(y):int(y) + int(h), int(x):int(x) + int(w)]

            cropped_img = np.expand_dims(np.expand_dims(resize_images(roi_gray_frame, image_size), -1), 0)

            emotion_prediction = emotion_model.predict(cropped_img)
            max_index = int(np.argmax(emotion_prediction))
            prediction_in_percent = str(emotion_prediction[0][max_index])
            cv2.putText(frame, self.emotion_classes[max_index], (int(x+5), int(y-20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            #cv2.putText(frame, self.emotion_classes[max_index], (0, frame.shape[0] - 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            #cv2.putText(frame, prediction_in_percent, (0, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            print("you were at ", datetime.datetime.now().strftime("%H:%M:%S"), self.emotion_classes[max_index], " with ", prediction_in_percent, "%" )
