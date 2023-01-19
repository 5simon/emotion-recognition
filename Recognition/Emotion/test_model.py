#from Recognition.face.camera import *
import random

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

            emotion_prediction = emotion_model.predict(cropped_img) * 100
            max_index = int(np.argmax(emotion_prediction))
            prediction_in_percent = str(("%.2f" % emotion_prediction[0][max_index]))
            cv2.putText(frame, self.emotion_classes[max_index] + ": " + prediction_in_percent + "%", (int(x+5), int(y-20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # put all emotions:
            color_white = (0, 0, 0)
            color_black = (255,255,255)
            color_green = (0, 255, 0)
            if self.emotion_classes[0] == self.emotion_classes[max_index]:
                cv2.putText(frame, self.emotion_classes[0] + ": " + str("%.2f" % emotion_prediction[0][0]) + "%",
                            (0, frame.shape[0] - 500 + 175), cv2.FONT_HERSHEY_SIMPLEX, 1, color_green, 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, self.emotion_classes[0] + ": " + str("%.2f" % emotion_prediction[0][0]) + "%",
                            (0, frame.shape[0] - 500 + 175), cv2.FONT_HERSHEY_SIMPLEX, 1, color_white, 2, cv2.LINE_AA)

            if self.emotion_classes[1] == self.emotion_classes[max_index]:
                cv2.putText(frame, self.emotion_classes[1] + ": " + str("%.2f" % emotion_prediction[0][1]) + "%",
                            (0, frame.shape[0] - 450 + 175), cv2.FONT_HERSHEY_SIMPLEX, 1, color_green, 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, self.emotion_classes[1] + ": " + str("%.2f" % emotion_prediction[0][1]) + "%",
                            (0, frame.shape[0] - 450 + 175), cv2.FONT_HERSHEY_SIMPLEX, 1, color_black, 2, cv2.LINE_AA)

            if self.emotion_classes[2] == self.emotion_classes[max_index]:
                cv2.putText(frame, self.emotion_classes[2] + ": " + str("%.2f" % emotion_prediction[0][2]) + "%",
                            (0, frame.shape[0] - 400 + 175), cv2.FONT_HERSHEY_SIMPLEX, 1, color_green, 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, self.emotion_classes[2] + ": " + str("%.2f" % emotion_prediction[0][2]) + "%",
                            (0, frame.shape[0] - 400 + 175), cv2.FONT_HERSHEY_SIMPLEX, 1, color_white, 2, cv2.LINE_AA)

            if self.emotion_classes[3] == self.emotion_classes[max_index]:
                cv2.putText(frame, self.emotion_classes[3] + ": " + str("%.2f" % emotion_prediction[0][3]) + "%",
                            (0, frame.shape[0] - 350 + 175), cv2.FONT_HERSHEY_SIMPLEX, 1, color_green, 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, self.emotion_classes[3] + ": " + str("%.2f" % emotion_prediction[0][3]) + "%",
                            (0, frame.shape[0] - 350 + 175), cv2.FONT_HERSHEY_SIMPLEX, 1, color_black, 2, cv2.LINE_AA)

            if self.emotion_classes[4] == self.emotion_classes[max_index]:
                cv2.putText(frame, self.emotion_classes[4] + ": " + str("%.2f" % emotion_prediction[0][4]) + "%",
                            (0, frame.shape[0] - 300 + 175), cv2.FONT_HERSHEY_SIMPLEX, 1, color_green, 2, cv2.LINE_AA)

            else:
                cv2.putText(frame, self.emotion_classes[4] + ": " + str("%.2f" % emotion_prediction[0][4]) + "%",
                            (0, frame.shape[0] - 300 + 175), cv2.FONT_HERSHEY_SIMPLEX, 1, color_white, 2, cv2.LINE_AA)

            if self.emotion_classes[5] == self.emotion_classes[max_index]:
                cv2.putText(frame, self.emotion_classes[5] + ": " + str("%.2f" % emotion_prediction[0][5]) + "%",
                            (0, frame.shape[0] - 250 + 175), cv2.FONT_HERSHEY_SIMPLEX, 1, color_green, 2, cv2.LINE_AA)

            else:
                cv2.putText(frame, self.emotion_classes[5] + ": " + str("%.2f" % emotion_prediction[0][5]) + "%",
                            (0, frame.shape[0] - 250 + 175), cv2.FONT_HERSHEY_SIMPLEX, 1, color_black, 2, cv2.LINE_AA)

            if self.emotion_classes[6] == self.emotion_classes[max_index]:
                cv2.putText(frame, self.emotion_classes[6] + ": " + str("%.2f" % emotion_prediction[0][6]) + "%",
                            (0, frame.shape[0] - 200 + 175), cv2.FONT_HERSHEY_SIMPLEX, 1, color_green, 2, cv2.LINE_AA)

            else:
                cv2.putText(frame, self.emotion_classes[6] + ": " + str("%.2f" % emotion_prediction[0][6]) + "%",
                            (0, frame.shape[0] - 200 + 175), cv2.FONT_HERSHEY_SIMPLEX, 1, color_white, 2, cv2.LINE_AA)

            # print all emotions:
            print("you were at ", datetime.datetime.now().strftime("%H:%M:%S"), self.emotion_classes[0], " with ", str("%.2f" % emotion_prediction[0][0]), "%" + " = " + str(emotion_prediction[0][0]))
            print("you were at ", datetime.datetime.now().strftime("%H:%M:%S"), self.emotion_classes[1], " with ", str("%.2f" % emotion_prediction[0][1]), "%" + " = " + str(emotion_prediction[0][1]))
            print("you were at ", datetime.datetime.now().strftime("%H:%M:%S"), self.emotion_classes[2], " with ", str("%.2f" % emotion_prediction[0][2]), "%" + " = " + str(emotion_prediction[0][2]))
            print("you were at ", datetime.datetime.now().strftime("%H:%M:%S"), self.emotion_classes[3], " with ", str("%.2f" % emotion_prediction[0][3]), "%" + " = " + str(emotion_prediction[0][3]))
            print("you were at ", datetime.datetime.now().strftime("%H:%M:%S"), self.emotion_classes[4], " with ", str("%.2f" % emotion_prediction[0][4]), "%" + " = " + str(emotion_prediction[0][4]))
            print("you were at ", datetime.datetime.now().strftime("%H:%M:%S"), self.emotion_classes[5], " with ", str("%.2f" % emotion_prediction[0][5]), "%" + " = " + str(emotion_prediction[0][5]))
            print("you were at ", datetime.datetime.now().strftime("%H:%M:%S"), self.emotion_classes[6], " with ", str("%.2f" % emotion_prediction[0][6]), "%" + " = " + str(emotion_prediction[0][6]))

