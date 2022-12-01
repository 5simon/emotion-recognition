import os
from Recognition.Emotion.help_functions import *
import cv2
import numpy as np

from Recognition.Emotion.test_model import TestModel
import time

# to make to the pics color just switch between grayImage and frame
class Camera:
    """
        variables
    """
    which_camera = 0
    capture = cv2.VideoCapture(which_camera)  # 0 for laptop, 2 for external camera
    check_camera = capture.isOpened()
    frame = []
    gray_image = []
    key = 0
    frame_masked = []
    # video
    output_video = ""
    # Coordinate the detected features
    x = 0.0
    y = 0.0
    w = 0.0
    h = 0.0

    cropped_img = []
    def __init__(self, which_camera=0):
        self.which_camera = which_camera
        print("Camera processing...")


    """
        * closeCamera closes all windows if the window is Existing 
    """

    def close_camera(self):
        capture = self.capture
        output_video = self.output_video

        if capture:
            capture.release()
        if output_video:
            output_video.release()
        cv2.destroyAllWindows()

        print("break all windows")

    """
        * openCamera opens the webCamera and during it will be the face detected
    """

    def face_recognition(self):
        frame = self.frame
        x = self.x
        y = self.y
        w = self.w
        h = self.h


        face_cascade = cv2.CascadeClassifier(
            '/home/simon/BA/emotion-recognition/venv/lib/python3.10/site-packages/cv2/data/haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            raise IOError("unable to load haarcascade_frontalface_default.xml")
        eye_cascade = cv2.CascadeClassifier(
            '/home/simon/BA/emotion-recognition/venv/lib/python3.10/site-packages/cv2/data/haarcascade_eye_tree_eyeglasses.xml')
        if eye_cascade.empty():
            raise IOError("unable to load haarcascade_eye_tree_eyeglasses.xml")
        mouth_cascade = cv2.CascadeClassifier(
            '/home/simon/BA/emotion-recognition/venv/lib/python3.10/site-packages/cv2/data/haarcascade_mouth.xml')
        if mouth_cascade.empty():
            raise IOError("unable to load haarcascade_mcs_mouth.xml")

        upper_body_cascade = cv2.CascadeClassifier(
            '/home/simon/BA/emotion-recognition/venv/lib/python3.10/site-packages/cv2/data/haarcascade_upperbody.xml')
        if upper_body_cascade.empty():
            raise IOError("unanble to load haarcascade_fullbody.xml")

        nose_cascade = cv2.CascadeClassifier(
            '/home/simon/BA/emotion-recognition/venv/lib/python3.10/site-packages/cv2/data/haarcascade_mcs_nose.xml')
        if nose_cascade.empty():
            raise IOError("unable to load haarcascade_mcs_nose.xml")

        self.gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        '''
            detecting features in face
        '''
        face_detect = face_cascade.detectMultiScale(
            self.gray_image,
            scaleFactor=1.3,
            minNeighbors=5,
            #minSize=(30, 30),
        )
        eye_detect = eye_cascade.detectMultiScale(
            self.gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        mouth_detect = mouth_cascade.detectMultiScale(
            self.gray_image,
            scaleFactor=3,
            minNeighbors=5,
            minSize=(30, 30),
        )
        upper_body_detect = upper_body_cascade.detectMultiScale(
            self.gray_image,
            scaleFactor=1.01,
            minNeighbors=11,
            minSize=(50, 100),
        )
        '''
            for the mask as circle center_cordinate as faceCoordinate and radius have to be declared
        '''
        start_punkt, end_punkt = [0, 0], [0, 0]

        # coordinate for rectangle for face detection :: green
        for (x, y, w, h) in face_detect:
            cv2.rectangle(frame, (int(x), int(y-50)), (int(x) + int(w), int(y) + int(h+10)), (0, 255, 0), 4)
            start_punkt = int(x), int(y)
            end_punkt = int(x) + int(w), 2 * (int(y) + int(h))

            self.cropped_img = np.expand_dims(np.expand_dims(resize_images(frame, 48), -1), 0)


        # '''
        #     coordinate for rectangle for eye detection :: red
        # '''
        # for (self.x, self.y, self.w, self.h) in eye_detect:
        #     cv2.rectangle(self.frame, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 0, 255), 2)
        # '''
        #     coordinate for rectangle for mouth detection :: blue
        # '''
        # for (self.x, self.y, self.w, self.h) in mouth_detect:
        #     cv2.rectangle(self.frame, (self.x, self.y), (self.x + self.w, self.y + self.h), (255, 0, 0), 2)
        # '''
        #     coordinate for rectangle for body detection :: black
        # '''
        # for (self.x, self.y, self.w, self.h) in upper_body_detect:
        #     cv2.rectangle(self.frame, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 0, 0), 2)

        '''
           black Background from numpy
        '''
        black_background = np.zeros(frame.shape[:2], np.uint8)
        '''
            create rectangle around the detected face
        '''
        cv2.rectangle(black_background, start_punkt, end_punkt, (255, 255, 255), -1)
        '''
            insert the circle to the frame
        '''
        self.frame_masked = cv2.bitwise_and(frame, frame, mask=black_background)

    def open_camera(self):
        check_camera = self.check_camera
        capture = self.capture
        key = self.key

        if not check_camera:
            print("can't open the camera!")
            exit()

        while check_camera:
            ret, frame = capture.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # font = cv.FONT_HERSHEY_PLAIN
            # cv.putText(self.frame, str(datetime.now()), (20, 40), font, 2, (255, 255, 255,), 2, cv.LINE_AA)
            #self.face_recognition()
            '''
                only for temporary
            '''
            self.gray_image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            face_cascade = cv2.CascadeClassifier(
                '/home/simon/BA/emotion-recognition/venv/lib/python3.10/site-packages/cv2/data/haarcascade_frontalface_default.xml')
            face_detect = face_cascade.detectMultiScale(
                self.gray_image,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30),
            )
            start_punkt, end_punkt = [0, 0], [0, 0]
            for (self.x, self.y, self.w, self.h) in face_detect:
                cv2.rectangle(self.frame, (int(self.x), int(self.y - 50)),
                              (int(self.x) + int(self.w), int(self.y) + int(self.h + 10)), (0, 255, 0), 4)
                start_punkt = int(self.x), int(self.y)
                end_punkt = int(self.x) + int(self.w), 2 * (int(self.y) + int(self.h))

                self.cropped_img = np.expand_dims(np.expand_dims(resize_images(self.frame, 48), -1), 0)
            black_background = np.zeros(self.frame.shape[:2], np.uint8)
            cv2.rectangle(black_background, start_punkt, end_punkt, (255, 255, 255), -1)
            self.frame_masked = cv2.bitwise_and(self.frame, self.frame, mask=black_background)

            # time.sleep(4)
            test = TestModel(
                "/home/simon/BA/emotion-recognition/Recognition/Emotion/model_3/model_3.json",
                "/home/simon/BA/emotion-recognition/Recognition/Emotion/model_3/model_3.h5"
            )
            test.emotion_recognition(self.frame, self.gray_image, self.check_camera, self.x, self.y, self.h, self.w, face_detect)

            cv2.imshow("Camera", self.frame)

            key = cv2.waitKey(1)
            # q for quit
            if key & 0xFF == ord("q"):
                print("Exiting....")
                exiting = False
                break
            else:
                exiting = True
        self.close_camera()
        return exiting

    """
        * openPath opens the folder, in it the images will be saved to testing
    """

    @staticmethod
    def open_path(path_name):
        path_name = path_name
        path_existing = os.path.exists(path_name)
        if (not path_existing):
            os.makedirs(path_name)
        os.chdir(path_name)

    """
        * saveImageFromCamera saves the images in the path
    """

    def save_image_from_camera(self):
        self.open_path(r'testImages')
        image_index = 0
        self.open_camera()
        wait = 1
        while self.open_camera():
            if wait == 1:
                file_name = 'frame_' + str(image_index) + '.jpg'
                cv2.imwrite(file_name, self.frame_masked)
                image_index = image_index + 1
                #wait = 0
        self.close_camera()
