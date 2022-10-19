import os

import cv2 as cv
import numpy as np

# to make to the pics color just switch between grayImage and frame
class Camera:
    """
        variables
    """
    capture = cv.VideoCapture(2)  # 0 for laptop, 2 for external camera
    check_camera = capture.isOpened()
    frame = []
    gray_image = []
    key = 0
    frame_masked = []
    # video
    output_video = ""

    def __init__(self):
        print("Camera processing...")

    """
        * closeCamera closes all windows if the window is Existing 
    """

    def close_camera(self):
        if self.capture:
            self.capture.release()
        if self.output_video:
            self.output_video.release()
        cv.destroyAllWindows()

        print("break all windows")

    """
        * openCamera opens the webCamera and during it will be the face detected
    """

    def face_recognition(self):
        face_cascade = cv.CascadeClassifier(
            '/home/simon/BA/emotion-recognition/venv/lib/python3.10/site-packages/cv2/data/haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            raise IOError("unable to load haarcascade_frontalface_default.xml")
        eye_cascade = cv.CascadeClassifier(
            '/home/simon/BA/emotion-recognition/venv/lib/python3.10/site-packages/cv2/data/haarcascade_eye_tree_eyeglasses.xml')
        if eye_cascade.empty():
            raise IOError("unable to load haarcascade_eye_tree_eyeglasses.xml")
        mouth_cascade = cv.CascadeClassifier(
            '/home/simon/BA/emotion-recognition/venv/lib/python3.10/site-packages/cv2/data/haarcascade_mouth.xml')
        if mouth_cascade.empty():
            raise IOError("unable to load haarcascade_mcs_mouth.xml")

        upper_body_cascade = cv.CascadeClassifier(
            '/home/simon/BA/emotion-recognition/venv/lib/python3.10/site-packages/cv2/data/haarcascade_upperbody.xml')
        if upper_body_cascade.empty():
            raise IOError("unanble to load haarcascade_fullbody.xml")

        nose_cascade = cv.CascadeClassifier(
            '/home/simon/BA/emotion-recognition/venv/lib/python3.10/site-packages/cv2/data/haarcascade_mcs_nose.xml')
        if nose_cascade.empty():
            raise IOError("unable to load haarcascade_mcs_nose.xml")
        self.gray_image = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)

        '''
            detecting features in face
        '''
        face_detect = face_cascade.detectMultiScale(
            self.gray_image,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30),
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
            cv.rectangle(self.gray_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            start_punkt = x, y
            end_punkt = x + w, 2 * (y + h)

        '''
            coordinate for rectangle for eye detection :: red 
        '''
        for (x, y, w, h) in eye_detect:
            cv.rectangle(self.gray_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        '''
            coordinate for rectangle for mouth detection :: blue
        '''
        for (x, y, w, h) in mouth_detect:
            cv.rectangle(self.gray_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        '''
            coordinate for rectangle for body detection :: black
        '''
        for (x, y, w, h) in upper_body_detect:
            cv.rectangle(self.gray_image, (x, y), (x + w, y + h), (0, 0, 0), 2)

        '''
           black Background from numpy
        '''
        black_background = np.zeros(self.gray_image.shape[:2], np.uint8)
        '''
            create circle around the detected face
        '''
        cv.rectangle(black_background, start_punkt, end_punkt, (255, 255, 255), -1)
        '''
            insert the circle to the frame
        '''
        self.frame_masked = cv.bitwise_and(self.gray_image, self.gray_image, mask=black_background)

    def open_camere(self):
        if not self.check_camera:
            print("can't open the camera!")
            exit()

        while self.check_camera:
            ret, self.frame = self.capture.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # font = cv.FONT_HERSHEY_PLAIN
            # cv.putText(self.frame, str(datetime.now()), (20, 40), font, 2, (255, 255, 255,), 2, cv.LINE_AA)
            self.face_recognition()
            cv.imshow("Camera", self.gray_image)

            self.key = cv.waitKey(1)
            # q for quit
            if self.key == ord("q"):
                print("Exiting....")
                return False
                break
            else:
                return True

    """
        * openPath opens the folder, in it the images will be saved to testing
    """

    def open_path(self):
        path_name = r'testImages'
        path_existing = os.path.exists(path_name)
        if (not path_existing):
            os.makedirs(path_name)
        os.chdir(path_name)

    """
        * saveImageFromCamera saves the images in the path
    """

    def save_image_from_camera(self):
        self.open_path()
        image_index = 0
        wait = 0
        self.open_camere()
        while self.open_camere():
            wait = 1
            if wait == 1:
                file_name = 'frame_' + str(image_index) + '.jpg'
                cv.imwrite(file_name, self.frame_masked)
                image_index = image_index + 1
                wait = 0
        self.close_camera()
