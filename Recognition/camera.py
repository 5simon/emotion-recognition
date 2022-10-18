import os

import cv2 as cv
import numpy as np


class Camera:
    """
        variables
    """
    capture = cv.VideoCapture(2)  # 0 for laptop, 2 for external camera
    checkCamera = capture.isOpened()
    frame = []
    key = 0
    frameMasked = []
    # video
    outputVideo = ""

    def __init__(self):
        print("Camera processing...")

    """
        * closeCamera closes all windows if the window is Existing 
    """

    def closeCamera(self):
        if self.capture:
            self.capture.release()
        if self.outputVideo:
            self.outputVideo.release()
        cv.destroyAllWindows()

        print("break all windows")

    """
        * openCamera opens the webCamera and during it will be the face detected
    """

    def faceRecognition(self):
        faceCascade = cv.CascadeClassifier(
            '/home/simon/BA/emotion-recognition/venv/lib/python3.10/site-packages/cv2/data/haarcascade_frontalface_default.xml')
        if faceCascade.empty():
            raise IOError("unable to load haarcascade_frontalface_default.xml")
        eyeCascade = cv.CascadeClassifier(
            '/home/simon/BA/emotion-recognition/venv/lib/python3.10/site-packages/cv2/data/haarcascade_eye_tree_eyeglasses.xml')
        if eyeCascade.empty():
            raise IOError("unable to load haarcascade_eye_tree_eyeglasses.xml")

        mouthCascade = cv.CascadeClassifier(
            '/home/simon/BA/emotion-recognition/venv/lib/python3.10/site-packages/cv2/data/haarcascade_mcs_mouth.xml')
        if mouthCascade.empty():
            raise IOError("unable to load haarcascade_mcs_mouth.xml")

        noseCascade = cv.CascadeClassifier(
            '/home/simon/BA/emotion-recognition/venv/lib/python3.10/site-packages/cv2/data/haarcascade_mcs_nose.xml')
        if noseCascade.empty():
            raise IOError("unable to load haarcascade_mcs_nose.xml")
        grayImage = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)

        '''
            detecting features in face
        '''
        faceDetect = faceCascade.detectMultiScale(
            grayImage,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30),
        )
        eyeDetect = eyeCascade.detectMultiScale(
            grayImage,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        mouthDetect = mouthCascade.detectMultiScale(
            grayImage,
            scaleFactor=3,
            minNeighbors=5,
            minSize=(30, 30),
        )
        noseDetect = noseCascade.detectMultiScale(
            grayImage,
            scaleFactor=1.1,
            minNeighbors=11,
            minSize=(30, 30),
        )
        '''
            for the mask as circle center_cordinate as faceCoordinate and radius have to be declared
        '''
        faceCoordinate, radius = [0, 0], 0

        # coordinate for rectangle for face detection :: green
        for (x, y, w, h) in faceDetect:
            cv.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            faceCoordinate = [x + w // 2, y + h // 2]
            radius = h // 2

        '''
            this code have been commented to make the image clearly
            maby it will be used later
        '''

        '''
            coordinate for rectangle for eye detection :: red 
        '''
        # for (x, y, w, h) in eyeDetect:
        #     cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        '''
            coordinate for rectangle for mouth detection :: blue
        '''
        # for (x, y, w, h) in mouthDetect:
        #     cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        '''
            coordinate for rectangle for nose detection :: black
        '''
        # for (x, y, w, h) in noseDetect:
        #     cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

        '''
           black Background from numpy
        '''
        blackBackground = np.zeros(self.frame.shape[:2], np.uint8)
        '''
            create circle around the detected face
        '''
        cv.circle(blackBackground, faceCoordinate, radius, (255, 255, 255), -1)
        '''
            insert the circle to the frame
        '''
        self.frameMasked = cv.bitwise_and(self.frame, self.frame, mask=blackBackground)

    def openCamere(self):
        if not self.checkCamera:
            print("can't open the camera!")
            exit()

        while self.checkCamera:
            ret, self.frame = self.capture.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # font = cv.FONT_HERSHEY_PLAIN
            # cv.putText(self.frame, str(datetime.now()), (20, 40), font, 2, (255, 255, 255,), 2, cv.LINE_AA)
            self.faceRecognition()
            cv.imshow("Camera", self.frame)

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

    def openPath(self):
        pathName = r'testImages'
        pathExisting = os.path.exists(pathName)
        if (not pathExisting):
            os.makedirs(pathName)
        os.chdir(pathName)

    """
        * saveImageFromCamera saves the images in the path
    """

    def saveImageFromCamera(self):
        self.openPath()
        imageIndex = 0
        wait = 0
        self.openCamere()
        while self.openCamere():
            wait = 1
            if wait == 1:
                filename = 'frame_' + str(imageIndex) + '.jpg'
                cv.imwrite(filename, self.frameMasked)
                imageIndex = imageIndex + 1
                wait = 0
        self.closeCamera()
