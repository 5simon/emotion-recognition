import os
from datetime import datetime

import cv2 as cv
import numpy as np


def saveVideoAndImageFromCamera():
    path = r'testImages'
    pathExist = os.path.exists(path)
    if (not pathExist):
        os.makedirs(path)
    os.chdir(path)

    imageIndex = 0
    wait = 0
    '''
        open camera
    '''
    capture = cv.VideoCapture(0)

    '''
        set suffix
    '''
    fourcc = cv.VideoWriter_fourcc(*"XVID")

    '''
        output video with (name, suffix, framrate, size of the image)
    '''
    nameVideo = "./video.avi"
    outputVideo = cv.VideoWriter(nameVideo, fourcc, 20.0, (640, 480))
    if not capture.isOpened():
        print("can't open the camera!")
        exit()

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        '''
            to save the video
        '''
        outputVideo.write(frame)

        '''
             to save each frame
        '''
        font = cv.FONT_HERSHEY_PLAIN
        cv.putText(frame, str(datetime.now()), (20, 40), font, 2, (255, 255, 255,), 2, cv.LINE_AA)

        cv.imshow('img', frame)

        key = cv.waitKey(100)
        wait = wait + 100
        if key == ord("q"):
            break
        if wait == 500:
            filename = 'frame_' + str(imageIndex) + '.jpg'
            cv.imwrite(filename, frame)
            imageIndex = imageIndex + 1
            wait = 0

    # close the videos
    capture.release()
    outputVideo.release()
    cv.destroyAllWindows()


def faceRecognition():
    faceCascade = cv.CascadeClassifier('venv/lib/python3.10/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    if faceCascade.empty():
        raise IOError("unable to load haarcascade_frontalface_default.xml")
    eyeCascade = cv.CascadeClassifier('venv/lib/python3.10/site-packages/cv2/data/haarcascade_eye_tree_eyeglasses.xml')
    if eyeCascade.empty():
        raise IOError("unable to load haarcascade_eye_tree_eyeglasses.xml")

    mouthCascade = cv.CascadeClassifier('venv/lib/python3.10/site-packages/cv2/data/haarcascade_mcs_mouth.xml')
    if eyeCascade.empty():
        raise IOError("unable to load haarcascade_mcs_mouth.xml")

    noseCascade = cv.CascadeClassifier('venv/lib/python3.10/site-packages/cv2/data/haarcascade_mcs_nose.xml')
    if eyeCascade.empty():
        raise IOError("unable to load haarcascade_mcs_nose.xml")

    capture = cv.VideoCapture(0)
    capture.set(3, 640)
    capture.set(4, 480)
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        grayImage = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

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
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
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
        blackBackground = np.zeros(frame.shape[:2], np.uint8)
        '''
            create circle around the detected face
        '''
        cv.circle(blackBackground, faceCoordinate, radius, (255, 255, 255), -1)
        '''
            insert the circle to the frame
        '''
        frameMasked = cv.bitwise_and(frame, frame, mask=blackBackground)
        cv.imshow("face recognition", frameMasked)
        key = cv.waitKey(1)
        if key == ord('q'):
            break
    capture.release()
    cv.destroyAllWindows()
