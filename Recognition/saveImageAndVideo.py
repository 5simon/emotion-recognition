import cv2 as cv
import os
from datetime import datetime

def saveVideoAndImageFromCamera():
    path = r'testImages'
    pathExist = os.path.exists(path)
    if (not pathExist):
        os.makedirs(path)
    os.chdir(path)

    imageIndex = 0
    wait = 0
    # open camera
    capture = cv.VideoCapture(0)

    # set suffix
    fourcc = cv.VideoWriter_fourcc(*"XVID")

    # output video with (name, suffix, framrate, size of the image)
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

        # to save the video
        outputVideo.write(frame)

        # to save each frame
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
            imageIndex= imageIndex + 1
            wait= 0

    # close the videos
    capture.release()
    outputVideo.release()
    cv.destroyAllWindows()