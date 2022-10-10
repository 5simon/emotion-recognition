import cv2
import cv2 as cv

def openCamera():
    capture = cv.VideoCapture(0)
    if not capture.isOpened():
        print("can't open the camera!")
        exit()

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv.imshow('frame', grayFrame)

        key = cv.waitKey(1)
        if key == ord("q"):
            break

    capture.release()
    cv.destroyAllWindows()