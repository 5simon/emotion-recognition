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
        cv.imshow('frame', frame)

        key = cv.waitKey(0)
        if key == ord("q"):
            break

    capture.release()
    cv.destroyAllWindows()



