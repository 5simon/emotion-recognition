import cv2
import cv2 as cv

def saveVideoFromCamera():
    capture = cv.VideoCapture(0)

    fourcc = cv.VideoWriter_fourcc(*"XVID")
    nameVideo = "./video.avi"
    output = cv.VideoWriter(nameVideo, fourcc, 20.0, (640, 480))

    '''
        TODO:
            make it able to save more videos than one
    '''

    if not capture.isOpened():
        print("can't open the camera!")
        exit()

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # to save the video
        output.write(frame)

        cv.imshow('img', frame)

        key = cv.waitKey(1)
        if key == ord("q"):
            break

    capture.release()
    output.release()
    cv.destroyAllWindows()