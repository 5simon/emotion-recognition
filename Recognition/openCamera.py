import cv2
import cv2 as cv

def saveVideoFromCamera():
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

        cv.imshow('img', frame)

        key = cv.waitKey(1)
        if key == ord("q"):
            break

    # close the videos
    capture.release()
    outputVideo.release()
    cv.destroyAllWindows()