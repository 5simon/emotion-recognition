import cv2
import dlib
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str, help="test methode by image")
parser.add_argument("-c", "--compare", type=str, help="compare between the tow methods")
mode = parser.parse_args()

def with_dlib(path):
    p = "/home/simon/BA/emotion-recognition/shape_predictor_68_face_landmarks.dat"

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    image_index_1 = 0
    image_index_2 = 0
    path = glob.glob(path)

    for image in path:
        img = cv2.imread(image)
        image_resized = cv2.resize(img, (512, 512))
        gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)


        if not rects:
            # print("No face")
            image_index_1 = image_index_1 + 1
        else:
            # print("face")

            image_index_2 = image_index_2 + 1

    print("No-DLIB: ", image_index_1)
    print("Yes-DLIB: ", image_index_2)
    return image_index_1, image_index_2

def with_cascade(path):
    image_index_1 = 0
    image_index_2 = 0
    path = glob.glob(path)

    for image in path:
        img = cv2.imread(image)
        image_resized = cv2.resize(img, (512, 512))
        gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(
            '/home/simon/BA/emotion-recognition/venv/lib/python3.10/site-packages/cv2/data/haarcascade_frontalface_default.xml')
        face_detect = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),
        )

        for (x, y, w, h) in face_detect:
            cv2.rectangle(image_resized, (int(x), int(y - 50)),
                          (int(x) + int(w), int(y) + int(h + 10)), (0, 255, 0), 4)

        if face_detect == ():
            # print("no face")
            image_index_1 = image_index_1 + 1

        else:
            # print("face")
            image_index_2 = image_index_2 + 1
    print("No-Cascade: ", image_index_1)
    print("Yes-Cascade: ", image_index_2)
        # cv2.imshow("sad", image_resized)
        # cv2.waitKey(0)
    return image_index_1, image_index_2


def process_single_image(image):
    img = cv2.imread(image)
    image_resized = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        '/home/simon/BA/emotion-recognition/venv/lib/python3.10/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(
        '/home/simon/BA/emotion-recognition/venv/lib/python3.10/site-packages/cv2/data/haarcascade_eye_tree_eyeglasses.xml')
    mouth_cascade = cv2.CascadeClassifier(
        '/home/simon/BA/emotion-recognition/venv/lib/python3.10/site-packages/cv2/data/haarcascade_mouth.xml')

    face_detect = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30),
    )
    eye_detect = eye_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    mouth_detect = mouth_cascade.detectMultiScale(
        gray,
        scaleFactor=3,
        minNeighbors=5,
        minSize=(30, 30),
    )
    for (x, y, w, h) in face_detect:
        cv2.rectangle(image_resized, (int(x), int(y - 50)),
                      (int(x) + int(w), int(y) + int(h + 10)), (0, 255, 0), 4)
    for (x, y, w, h) in eye_detect:
        # red
        cv2.rectangle(image_resized, (int(x), int(y)),
                      (int(x) + int(w), int(y) + int(h)), (0, 0, 255), 4)
    for (x, y, w, h) in mouth_detect:
        # blue
        cv2.rectangle(image_resized, (int(x), int(y)),
                      (int(x) + int(w), int(y) + int(h)), (255, 0, 0), 4)
    cv2.imshow("test cascade", image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if mode.image:
    process_single_image(image=mode.image)

if mode.compare:
    not_done_cascade,done_cascade = with_cascade("../../../bilderzumTesten/*")

    not_done_dlib,done_dlib= with_dlib("../../../bilderzumTesten/*")



    X = ["Landmarks", "Viola&jones"]
    done = [done_dlib, done_cascade]
    not_done = [not_done_dlib, not_done_cascade]

    X_axis = np.arange(len(X))

    plt.bar(X_axis - 0.2, done, 0.4, label='Verarbeitet')
    plt.bar(X_axis + 0.2, not_done, 0.4, label='nicht Verarbeitet')

    plt.xticks(X_axis, X)
    plt.xlabel("Methoden")
    plt.ylabel("Bilder")
    plt.legend()
    plt.show()

    # with 100 Pics
    # No-DLIB:  17
    # Yes-DLIB:  83

    # No-Cascade:  62
    # Yes-Cascade:  38

    # with Happy Pics (7215)
    # No - DLIB: 1526
    # Yes - DLIB: 5689

    # No - Cascade: 4318
    # Yes - Cascade: 2897

    # with 34356
    # No - Cascade: 22628
    # Yes - Cascade: 11728

    # Yes - DLIB: 24257
    # No - DLIB: 10099






