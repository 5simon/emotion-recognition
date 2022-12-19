import cv2
import dlib
import glob
import numpy as np
import matplotlib.pyplot as plt


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
            print("No face")
            image_index_1 = image_index_1 + 1
        else:
            print("face")

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
            print("no face")
            image_index_1 = image_index_1 + 1

        else:
            print("face")
            image_index_2 = image_index_2 + 1
    print("No-Cascade: ", image_index_1)
    print("Yes-Cascade: ", image_index_2)
        # cv2.imshow("sad", image_resized)
        # cv2.waitKey(0)
    return image_index_1, image_index_2

not_done_cascade,done_cascade = with_cascade("Pics_for_test/*")

not_done_dlib,done_dlib= with_dlib("Pics_for_test/*")



X = ["Landmarks", "Viola&jones"]
done = [done_dlib, done_cascade]
not_done = [not_done_dlib, not_done_cascade]

X_axis = np.arange(len(X))

plt.bar(X_axis - 0.2, done, 0.4, label='bearbeitet')
plt.bar(X_axis + 0.2, not_done, 0.4, label='nicht bearbeitet')

plt.xticks(X_axis, X)
plt.xlabel("Methoden")
plt.ylabel("Bilder")
plt.legend()
plt.show()

# No-DLIB:  17
# Yes-DLIB:  83

# No-Cascade:  62
# Yes-Cascade:  38
