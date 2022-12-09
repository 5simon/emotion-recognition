import os
from time import sleep
import argparse

import cv2
import dlib
import numpy as np
import glob
from Emotion.helpFunctions import resize_images

# Define what landmarks you want:
JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_BRIDGE_POINTS = list(range(27, 31))
LOWER_NOSE_POINTS = list(range(31, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))
ALL_POINTS = list(range(0, 68))

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--face", type=str, help= "save images with detected face")
parser.add_argument("-em", "--eyeMouth", type=str, help="save images with just detected mouth and eyes")

mode = parser.parse_args()
def draw_shape_lines_range(np_shape, image, range_points, is_closed=False):
    """Draws the shape using lines to connect the different points"""

    np_shape_display = np_shape[range_points]
    points = np.array(np_shape_display, dtype=np.int32)
    cv2.polylines(image, [points], is_closed, (255, 255, 0), thickness=1, lineType=cv2.LINE_8)
    # print(points)
    # print("-------------------")
    # cv2.rectangle(image, (ALL_POINTS[23], ALL_POINTS[27]), (ALL_POINTS[17], ALL_POINTS[29]), (0, 255, 0), 1)


def draw_shape_points_pos_range(np_shape, image, points):
    """Draws the shape using points and position for every landmark filtering by points parameter"""

    np_shape_display = np_shape[points]
    draw_shape_points_pos(np_shape_display, image)


def draw_shape_points_pos(np_shape, image):
    """Draws the shape using points and position for every landmark"""

    for idx, (x, y) in enumerate(np_shape):
        # Draw the positions for every detected landmark:
        cv2.putText(image, str(idx + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255))

        # Draw a point on every landmark position:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)


def draw_shape_points_range(np_shape, image, points):
    """Draws the shape using points for every landmark filtering by points parameter"""

    np_shape_display = np_shape[points]
    draw_shape_points(np_shape_display, image)


def draw_shape_points(np_shape, image):
    """Draws the shape using points for every landmark"""

    # Draw a point on every landmark position:
    for (x, y) in np_shape:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)


def shape_to_np(dlib_shape, dtype="int"):
    """Converts dlib shape object to numpy array"""

    # Initialize the list of (x,y) coordinates
    coordinates = np.zeros((dlib_shape.num_parts, 2), dtype=dtype)

    # Loop over all facial landmarks and convert them to a tuple with (x,y) coordinates:
    for i in range(0, dlib_shape.num_parts):
        coordinates[i] = (dlib_shape.part(i).x, dlib_shape.part(i).y)

    # Return the list of (x,y) coordinates:
    return coordinates

def preprocessing_images_detect_eye_mouth(old_path, new_path):
    p = "/home/simon/BA/emotion-recognition/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)
    frame = cv2.imread("frame_30.jpg")
    image_resized = cv2.resize(frame, (512, 512))


    image_index_1 = 0
    image_index_2 = 0
    path = glob.glob(old_path)

    for image in path:
        img = cv2.imread(image)
        image_resized = cv2.resize(img, (512, 512))
        gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        if not rects:
            file_name = new_path + '/frame_not_detected_' + str(image_index_1) + '.jpg'
            cv2.imwrite(file_name, image_resized)
            image_index_1 = image_index_1 + 1

            # cv2.imshow("just the face", image_resized)
        else:
            for (i, rect) in enumerate(rects):
                # Draw a box around the face:
                cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 1)
                # Get the shape using the predictor:
                shape = predictor(gray, rect)

                start_punkt_1 = shape.part(17).x, shape.part(17).y - 30
                end_punkt_1 = shape.part(28).x, shape.part(28).y + 15

                start_punkt_2 = shape.part(15).x, shape.part(18).y - 15
                end_punkt_2 = shape.part(28).x, shape.part(28).y + 15

                start_punkt_3 = shape.part(49).x - 30, shape.part(49).y - 15
                end_punkt_3 = shape.part(55).x + 30, shape.part(55).y + 15

                # eye = cv2.rectangle(image_resized, (shape.part(17).x, shape.part(17).y - 30),
                #                      (shape.part(28).x, shape.part(28).y + 15), (255, 0, 0), 5)
                # eye = cv2.rectangle(image_resized, (shape.part(15).x, shape.part(18).y - 15),
                #                      (shape.part(28).x, shape.part(28).y + 15), (0, 255, 0), 5)
                # mouth = cv2.rectangle(image_resized, (shape.part(49).x - 30, shape.part(49).y - 15),
                #                      (shape.part(55).x + 30, shape.part(55).y + 15), (0, 0, 255), 5)
            black_background = np.zeros(image_resized.shape[:2], np.uint8)

            cv2.rectangle(black_background, start_punkt_1, end_punkt_1, (255, 255, 255), -1)
            cv2.rectangle(black_background, start_punkt_2, end_punkt_2, (255, 255, 255), -1)
            cv2.rectangle(black_background, start_punkt_3, end_punkt_3, (255, 255, 255), -1)

            frame_masked = cv2.bitwise_and(image_resized, image_resized, mask=black_background)
            # cv2.imshow("just the face", frame_masked)

            # os.chdir(r"test")

            file_name = new_path + '/frame_' + str(image_index_2) + '.jpg'
            cv2.imwrite(file_name, frame_masked)
            image_index_2 = image_index_2 + 1

            # Display the resulting frame
            # cv2.imshow("Landmarks detection using dlib", image_resized)
            # cv2.waitKey(0)
    print(image_index_1)
    print(image_index_2)


def preprocessing_images_detect_face(old_path, new_path):
    # Name of the two shape predictors:
    p = "/home/simon/BA/emotion-recognition/shape_predictor_68_face_landmarks.dat"
    # p = "shape_predictor_5_face_landmarks.dat"

    # Initialize frontal face detector and shape predictor:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    # Create VideoCapture object to get images from the webcam:
    # video_capture = cv2.VideoCapture(0)
    #
    # frame_width = int(video_capture.get(3))
    # frame_height = int(video_capture.get(4))
    #
    # size = (frame_width, frame_height)
    #
    image_index = 0
    # path = glob.glob("archive/train/sad/*")
    path = glob.glob(old_path)

    for image in path:
        img = cv2.imread(image)
        image_resized = cv2.resize(img, (512, 512))
        gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        start_punkt, end_punkt = [0, 0], [0, 0]

        if not rects:
            file_name = new_path + '/frame_not_detected_' + str(image_index) + '.jpg'
            cv2.imwrite(file_name, image_resized)
            image_index = image_index + 1

            # cv2.imshow("just the face", image_resized)
        else:
            for (i, rect) in enumerate(rects):
                # cv2.rectangle(image_resized, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 1)
                start_punkt = rect.left(), rect.top()
                end_punkt = rect.right(), rect.bottom()

            cropped_img = np.expand_dims(np.expand_dims(resize_images(image_resized, 512), -1), 0)
            black_background = np.zeros(image_resized.shape[:2], np.uint8)
            cv2.rectangle(black_background, start_punkt, end_punkt, (255, 255, 255), -1)
            frame_masked = cv2.bitwise_and(image_resized, image_resized, mask=black_background)
            #cv2.imshow("just the face", frame_masked)

            # os.chdir(r"test")

            file_name = new_path + '/frame_' + str(image_index) + '.jpg'
            cv2.imwrite(file_name, frame_masked)
            image_index = image_index + 1

            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

if mode.face:
    ''' Train data '''
    print("it starts for train data")
    preprocessing_images_detect_face("archive/train/sad/*", "data/train/sad")
    print("done-train-Sad")
    preprocessing_images_detect_face("archive/train/fear/*", "data/train/fear")
    print("done-train-fear")
    preprocessing_images_detect_face("archive/train/angry/*", "data/train/angry")
    print("done-train-angry")
    preprocessing_images_detect_face("archive/train/happy/*", "data/train/happy")
    print("done-train-happy")
    preprocessing_images_detect_face("archive/train/surprise/*", "data/train/surprise")
    print("done-train-surprise")
    preprocessing_images_detect_face("archive/train/neutral/*", "data/train/neutral")
    print("done-train-neutral")
    preprocessing_images_detect_face("archive/train/disgust/*", "data/train/disgust")
    print("done-train-disgust")

    ''' Test data '''
    print("it starts for test data")
    preprocessing_images_detect_face("archive/test/sad/*", "data/test/sad")
    print("done-test-sad")
    preprocessing_images_detect_face("archive/test/fear/*", "data/test/fear")
    print("done-test-fear")
    preprocessing_images_detect_face("archive/test/angry/*", "data/test/angry")
    print("done-test-angry")
    preprocessing_images_detect_face("archive/test/happy/*", "data/test/happy")
    print("done-test-happy")
    preprocessing_images_detect_face("archive/test/surprise/*", "data/test/surprise")
    print("done-test-surprise")
    preprocessing_images_detect_face("archive/test/neutral/*", "data/test/neutral")
    print("done-test-neutral")
    preprocessing_images_detect_face("archive/test/disgust/*", "data/test/disgust")
    print("done-test-disgust")

if mode.eyeMouth:
    ''' Train data '''
    print("it starts for train data")
    preprocessing_images_detect_eye_mouth("/home/simon/BA/archive/train/sad/*","/home/simon/BA/data/train/sad")
    print("done-train-Sad")
    preprocessing_images_detect_eye_mouth("/home/simon/BA/archive/train/fear/*","/home/simon/BA/data/train/fear")
    print("done-train-fear")
    preprocessing_images_detect_eye_mouth("/home/simon/BA/archive/train/happy/*","/home/simon/BA/data/train/happy")
    print("done-train-happy")
    preprocessing_images_detect_eye_mouth("/home/simon/BA/archive/train/angry/*","/home/simon/BA/data/train/angry")
    print("done-train-angry")
    preprocessing_images_detect_eye_mouth("/home/simon/BA/archive/train/disgust/*","/home/simon/BA/data/train/disgust")
    print("done-train-disgust")
    preprocessing_images_detect_eye_mouth("/home/simon/BA/archive/train/surprise/*","/home/simon/BA/data/train/surprise")
    print("done-train-surprise")
    preprocessing_images_detect_eye_mouth("/home/simon/BA/archive/train/neutral/*","/home/simon/BA/data/train/neutral")
    print("done-train-neutral")

    ''' Test data '''
    print("it starts for test data")
    preprocessing_images_detect_eye_mouth("/home/simon/BA/archive/test/sad/*", "/home/simon/BA/data/test/sad")
    print("done-test-Sad")
    preprocessing_images_detect_eye_mouth("/home/simon/BA/archive/test/fear/*", "/home/simon/BA/data/test/fear")
    print("done-test-fear")
    preprocessing_images_detect_eye_mouth("/home/simon/BA/archive/test/happy/*", "/home/simon/BA/data/test/happy")
    print("done-test-happy")
    preprocessing_images_detect_eye_mouth("/home/simon/BA/archive/test/angry/*", "/home/simon/BA/data/test/angry")
    print("done-test-angry")
    preprocessing_images_detect_eye_mouth("/home/simon/BA/archive/test/disgust/*", "/home/simon/BA/data/test/disgust")
    print("done-test-disgust")
    preprocessing_images_detect_eye_mouth("/home/simon/BA/archive/test/surprise/*", "/home/simon/BA/data/test/surprise")
    print("done-test-surprise")
    preprocessing_images_detect_eye_mouth("/home/simon/BA/archive/test/neutral/*", "/home/simon/BA/data/test/neutral")
    print("done-test-neutral")
