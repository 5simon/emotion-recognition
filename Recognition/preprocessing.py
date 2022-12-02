import os
from time import sleep

import cv2
import dlib
import numpy as np
import glob
from Recognition.Emotion.helpFunctions import resize_images

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

def draw_shape_lines_range(np_shape, image, range_points, is_closed=False):
    """Draws the shape using lines to connect the different points"""

    np_shape_display = np_shape[range_points]
    points = np.array(np_shape_display, dtype=np.int32)
    cv2.polylines(image, [points], is_closed, (255, 255, 0), thickness=1, lineType=cv2.LINE_8)


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
image_index = 0
path = glob.glob("archive/train/sad/*")
for image in path:
    img = cv2.imread(image)
    image_resized = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    start_punkt, end_punkt = [0, 0], [0, 0]

    for (i, rect) in enumerate(rects):
        cv2.rectangle(image_resized, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 1)
        start_punkt = rect.left(), rect.top()
        end_punkt = rect.right(), rect.bottom()

    cropped_img = np.expand_dims(np.expand_dims(resize_images(image_resized, 512), -1), 0)
    black_background = np.zeros(image_resized.shape[:2], np.uint8)
    cv2.rectangle(black_background, start_punkt, end_punkt, (255, 255, 255), -1)
    frame_masked = cv2.bitwise_and(image_resized, image_resized, mask=black_background)
    cv2.imshow("just the face", frame_masked)


    # os.chdir(r"test")

    file_name = 'test/frame_' + str(image_index) + '.jpg'
    cv2.imwrite(file_name, frame_masked)
    image_index = image_index + 1

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# image_test = cv2.imread("../Training_33331.jpg")
# image_test_1 = cv2.resize(image_test, (512, 512))
# gray_test = cv2.cvtColor(image_test_1, cv2.COLOR_BGR2GRAY)
# rects_test = detector(gray_test, 0)
#
# print(rects_test)
#
# start_punkt, end_punkt = [0, 0], [0, 0]
#
# for (i, rect) in enumerate(rects_test):
#     cv2.rectangle(image_test, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 1)
#     start_punkt = rect.left(), rect.top()
#     end_punkt = rect.right(), rect.bottom()
# cropped_img = np.expand_dims(np.expand_dims(resize_images(image_test_1, 512), -1), 0)
# black_background = np.zeros(image_test_1.shape[:2], np.uint8)
# cv2.rectangle(black_background, start_punkt, end_punkt, (255, 255, 255), -1)
# frame_masked = cv2.bitwise_and(image_test_1, image_test_1, mask=black_background)
# cv2.imshow("just the face", frame_masked)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# while True:
#
#     # Capture frame from the VideoCapture object:
#     ret, frame = video_capture.read()
#
#     # Just for debugging purposes:
#     # frame = test_face.copy()
#
#     # Convert frame to grayscale:
#     # frame = cv2.resize(frame,(0,0),fx = 0.5 , fy = 0.5)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Detect faces:
#     rects = detector(gray, 0)
#
#     # For each detected face, find the landmark.
#     for (i, rect) in enumerate(rects):
#         # Draw a box around the face:
#         cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 1)
#
#         # Get the shape using the predictor:
#         shape = predictor(gray, rect)
#
#         # Convert the shape to numpy array:
#         shape = shape_to_np(shape)
#
#         # Draw all lines connecting the different face parts:
#         # draw_shape_lines_all(shape, frame)
#
#         # Draw jaw line:
#         draw_shape_lines_range(shape, frame, JAWLINE_POINTS)
#
#         # Draw all points and their position:
#         draw_shape_points_pos(shape, frame)
#         # You can also use:
#         # draw_shape_points_pos_range(shape, frame, ALL_POINTS)
#
#         # Draw all shape points:
#         draw_shape_points(shape, frame)
#
#         # Draw left eye, right eye and bridge shape points and positions
#         # draw_shape_points_pos_range(shape, frame, LEFT_EYE_POINTS + RIGHT_EYE_POINTS + NOSE_BRIDGE_POINTS)
#
#     # Display the resulting frame
#     # out.write(frame)
#     cv2.imshow("Landmarks detection using dlib", frame)
#
#     # Press 'q' key to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release everything:
# video_capture.release()
# cv2.destroyAllWindows()