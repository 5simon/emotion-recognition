from Recognition.Emotion.model import *

if __name__ == "__main__":
    print("tensorflow is installed and has the version: ", tf.__version__)
    print("\nopencv is installed and has the version: ", cv.__version__)

    # camera_window = Camera()
    # camera_window.save_image_from_camera()

    training_data()
