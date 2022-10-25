from Recognition.face.camera import *


class TestModel:
    emotion_classes = {
        0: "Angry", 1: "Disgusted", 2: "Fear",
        3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
    }
    filename_json = ""
    filename_h5 = ""

    def __init__(self, filename_json, filename_h5):
        self.filename_json = filename_json
        self.filename_h5 = filename_h5
        print("I am testing your state emotion :) always be happy")

    # Calling  open_emotion_model be like
    # emotion_model = open_emotion_model("model_1/model.json", "model_1/model.h5")
    #
    def open_emotion_model(self, filename_json, filename_h5):
        # filename_json can be like : 'model_1/model.json'
        # filename_h5 can be like "model_1/model.h5"

        file = open(filename_json, 'r')
        model_as_json = file.read()
        file.close()
        model = tf.keras.models.model_from_json(model_as_json)
        model.load_weights(filename_h5)

        return model

    def emotion_recognition(self, image_size=48):
        emotion_model = self.open_emotion_model(self.filename_json, self.filename_h5)

        # open camera with face detection
        window = Camera(which_camera=0)
        window.open_camera()
        while window.open_camera():
            window.face_recognition()
            cropped_img = np.expand_dims(np.expand_dims(resize_images(window.gray_image, image_size), -1), 0)
            emotion_prediction = emotion_model.predict(cropped_img)
            max_index = int(np.argmax(emotion_prediction))
            cv.putText(window.frame, self.emotion_classes[max_index], (window.x + 5, window.y - 30),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)

        window.close_camera()
        print(window.frame)
