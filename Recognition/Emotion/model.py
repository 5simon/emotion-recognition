import tensorflow as tf
# from Recognition.face.camera import *

'''
    Conv -> BN -> Activation -> Conv -> BN -> Activation -> MaxPooling
    Conv -> BN -> Activation -> Conv -> BN -> Activation -> MaxPooling
    Conv -> BN -> Activation -> Conv -> BN -> Activation -> MaxPooling
    Flatten
    Dense -> BN -> Activation
    Dense -> BN -> Activation
    Dense -> BN -> Activation
    Output layer
'''
index_train_images = 28709
index_validation_images = 7178
epoches = 50
batch_size = 64
image_size = 48
train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
validation_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

train_generator = train_data_generator.flow_from_directory(
    "../archive/train",
    target_size=(image_size, image_size),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical"
)

validation_generator = validation_data_generator.flow_from_directory(
    "../archive/test",
    target_size=(image_size, image_size),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical"
)

# create Model
model = tf.keras.Sequential()

# relu = f(x) = max(0,x)
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(image_size, image_size, 1)))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

model_info = model.fit_generator(
    train_generator,
    steps_per_epoch=index_train_images // batch_size,
    epochs=epoches,
    validation_data=validation_generator,
    validation_steps=index_validation_images // batch_size
)

model.summary()

#save infos

# open path
# Camera.open_path(r"Model infos")
#
# save_model = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(save_model)
#
# model.save_weights("model.h5")

# save model structure in jason file
save_model = model.to_json()
with open("model_1/model.json", "w") as json_file:
    json_file.write(save_model)

# save trained model weight in .h5 file
model.save_weights("model.h5")