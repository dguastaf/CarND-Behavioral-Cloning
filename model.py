import cv2
import gc
import os
import json
import random
import numpy as np
from itertools import cycle
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Dropout
from keras.optimizers import Adam

from data_source import DataSource

# File paths names
DATA_DIR = 'data'
IMAGE_DIR = DATA_DIR + "/IMG/"
DATA_FILENAME = '/driving_log.csv'

# Image cropping
X1 = 20
X2 = 300
Y1 = 80
Y2 = 140

HEIGHT = Y2 - Y1
WIDTH = X2 - X1

# Create object to retrieve data
data_source = DataSource(DATA_DIR, DATA_FILENAME)


def import_image(filename):
    _, filename = os.path.split(filename)
    return cv2.imread(IMAGE_DIR + filename)


def crop_image(data):
    return data[Y1:Y2, X1:X2]


def flip(data, angle):
    data = cv2.flip(data, 1)
    angle = -angle
    return data, angle


# Generator for getting training data
def train_gen(batch_size=100):
    while True:
        image_batch = np.zeros((batch_size, HEIGHT, WIDTH, 3))
        angle_batch = np.zeros(batch_size)
        for i in range(batch_size):
            image_name, angle = data_source.next()

            data = import_image(image_name)
            data = crop_image(data)

            rand = random.randrange(0, 2)
            if rand == 0:
                data, angle = flip(data, angle)

            image_batch[i] = data
            angle_batch[i] = angle
        yield image_batch, angle_batch


def validation_gen(images, angles, batch_size=50):
    while True:
        image_batch = np.zeros((batch_size, HEIGHT, WIDTH, 3))
        angle_batch = np.zeros(batch_size)
        i = 0

        for image, angle in cycle(zip(images, angles)):
            _, image_name = os.path.split(image)

            data = import_image(image_name)
            data = crop_image(data)

            image_batch[i] = data
            angle_batch[i] = angle

            i += 1

            if i == batch_size:
                yield image_batch, angle_batch
                image_batch = np.zeros((batch_size, HEIGHT, WIDTH, 3))
                angle_batch = np.zeros(batch_size)
                i = 0


def get_model():
    model = Sequential()
    model.add(Convolution2D(24, 5, 5,
              input_shape=(HEIGHT, WIDTH, 3),
              subsample=(2, 2),
              border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5,
              subsample=(2, 2),
              border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5,
              subsample=(2, 2),
              border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(Dense(1164))
    model.add(Activation('relu'))
    model.add(Dropout(.5))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss='mse')

    return model


print("Training size: {}".format(data_source.trainDataSize()))
print("Validation size: {}".format(len(data_source.validation_images)))

model = get_model()

model.fit_generator(
    train_gen(),
    samples_per_epoch=10000,
    nb_epoch=10,
    validation_data=validation_gen(data_source.validation_images,
                                   data_source.validation_angles),
    nb_val_samples=200)

print("Saving weights & config file")

model.save_weights("steering_model.h5", True)
with open('steering_model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)

print("Saved")

# TF bug - https://github.com/tensorflow/tensorflow/issues/3388
gc.collect()
