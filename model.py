from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
import cv2
import numpy as np
from data_source import DataSource

DATA_DIR = 'data'
DATA_FILENAME = '/driving_log.csv'

data_source = DataSource(DATA_DIR, DATA_FILENAME)


def gen(batch_size=30):
    images = np.zeros((batch_size, 160, 320, 3))
    steering_angles = np.zeros(batch_size)
    i = 0

    while True:
        row = data_source.nextRow()
        center, _, _, steering, _, _, _ = row
        centerImage = cv2.imread(DATA_DIR + "/" + center)
        images[i] = centerImage
        steering_angles[i] = steering
        i += 1
        if i is batch_size:
            yield images, steering_angles
            i = 0
            images = np.zeros((batch_size, 160, 320, 3))
            steering_angles = np.zeros(batch_size)
    yield images, steering_angles


def get_model():
    model = Sequential()
    model.add(Flatten(input_shape=(160, 320, 3)))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.compile('sgd', 'mse')
    return model


model = get_model()

model.fit_generator(
    gen(),
    samples_per_epoch=5000,
    nb_epoch=10)
