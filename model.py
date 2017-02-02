import cv2
import gc
import os
import json
import random
import numpy as np
from itertools import cycle
from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution2D, Dropout
from keras.optimizers import Adam

from data_source import DataSource

# File paths
DATA_DIR = 'data'
IMAGE_DIR = "IMG"
DATA_FILENAME = 'driving_log.csv'
OUT_FILENAME = 'steering_model'

# Image cropping
X1 = 20
X2 = 300
Y1 = 80
Y2 = 140

HEIGHT = Y2 - Y1
WIDTH = X2 - X1

# Training hyperparameters
LEARNING_RATE = 0.0001
NB_EPOCH = 10
NB_TRAIN_SAMPLES = 10000
NB_VAL_SAMPLES = 1000


#### Working with images ####

# Imports an image from the given filename.
# This filename can either be a single image or a full path.
# The function will parse out the filename and read it from the
# directory where the images are stored
def import_image(filename):
    _, filename = os.path.split(filename)
    return cv2.imread("{}/{}/{}".format(DATA_DIR, IMAGE_DIR, filename))


# Crops an image to eliminate any noise in training.
# Mainly we want to just look at the road and crop everything
# else out.
def crop_image(data):
    return data[Y1:Y2, X1:X2]


# To generate more data points without needing to gather
# more data, we flip random images horizontally. It also filps
# the steering angle.
def flip(data, angle):
    data = cv2.flip(data, 1)
    angle = -angle
    return data, angle


#### Training ####

# Generator for getting training data.
# 1. Initialize 2 numpy arrays that will hold images (input) & angles (output)
# 2. Read in an image and crop it
# 3. Flip a coin to determine if we should flip the image. This will help
#    generate additional data points witout needing to gather more data
# 4. Add the image & angle to the set
# 5. Repeat 2-4 until you've read in batch_size images.
# 6. Yield the two lists
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


# Generate a set of validation images. Each image is read in and cropped
# to the same dimensions that are used in training.
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


def save_model(model):
    model.save_weights("{}.h5".format(OUT_FILENAME), True)
    with open('{}.json'.format(OUT_FILENAME), 'w') as outfile:
        json.dump(model.to_json(), outfile)


# Build the convolutional neural network model. This model mirrors most of
# the NVIDIA model detailed in this blog:
# https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
#
# Architecture layers:
# - 5 2D convolutional
# - 4 fully connected
# - 2 dropout regularization
# - relu activation between each layer
#
# Optimizer: Adam with a learning rate of 0.0001
# Loss function: mean squared error
def get_model():

    activation = 'relu'

    # Initialize the neural network
    model = Sequential()

    # Convolutional layer 1
    model.add(Convolution2D(24, 5, 5,
              input_shape=(HEIGHT, WIDTH, 3),
              subsample=(2, 2),
              border_mode='same',
              activation=activation))

    # Convolutional layer 2
    model.add(Convolution2D(36, 5, 5,
              subsample=(2, 2),
              border_mode='same',
              activation=activation))

    # Convolutional layer 3
    model.add(Convolution2D(48, 5, 5,
              subsample=(2, 2),
              border_mode='same',
              activation=activation))

    # Convolutional layer 4
    model.add(Convolution2D(64, 3, 3,
                            border_mode='same',
                            activation=activation))

    # Convolutional layer 5
    model.add(Convolution2D(64, 3, 3,
                            border_mode='same',
                            activation=activation))

    # Flatten convolutional layers into single layer to
    # feed into fully connected layers
    model.add(Flatten())

    # Drop 20% of input units
    model.add(Dropout(.2))

    # Fully connected layer 1
    model.add(Dense(1164, activation=activation))

    # Drop 50% of input units
    model.add(Dropout(.5))

    # Fully connected layer 2
    model.add(Dense(100, activation=activation))

    # Fully connected layer 3
    model.add(Dense(50, activation=activation))

    # Fully connected layer 4
    model.add(Dense(10, activation=activation))

    # Fully connected layer 5 - one output
    model.add(Dense(1))

    # Build model with adam optimizer and mean squared error loss function
    model.compile(optimizer=Adam(lr=LEARNING_RATE), loss='mse')

    return model


# Create object to retrieve data
data_source = DataSource()
data_source.read_data(DATA_DIR, DATA_FILENAME)

# Print some stats about our training & validation data
data_source.print_stats()

model = get_model()

# Train model with hyperparams
model.fit_generator(
    train_gen(),
    samples_per_epoch=NB_TRAIN_SAMPLES,
    nb_epoch=NB_EPOCH,
    validation_data=validation_gen(data_source.validation_images,
                                   data_source.validation_angles),
    nb_val_samples=NB_VAL_SAMPLES)

# Save model weights and structure
print("Saving weights & config file")
save_model(model)
print("Saved")

# TF bug - https://github.com/tensorflow/tensorflow/issues/3388
gc.collect()
