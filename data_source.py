import csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random


# Class the reads & stores the training/validation data for
# training our neural network
class DataSource:

    def __init__(self):
        # Left turns
        self.__left = TrainingBag()

        # Right turns
        self.__right = TrainingBag()

        # Straight driving
        self.__straight = TrainingBag()

        self.validation_images = None
        self.validation_angles = None

    def read_data(self, directory, csv_filename):
        image_names = []
        angles = []

        csv_filepath = '{}/{}'.format(directory, csv_filename)

        with open(csv_filepath, 'r') as csvfile:
            # skip the first row of the csv file since it's the column names
            next(csvfile, None)

            # Each row: center, left, right, steering, throttle, brake, speed
            # (center, left, & right are filenames for images from cameras)
            for center, left, right, steering, _, _, _ in csv.reader(csvfile):
                steering = float(steering)

                # Images are sorted into different lists
                # We also adjust the steering angle for left & right
                # images to train the car to recover if it deviates
                # too far to either side of the track
                image_names.append(center)
                angles.append(steering)

                image_names.append(left)
                angles.append(steering + .25)

                image_names.append(right)
                angles.append(steering - .25)

        # Shuffle data before we split into training & validation sets
        shuffle(image_names, angles)

        train_images, self.validation_images, \
            train_angles, self.validation_angles \
            = train_test_split(image_names, angles, test_size=0.2,
                               random_state=0)

        # Sort the data into different bags based on the turning angle.
        for image_name, angle in zip(train_images, train_angles):
            if angle < -.15:  # left turn
                self.__left.add(image_name, angle)
            elif angle > .15:  # right turn
                self.__right.add(image_name, angle)
            else:  # center
                self.__straight.add(image_name, angle)

    # Returns an image & angle pair.__straight
    # To ensure the data is equally weighted between left turns, right turns,
    # and straight driving, we will pick from a random bag
    def next(self):
        bagNum = random.randrange(0, 3)
        if bagNum is 0:
            img, angle = self.__left.nextItem()
            return img, angle
        elif bagNum is 1:
            img, angle = self.__right.nextItem()
            return img, angle
        else:
            img, angle = self.__straight.nextItem()
            return img, angle

    # print info about our training data
    def print_stats(self):
        print("Left: {}".format(self.__left.size()))
        print("Right: {}".format(self.__right.size()))
        print("Center: {}".format(self.__straight.size()))

        training_size = self.__left.size() \
            + self.__straight.size() \
            + self.__right.size()

        print("Training size: {}".format(training_size))
        print("Validation size: {}".format(len(self.validation_images)))


# Data structure to hold training data
class TrainingBag:

    def __init__(self):
        self.__data = []
        self.__ctr = 0

    # gets the next item in the bag
    def nextItem(self):
        image_name, angle = self.__data[self.__ctr]
        self.__ctr += 1
        if self.__ctr == len(self.__data):
            self.__ctr = 0
        return image_name, angle

    def add(self, image_name, angle):
        self.__data.append((image_name, angle))

    def size(self):
        return len(self.__data)
