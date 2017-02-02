import csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import random


class DataSource:
    def __init__(self, directory, csv_filename):
        self.__directory = directory
        self.__csv_filename = csv_filename

        self.__left = TrainingBag()
        self.__right = TrainingBag()
        self.__center = TrainingBag()

        self.validation_images = None
        self.validation_angles = None

        self.__readAndShuffleData()

    def __readAndShuffleData(self):
        path = '{}/{}'.format(self.__directory, self.__csv_filename)

        image_names = []
        angles = []

        with open(path, 'r') as csvfile:
            next(csvfile, None)
            for center, left, right, steering, _, _, _ in csv.reader(csvfile):
                steering = float(steering)

                image_names.append(center)
                angles.append(steering)
                image_names.append(left)
                angles.append(steering + .25)
                image_names.append(right)
                angles.append(steering - .25)

        print("Shuffling array... ")
        shuffle(image_names, angles)
        print("Done")

        train_images, self.validation_images, \
            train_angles, self.validation_angles \
            = train_test_split(image_names, angles, test_size=0.2,
                               random_state=0)

        for image_name, angle in zip(train_images, train_angles):
            if angle < -.15:  # left turn
                self.__left.add(image_name, angle)
            elif angle > .15:  # right turn
                self.__right.add(image_name, angle)
            else:  # center
                self.__center.add(image_name, angle)

    def next(self):
        # pick a random number to choose which bag we should read from
        # this keeps the data equally weighted between left, right, and center
        bagNum = random.randrange(0, 3)
        if bagNum is 0:
            img, angle = self.__left.nextItem()
            return img, angle
        elif bagNum is 1:
            img, angle = self.__right.nextItem()
            return img, angle
        else:
            img, angle = self.__center.nextItem()
            return img, angle

    def trainDataSize(self):
        print("Left: {}".format(self.__left.size()))
        print("Right: {}".format(self.__right.size()))
        print("Center: {}".format(self.__center.size()))
        return self.__left.size() + self.__center.size() + self.__right.size()


class TrainingBag:

    def __init__(self):
        self.__data = []
        self.__ctr = 0

    def nextItem(self):
        # print("ctr: {}".format(self.__ctr))
        image_name, angle = self.__data[self.__ctr]
        self.__ctr += 1
        if self.__ctr == len(self.__data):
            self.__ctr = 0
        return image_name, angle

    def add(self, image_name, angle):
        self.__data.append((image_name, angle))

    def size(self):
        return len(self.__data)

    def getData(self):
        return self.__data[0]
