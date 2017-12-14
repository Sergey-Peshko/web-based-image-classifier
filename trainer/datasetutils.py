import glob
import os

import cv2
import numpy as np
from sklearn.utils import shuffle

from DataSet import DataSet


def read_classes(train_path):
    lol = os.path.join(train_path, "*")
    files = glob.glob(lol)

    classes = []

    for fl in files:
        classes.append(os.path.basename(fl))

    return classes


def load_train(train_path, image_size, classes):
    images = []
    labels = []

    print('Going to read training images')
    for field in classes:
        index = classes.index(field)
        print('Now going to read {} files (Index: {})'.format(field, index))
        path = os.path.join(train_path, field, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
    images = np.array(images)
    labels = np.array(labels)

    return images, labels


def read_train_sets(train_path, image_size, classes, validation_size):
    class DataSets(object):
        pass

    data_sets = DataSets()

    images, labels = load_train(train_path, image_size, classes)
    images, labels = shuffle(images, labels)

    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])

    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]

    train_images = images[validation_size:]
    train_labels = labels[validation_size:]

    data_sets.train = DataSet(train_images, train_labels)
    data_sets.valid = DataSet(validation_images, validation_labels)

    return data_sets
