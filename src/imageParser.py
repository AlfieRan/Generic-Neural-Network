import numpy as np
from mnist import MNIST

mndata = MNIST('../samples')


def process(images, labels):
    allImages = mndata.process_images_to_lists(images)
    Y = np.array(labels).T
    X = np.array(allImages).T / 255
    return X, Y


def get_training():
    images, labels = mndata.load_training()
    return process(images, labels)


def get_testing():
    images, labels = mndata.load_testing()
    return process(images, labels)
