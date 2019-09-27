from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib
import argparse

import tensorflow as tf
import numpy as np

from tensorflow.keras import datasets, preprocessing, models
from model import create_model

image_height = 32
image_width = 32
image_channels = 3

default_model_path = 'models/model.h5'
default_log_path = 'logs/log.csv'


def load_train_and_eval_data():
    (train_images, train_labels), (eval_images, eval_labels) = datasets.cifar10.load_data()
    train_images, eval_images = train_images.astype(np.float32) / np.float32(255), eval_images.astype(np.float32) / np.float32(255)
    train_data_size, eval_data_size = len(train_images), len(eval_images)
    return (train_images, train_labels, train_data_size), (eval_images, eval_labels, eval_data_size)

def load_eval_data():
    images, labels = datasets.cifar10.load_data()[1]
    images.astype(np.float32) / np.float32(255)
    labels = [label[0] for label in labels.tolist()]
    data_size = len(images)
    return images, labels, data_size

def load_test_data():
    images_path = pathlib.Path('test-images')

    image_paths = list(images_path.glob('*/*'))
    image_paths = [str(path) for path in image_paths]

    label_names = sorted(item.name for item in images_path.glob('*/') if item.is_dir())
    label_to_index = dict((name, index) for index,name in enumerate(label_names))

    def load_and_preprocess_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=image_channels)
        image_shape = image.get_shape().as_list()
        image = tf.image.resize_with_crop_or_pad(image, image_shape[0], image_shape[0])
        image = tf.image.resize(image, [image_height, image_width])
        return image

    images = np.array([load_and_preprocess_image(path) for path in image_paths])
    images = images.astype(np.float32) / np.float32(255)
    labels = [label_to_index[pathlib.Path(path).parent.name] for path in image_paths]
    size = len(images)

    return images, labels, size


def create_data_gen(images):
    data_gen = preprocessing.image.ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
    data_gen.fit(images)
    return data_gen

def create_train_data_gen(images):
    data_gen = preprocessing.image.ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    data_gen.fit(images)
    return data_gen

def load_model(model_path, summary=True):
    try:
        model = models.load_model(model_path)
    except:
        model = create_model((image_height, image_width, image_channels))
        model.save(model_path)
    if summary: model.summary()
    return model
