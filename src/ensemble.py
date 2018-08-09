# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 11:51:52 2018

@author: Shreyans
"""

import os
import sys

sys.path.append('../src')
sys.path.append('../data/train_img')

import numpy as np
import pandas as pd
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.models import Model, load_model
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, History
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from sklearn.model_selection import train_test_split




def get_jpeg_data_files_paths():
    """
    Returns the input file folders path
    :return: list of strings
        The input file paths as list [train_jpeg_dir, test_jpeg_dir, train_csv_file]
    """

    data_root_folder = os.path.abspath("../data/")
    train_jpeg_dir = os.path.join(data_root_folder, 'train_img/')
    test_jpeg_dir = os.path.join(data_root_folder, 'test_img/')
    train_csv_file = os.path.join(data_root_folder, 'meta-data', 'train.csv')
    test_csv_file = os.path.join(data_root_folder, 'meta-data', 'test.csv')
    return [train_jpeg_dir, test_jpeg_dir, train_csv_file, test_csv_file]

def get_train_generator(batch_size, filenames, labels_df, train_jpeg_dir):
        """
        Returns a batch generator which transforms chunk of raw images into numpy matrices
        and then "yield" them for the classifier. Doing so allow to greatly optimize
        memory usage as the images are processed then deleted by chunks (defined by batch_size)
        instead of preprocessing them all at once and feeding them to the classifier.
        :param batch_size: int
            The batch size
        :param filenames: Series
            The list of train image filenames
        :param labels_df: DataFrame
            Training labels
        :param train_jpeg_dir: str
            Train directory path
        :return: generator
            The batch generator
        """
        # Image Augmentation
        datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            horizontal_flip=True,
            vertical_flip=True)  # randomly flip images horizontally
        loop_range = len(filenames)
        while True:
            for i in range(loop_range):
                start_offset = batch_size * i

                # The last remaining files could be smaller than the batch_size
                range_offset = min(batch_size, loop_range - start_offset)

                # If we reached the end of the list then we break the loop
                if range_offset <= 0:
                    break

                batch_features = np.zeros((range_offset, *img_resize, 3))
                batch_labels = np.zeros((range_offset, len(labels_df.columns)))

                for j in range(range_offset):   
                    img_path = train_jpeg_dir + filenames.iloc[start_offset + j]
                    img = image.load_img(img_path, target_size=img_resize)
                    img = image.img_to_array(img)

                    img_array = img[:, :, ::-1]
                    # Zero-center by mean pixel
                    img_array[:, :, 0] -= 103.939
                    img_array[:, :, 1] -= 116.779
                    img_array[:, :, 2] -= 123.68

                    batch_features[j] = img_array
                    batch_labels[j] = labels_df.iloc[start_offset + j]

                # Augment the images (using Keras allow us to add randomization/shuffle to augmented images)
                # Here the next batch of the data generator (and only one for this iteration)
                # is taken and returned in the yield statement
                yield next(datagen.flow(batch_features, batch_labels, range_offset))
                
def get_validation_generator(batch_size, filenames, labels_df, train_jpeg_dir):
    """
        Returns a batch generator which transforms chunk of raw images into numpy matrices
        and then "yield" them for the classifier. Doing so allow to greatly optimize
        memory usage as the images are processed then deleted by chunks (defined by batch_size)
        instead of preprocessing them all at once and feeding them to the classifier.
        :param batch_size: int
            The batch size
        :param filenames: Series
            The list of validation image filenames
        :param labels_df: DataFrame
            Validation labels
        :param train_jpeg_dir: str
            Validation directory path
        :return: generator
            The batch generator
        """
        # Image Augmentation
    datagen = ImageDataGenerator(rescale=1./255)
    loop_range = len(filenames)
    while True:
            for i in range(loop_range):
                start_offset = batch_size * i

                # The last remaining files could be smaller than the batch_size
                range_offset = min(batch_size, loop_range - start_offset)

                # If we reached the end of the list then we break the loop
                if range_offset <= 0:
                    break

                batch_features = np.zeros((range_offset, *img_resize, 3))
                batch_labels = np.zeros((range_offset, len(labels_df.columns)))

                for j in range(range_offset):   
                    img_path = train_jpeg_dir + filenames.iloc[start_offset + j]
                    img = image.load_img(img_path, target_size=img_resize)
                    img = image.img_to_array(img)

                    batch_features[j] = img
                    batch_labels[j] = labels_df.iloc[start_offset + j]

                # Here the next batch of the data generator (and only one for this iteration)
                # is taken and returned in the yield statement
                yield next(datagen.flow(batch_features, batch_labels, range_offset))
                
def get_prediction_generator(batch_size, test_filename, test_jpeg_dir):
        """
        Returns a batch generator which transforms chunk of raw images into numpy matrices
        and then "yield" them for the classifier. Doing so allow to greatly optimize
        memory usage as the images are processed then deleted by chunks (defined by batch_size)
        instead of preprocessing them all at once and feeding them to the classifier.
        :param batch_size: int
            The batch size
        :param test_filename: Series
            The list of test image filenames
        :param test_jpeg_dir: str
            Test directory path
        :return: generator
            The batch generator
        """

        # NO SHUFFLE HERE as we need our predictions to be in the same order as the inputs
        loop_range = len(test_filename)
        while True:
            for i in range(loop_range):
                start_offset = batch_size * i

                # The last remaining files could be smaller than the batch_size
                range_offset = min(batch_size, loop_range - start_offset)

                # If we reached the end of the list then we break the loop
                if range_offset <= 0:
                    break

                img_arrays = np.zeros((range_offset, *img_resize, 3))

                for j in range(range_offset):
                    img_path = test_jpeg_dir + test_filename.iloc[start_offset + j]
                    img = image.load_img(img_path, target_size=img_resize)
                    img = image.img_to_array(img)


                    img_array = img[:, :, ::-1]
                    # Zero-center by mean pixel
                    img_array[:, :, 0] -= 103.939
                    img_array[:, :, 1] -= 116.779
                    img_array[:, :, 2] -= 123.68
                    img_array = img_array / 255

                    img_arrays[j] = img_array
                yield img_arrays

def create_inception_resnet(img_dim=(139, 139, 3)):
    input_tensor = Input(shape=img_dim)
    base_model = InceptionResNetV2(include_top=False,
                       weights='imagenet',
                       input_shape=img_dim)
    
    for layer in base_model.layers[:8]:
       layer.trainable = False
       
    bn = BatchNormalization()(input_tensor)
    x = base_model(bn)
    x = Flatten()(x)
    output = Dense(85, activation='sigmoid')(x)
    model = Model(input_tensor, output)
    return model

def create_inceptionV3(img_dim=(139, 139, 3)):
    input_tensor = Input(shape=img_dim)
    base_model = InceptionV3(include_top=False,
                       weights='imagenet',
                       input_shape=img_dim)
    
    for layer in base_model.layers[:8]:
       layer.trainable = False
       
    bn = BatchNormalization()(input_tensor)
    x = base_model(bn)
    x = Flatten()(x)
    output = Dense(85, activation='sigmoid')(x)
    model = Model(input_tensor, output)
    return model

def create_vgg16(img_dim=(128, 128, 3)):
    input_tensor = Input(shape=img_dim)
    base_model = VGG16(include_top=False,
                       weights='imagenet',
                       input_shape=img_dim)
    
    for layer in base_model.layers[:8]:
       layer.trainable = False
       
    bn = BatchNormalization()(input_tensor)
    x = base_model(bn)
    x = Flatten()(x)
    output = Dense(85, activation='sigmoid')(x)
    model = Model(input_tensor, output)
    return model