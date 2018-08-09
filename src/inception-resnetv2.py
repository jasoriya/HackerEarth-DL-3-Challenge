# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 02:00:58 2018

@author: Shreyans
"""

import os
import sys
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

sys.path.append('../src')
sys.path.append('../data/train_img')

import numpy as np
import pandas as pd
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model, load_model
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, History
from keras.optimizers import Adam, SGD
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

def create_model(img_dim=(139, 139, 3)):
    input_tensor = Input(shape=img_dim)
    base_model = InceptionResNetV2(include_top=False,
                       weights='imagenet',
                       input_shape=img_dim)
    
#    for layer in base_model.layers[:8]:
#       layer.trainable = False
    for layer in base_model.layers[:64]:
       layer.trainable = False
       
    bn = BatchNormalization()(input_tensor)
    x = base_model(bn)
    x = Flatten()(x)
#    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(85, activation='sigmoid')(x)
    model = Model(input_tensor, output)
    return model


train_jpeg_dir, test_jpeg_dir, train_csv_file, test_csv_file = get_jpeg_data_files_paths()
labels_df = pd.read_csv(train_csv_file)
filenames = labels_df['Image_name']
labels_df = labels_df.drop("Image_name", axis = 1)
test_labels = pd.read_csv(test_csv_file)

train_labels, val_labels = train_test_split(labels_df, train_size = 0.8, random_state=42)
train_filenames, val_filenames = train_test_split(filenames, train_size = 0.8, random_state=42)

img_resize = (299, 299) # The resize size of each image ex: (64, 64) or None to use the default image size


model = create_model(img_dim=(299, 299, 3))
model.summary()

history = History()

callbacks = [history, 
             EarlyStopping(monitor='val_loss', patience=5, verbose=1, min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, cooldown=0, min_lr=1e-7, verbose=1),
             ModelCheckpoint(filepath='../weights/weights_299_ir.best.hdf5', 
             monitor='val_acc', verbose=1, 
             save_best_only=True, save_weights_only=True, mode='auto')]

batch_size = 6
train_generator = get_train_generator(batch_size, train_filenames, train_labels, train_jpeg_dir)
val_generator = get_validation_generator(batch_size, val_filenames, val_labels, train_jpeg_dir)
steps = len(train_filenames) / batch_size
val_steps = len(val_filenames) / batch_size
#steps = len(train_filenames) / 60
#val_steps = len(val_filenames) / 40

opt = Adam(lr=1e-4)
#opt = SGD(lr=0.001, momentum=0.9, decay=1e-6,nesterov=True)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics = ['accuracy'])

history = model.fit_generator(train_generator, steps, epochs=200, verbose=1, validation_data=val_generator, validation_steps = val_steps, callbacks=callbacks)

#model.load_weights('../weights/weights_256_ir.best.hdf5')

pred_generator = get_prediction_generator(batch_size, test_labels.Image_name, test_jpeg_dir)

predictions_labels = model.predict_generator(generator=pred_generator, verbose=1, steps = len(test_labels) / batch_size)

predictions_labels = np.round(predictions_labels)
pred_df = pd.DataFrame(predictions_labels, columns=list(train_labels.columns))
pred_df.insert(0, "Image_name", test_labels.Image_name)
pred_df.to_csv("predictions_inception_resnet_299_56.csv", index = False)