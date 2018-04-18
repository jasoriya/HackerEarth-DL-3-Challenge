# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 02:01:58 2018

@author: Shreyans
"""

import keras
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model, Sequential
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, History

batch_size = 32  # orig paper trained all networks with batch_size=128
epochs = 200
num_classes = 85
lr = 1e-4

train_features = np.load("../data/train_features.npy")
test_features = np.load("../data/test_features.npy")

train_data = pd.read_csv("../data/meta-data/train.csv")
test_data = pd.read_csv("../data/meta-data/test.csv")

train_labels = train_data.drop("Image_name", axis = 1)

input_shape = train_features.shape[1:]


model = Sequential()
model.add(Dense(4096, activation='relu', input_shape = input_shape))
#model.add(Dropout(0.25))
model.add(Dense(8192, activation='relu'))
#model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
#model.add(Dropout(0.25))
model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='sigmoid'))

opt = Adam(lr=lr, decay=1e-6)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

history = History()
callbacks = [history, 
             EarlyStopping(monitor='val_loss', patience=3, verbose=1, min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, cooldown=0, min_lr=1e-7, verbose=1),
             ModelCheckpoint(filepath='../weights/weights.best.hdf5', verbose=1,       
             save_best_only=True, save_weights_only=True, mode='auto')]
             
model.fit(train_features, train_labels.values, validation_split=0.2, epochs=100, batch_size=32, shuffle=True, callbacks=callbacks)

pred_labels = model.predict(test_features, batch_size=32, verbose=1)
pred_labels = np.round(pred_labels)
pred_df = pd.DataFrame(pred_labels, columns=list(train_labels.columns))
pred_df.insert(0, "Image_name", train_data.Image_name)
pred_df.to_csv("predictions_nn.csv", index = False)
