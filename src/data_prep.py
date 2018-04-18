# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 22:12:06 2018

@author: Shreyans
"""

import pandas as pd
import numpy as np
from keras import applications
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
import os

train_data = pd.read_csv("../data/meta-data/train.csv")
test_data = pd.read_csv("../data/meta-data/test.csv")

def named_model(name):
    # include_top=False removes the fully connected layer at the end/top of the network
    # This allows us to get the feature vector as opposed to a classification
    if name == 'Xception':
        return applications.xception.Xception(weights='imagenet', include_top=False, pooling='avg')

    if name == 'VGG16':
        return applications.vgg16.VGG16(weights='imagenet', include_top=False, pooling='avg')

    if name == 'VGG19':
        return applications.vgg19.VGG19(weights='imagenet', include_top=False, pooling='avg')

    if name == 'InceptionV3':
        return applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')

    if name == 'MobileNet':
        return applications.mobilenet.MobileNet(weights='imagenet', include_top=False, pooling='avg')

    return applications.resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')

def image_vectors(data, model):
    source_dir = os.path.dirname(os.getcwd())
    arr = []
    for i in range(len(data)):
        print("Extracting feature data of Image-" + str(i) + ".jpg")
        img_path = os.path.join(source_dir, 'data/train_img/' + data.iloc[i])
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        x = preprocess_input(x)
        
        features = model.predict(x)[0]
        features_arr = np.char.mod('%f', features)
        arr.append(features_arr)
    return arr

model = named_model("InceptionV3")
features_data = image_vectors(train_data.Image_name, model)
#np.save("../data/train_features",features_data)
feature_test = image_vectors(test_data.Image_name, model)
np.save("../data/test_features.npy", feature_test)



