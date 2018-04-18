# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 00:56:03 2018

@author: Shreyans
"""

from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

train_features = np.load("../data/train_features.npy")
test_features = np.load("../data/test_features.npy")

train_data = pd.read_csv("../data/meta-data/train.csv")
test_data = pd.read_csv("../data/meta-data/test.csv")

train_labels = train_data.drop("Image_name", axis = 1)

clf = DecisionTreeClassifier(random_state=123456)
clf.fit(train_features, train_labels.attrib_01)
pred_labels = clf.predict(test_features)

pred_df = pd.DataFrame(pred_labels, columns=list(train_labels.columns))
pred_df.insert(0, "Image_name", train_data.Image_name)
pred_df.to_csv("predictions.csv", index = False)