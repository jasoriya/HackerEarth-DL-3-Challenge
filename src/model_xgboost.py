# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 03:13:51 2018

@author: Shreyans
"""

from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import pandas as pd
import pickle

train_features = np.load("../data/train_features.npy")
test_features = np.load("../data/test_features.npy")

train_data = pd.read_csv("../data/meta-data/train.csv")
test_data = pd.read_csv("../data/meta-data/test.csv")

train_labels = train_data.drop("Image_name", axis = 1)

model = OneVsRestClassifier(XGBClassifier(), n_jobs = -1)
model.fit(train_features, train_labels)

file = open("xgboostModel",'rb')
model = pickle.load(file)
file.close()

pred_labels = model.predict(test_features)

pred_df = pd.DataFrame(pred_labels, columns=list(train_labels.columns))
pred_df.insert(0, "Image_name", test_data)
pred_df.to_csv("predictions_xgboost.csv", index = False)