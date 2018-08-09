# -*- coding: utf-8 -*-
"""
Created on Sat May 19 22:13:46 2018

@author: Shreyans
"""

import pandas as pd
import numpy as np

import glob

extension = 'csv'
result = [i for i in glob.glob('*.{}'.format(extension))]

data= []
for item in result:
    df = pd.read_csv(item)
    df.drop('Image_name', inplace = True, axis =1)
    data.append(df.values)
    
wt = [0.1, 0.6, 0.5, 0.7, 0.7, 0.7, 0.7, 0.5, 0.6, 0.6, 1, 0.6, 0.25, 0.6, 0.6, 0.2]
pred = np.round(np.average([data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8],data[9],data[10],data[11],data[12],data[13],data[14],data[15]], axis = 0, weights=wt))

dummy = pd.read_csv(result[0])
image_df = dummy.truncate(after="Image_name", axis = "columns")
pred_df = dummy.truncate(after=-1)
pred_df = pd.DataFrame(pred, columns=list(dummy.columns)[1:])
pred_df.insert(0, "Image_name", image_df)
pred_df.to_csv("ensemble_wt_average.csv", index = False)