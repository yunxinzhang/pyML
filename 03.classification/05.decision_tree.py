# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 17:36:34 2018

@author: zyx
"""

import pandas as pd

data = pd.read_csv("Social_Network_Ads.csv")
x = data.iloc[:,[2,3]].values
y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
xtr, xt, ytr, yt = train_test_split(x,y)


# decision tree
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(xtr, ytr)
s = dtc.score(xt, yt)