# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 17:29:17 2018

@author: zyx
"""

import pandas as pd

data = pd.read_csv("Social_Network_Ads.csv")
x = data.iloc[:,[2,3]].values
y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
xtr, xt, ytr, yt = train_test_split(x,y)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xtr_st = sc.fit_transform(xtr)

# naive bayes
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(xtr_st, ytr)
yp = gnb.predict(sc.transform(xt))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true=yt, y_pred=yp)

