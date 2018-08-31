# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 15:03:55 2018

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

from sklearn.svm import SVC

#svc = SVC(kernel='linear')
svc = SVC(kernel='rbf')
svc.fit(xtr_st, ytr)
yp = svc.predict(sc.transform(xt))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true=yt, y_pred=yp)
