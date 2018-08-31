# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 11:36:27 2018

@author: zyx
"""

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Social_Network_Ads.csv")
x = data.iloc[:,[2,3]].values
y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
xtr, xt, ytr, yt = train_test_split(x,y)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
xtr_st = sc.fit_transform(xtr)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(xtr_st, ytr)
yp_prob = lr.predict_proba(sc.transform(xt))
yp = lr.predict(sc.transform(xt))

x1 = xt[yp==0]
x2 = xt[yp==1]
x3 = x[y==0]
x4 = x[y==1]
plt.subplot(1,2,1)
plt.scatter(x1[:,0], x1[:,1])
plt.scatter(x2[:,0], x2[:,1])
plt.subplot(1,2,2)
plt.scatter(x3[:,0], x3[:,1])
plt.scatter(x4[:,0], x4[:,1])
plt.show()

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true=yt, y_pred=yp)

