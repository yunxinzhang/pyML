# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 17:38:59 2018

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


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=20)
rfc.fit(xtr_st, ytr)
s = rfc.score(sc.transform(xt), yt)