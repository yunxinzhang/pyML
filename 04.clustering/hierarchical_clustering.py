# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 16:19:44 2018

@author: zyx
"""

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Mall_Customers.csv")
x = data.iloc[:,[3,4]].values

from scipy.cluster import hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(x, method="ward"))
plt.show()

from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y = hc.fit_predict(x)

plt.scatter(x[y==0,0], x[y==0,1], c='red')
plt.scatter(x[y==1,0], x[y==1,1], c='yellow')
plt.scatter(x[y==2,0], x[y==2,1], c='blue')
plt.scatter(x[y==3,0], x[y==3,1], c='green')
plt.scatter(x[y==4,0], x[y==4,1], c='cyan')