# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 15:17:19 2018

@author: zyx
"""

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Mall_Customers.csv")
x = data.iloc[:,[3,4]].values


    


# show the graph
# =============================================================================
# plt.scatter(x[:,0], x[:,1])
# plt.show()
# =============================================================================
# =============================================================================
# choose n_clusters
# =============================================================================
from sklearn.cluster import KMeans

dist = []
for i in range(1,15):
    km = KMeans(n_clusters=i, init="k-means++")
    km.fit(x)
    dist.append(km.inertia_)
   
plt.plot(range(1,15), dist)  
plt.show()  

# =============================================================================
# 
# =============================================================================
#should init many times , choose the best result
km = KMeans(n_clusters=5, init="k-means++")
y = km.fit_predict(x)  # don't use wrong function

#import numpy as np
#y = np.argmax(yp,1)

plt.scatter(x[y==0,0], x[y==0,1], c='red')
plt.scatter(x[y==1,0], x[y==1,1], c='yellow')
plt.scatter(x[y==2,0], x[y==2,1], c='blue')
plt.scatter(x[y==3,0], x[y==3,1], c='green')
plt.scatter(x[y==4,0], x[y==4,1], c='cyan')

plt.show()

