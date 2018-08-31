# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 18:02:10 2018

@author: zyx
"""

import pandas as pd

data = pd.read_csv("Ads_CTR_Optimisation.csv")
x = data.iloc[:,:].values

import numpy as np


# my ucb algorithm
avgs = np.zeros((1, x.shape[1]))
chooses = np.zeros((1, x.shape[1]))
scores = np.zeros((1, x.shape[1]))
ucbs = np.ones((1, x.shape[1])) * 1e10
res = []
cnt = 0
for i in range(len(x)):
    chs = np.argmax(ucbs)
    res.append(chs)
    chooses[0,chs] += 1
    if x[i, chs] == 1:
        scores[0,chs] += 1
        cnt += 1
    avgs[0,chs] = scores[0,chs]/chooses[0,chs]
    ucbs[0,chs] = avgs[0,chs] + np.sqrt(3*np.log(i+1)/(2*chooses[0,chs]))
    
sus = np.sum(x,0)
