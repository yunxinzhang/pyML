# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 17:52:29 2018

@author: zyx
"""

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Ads_CTR_Optimisation.csv")
x = data.iloc[:,:].values

import numpy as np

chooses = []
score = 0
for i in range(len(x)):
    choose = np.random.randint(10)
    chooses.append(choose)
    if x[i,choose] == 1 :
        score += 1

plt.hist(chooses)
plt.show()