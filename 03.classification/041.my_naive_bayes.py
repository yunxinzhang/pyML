# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 17:38:02 2018

@author: zyx
"""
import numpy as np

def calc_ent(val, cls):
    ind = np.argsort(val)
    cls_sort = cls[ind]
    ncls1_all = sum(cls)
    ncls0_all = len(cls) - ncls1_all
    ncls0 = 0 
    ncls1 = 0
    cut_min = -1
    ent_min = 1e9
    para = {}
    if cls_sort[0] == 1:
        ncls1 = 1
    else:
        ncls0 = 1
    for cut in range(1,len(ind)):
        
        if cls_sort[cut-1] != cls_sort[cut]:
            ent1 = -np.log(ncls0/cut+1e-100)*(ncls0/cut) - np.log(ncls1/cut+1e-100)*(ncls1/cut)
            n0 = ncls0_all - ncls0
            n1 = ncls1_all - ncls1
            rm = len(cls) - cut
            ent2 = -np.log(n0/rm + 1e-100)*(n0/rm) - np.log(n1/rm +1e-100) *(n1/rm)
            ent = cut/len(cls) * ent1 + rm/len(cls) * ent2
            if ent < ent_min:
                cut_min = (val[ind[cut-1]]+ val[ind[cut]])/2
                ent_min = ent
                para['n00'] = ncls0
                para['n01'] = ncls1
                para['n10'] = n0
                para['n11'] = n1
        else:
            if cls_sort[cut] == 1:
                ncls1 += 1
            else:
                ncls0 += 1
            continue
    return cut_min, ent_min, para

# =============================================================================
# val = np.array([1,2,3,4,5,6])
# cls = np.array([0,0,1,1,1,1])
# cut, ent = calc_ent(val,cls)
# =============================================================================

import pandas as pd
data = pd.read_csv("Social_Network_Ads.csv")
x = data.iloc[:,[2,3]].values
y = data.iloc[:,-1].values

# entropy is not used
from sklearn.model_selection import train_test_split
xtr, xt, ytr, yt = train_test_split(x,y)

cut0, _ , para0 = calc_ent(xtr[:,0], ytr)
cut1, _ , para1 = calc_ent(xtr[:,1], ytr)

# =============================================================================
# p_cls1 = sum(ytr)/len(ytr)
# p_cls0 = 1-p_cls1
# x00 = sum(xtr[:,0] < cut0)
# x01 = sum(xtr[:,0] > cut0)
# x10 = sum(xtr[:,1] < cut1)
# x11 = sum(xtr[:,1] > cut1)
# p_f00 = x00/len(ytr)
# p_f01 = x01/len(ytr)
# p_f10 = x10/len(ytr)
# p_f11 = x11/len(ytr)
# =============================================================================



p_cls1 = sum(ytr)/len(ytr)
p_cls0 = 1-p_cls1
a = np.std(xtr[:,0])/2
b = np.std(xtr[:,1])/2

# likehood 
def calc_fea(x, xs, ys, d):
    cnt0 , cnt1 = 0 , 0
    for i in range(len(xs)):
        if(np.abs(xs[i]-x)<d):
            if(ys[i]==0):
                cnt0 += 1
            else:
                cnt1 += 1
    return cnt0/(len(ys)-sum(ys))+1e-100, cnt1/sum(ys)+1e-100

xt_p = np.zeros((len(xt), 2))
for i in range(len(xt)):
   p00, p01 = calc_fea(xt[i, 0], xtr[:,0], ytr, a)
   p10, p11 = calc_fea(xt[i, 1], xtr[:,1], ytr, b)
   p0 = p00 * p10 * p_cls0
   p1 = p01 * p11 * p_cls1
   xt_p[i] = [p0, p1]

yp = np.argmax(xt_p,1)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true=yt, y_pred=yp)

    









