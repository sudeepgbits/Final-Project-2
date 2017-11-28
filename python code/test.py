# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 20:14:11 2017

@author: SUDEEP
"""

import numpy as np

import matplotlib.mlab as mlab



a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(a)

k = np.array([[1, 2], [6, 3]])
print(k)

idx = mlab.find(a == k)

print(idx)