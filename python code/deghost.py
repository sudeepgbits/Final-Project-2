# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 23:23:10 2017

@author: SUDEEP
"""

import numpy as np
import cv2


import scipy.ndimage.filters as nd_filters
import scipy.signal as spsig
import scipy as sp
from math import exp
import est_attenuation

class configs:
    dx = 0
    dy = 0
    c = 0
    
def mat2image(input):
   
    return ((input * 255).astype(np.uint8))

def deghost(test_case):
    if (test_case == 'apple'):
        pass
    else:
        [Iin, config] = simple()

        

    return 0


def circle(h, w, x, y, r):
    I = np.zeros((h, w))
   
    cv2.circle(I, (x, y), r, (255,255,255), -1)
   
    return I


def local_filter(x, order):
    x.sort()
    return x[order]

def ordfilt2(A, order, mask_size):
    return nd_filters.generic_filter(A, lambda x, ord=order: local_filter(x, ord), size=(mask_size, mask_size))



def kernel_est(I_in):
    
    I_in=I_in.astype(np.float32)

    I_in = cv2.cvtColor(I_in, cv2.COLOR_RGB2GRAY)

    Laplacian = np.array([[0, -1, 0], [-1, 4, -1],[0, -1, 0]])
    print('applying filter')
    resp = cv2.filter2D(I_in, -1, Laplacian, borderType=cv2.BORDER_DEFAULT)
    resp = (resp - resp.min()) / (resp.max() - resp.min())
    #cv2.imshow('gray image', resp)
    #cv2.waitKey(0)
    print('resp shape =', np.shape(resp))
    print('calculating correlation')
    auto_corr = spsig.correlate2d(resp, resp)
    print('auto_corr  = ', auto_corr)
    bdry = 370
    print auto_corr.shape
    auto_corr = auto_corr[bdry-1:-bdry, bdry-1:-bdry]
    print auto_corr.shape
    print('calculating max value')
    max_1 = ordfilt2(auto_corr, 24, 5)
    max_2 = ordfilt2(auto_corr, 23, 5)
    
    print('max1 shape=', max_1.shape)
    auto_corr[-1/2 - 4 : -1/2 + 4, -1/2 - 4 : -1/2+4]=0;
    
    print('finding max val')
    
    #candidates = np.where((auto_corr == max_1) and ((max_1 - max_2)>70))
    [candidates, candidates_i, candidates_j] = find(auto_corr, max_1, max_2)
    print ('candidates =', candidates)
    auto_corr_flat = auto_corr.flatten()
    #candidates_val = auto_corr_flat[candidates]
    #print ('candidates val =', candidates_val)
    cur_max = 0
    dx = 0
    dy = 0
    [x,y] = np.shape(auto_corr)
    
    offsetx = x//2 + 1
    offsety = y//2 + 1
    candidates_val = np.array([])
    print('calculating dx dy')
    for i in range(0,np.size(candidates)):
        candidates_val[i] = np.insert(candidates_val, i, auto_corr_flat[int(candidates[i])]) 
        if (candidates_val[i] > cur_max):
            [dy, dx] = [int(candidates_i[i]), int(candidates_j[i])]
            dy = dy - offset[0]
            dx = dx - offset[1]
    
    print(dy)
    print(dx)
    
    c = est_attenuation.est_attenuation(I_in, dx, dy)
    
    return[dx, dy, c]

def find(auto_corr, max_1, max_2):
    [x,y] = np.shape(auto_corr)
    k = 0
    u = 0
    candidates = np.array([])
    candidates_i = np.array([])
    candidates_j = np.array([])
    for i in range(0,x):
        for j in range(0,y):
            if ((auto_corr[i,j] == max_1[i,j]) and ((max_1[i,j] - max_2[i,j])>70)):
                candidates = np.insert(candidates, u, k)
                candidates_i = np.insert(candidates_i, u, i)
                candidates_j = np.insert(candidates_j, u, j)
                u = u+1
            k = k+1
    
    
    return [candidates, candidates_i, candidates_j]
    
    

def simple():
    config1 = configs()
    dimension = [64, 64]
    
    I1 = circle(dimension[0], dimension[1], 40, 40, 20) * 0.4
    I2 = np.zeros((dimension[0], dimension[1]))
    I2[9:49, 4:24]= 0.3
    
    config1.dx = 15
    config1.dy = 7
    config1.c = 0.5
    
    
    return I1
    





def findHomography(image_1_kp, image_2_kp, matches):
    image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    # WRITE YOUR CODE HERE.
    i = 0
    for mat in matches:
        
        #print ('loop =' + str(i))
        image_1_points[i,:,:] = np.float32(image_1_kp[mat.queryIdx].pt)
        
        image_2_points[i,:,:] = np.float32(image_2_kp[mat.trainIdx].pt)
        i = i + 1
    
    #print(image_1_points)
    
    return image_1_points

    
    
    
    
    
    
    