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

    Laplacian = np.array([[0., -1., 0], [-1., 4., -1.],[0, -1., 0]])
    
    resp = cv2.filter2D(I_in, -1, Laplacian, borderType=cv2.BORDER_DEFAULT)
    #auto_corr = spsig.correlate2d(resp, resp)
    auto_corr = sp.signal.fftconvolve(resp, resp[::-1, ::-1])
    bdry = 370
    auto_corr = auto_corr[bdry-1:-bdry, bdry-1:-bdry]
    max_1 = ordfilt2(auto_corr, 24, 5)
    max_2 = ordfilt2(auto_corr, 23, 5)
    
    (x,y) = auto_corr.shape
    auto_corr[(x/2) - 4 : (x/2) + 4, (y/2) - 4 : (y/2)+4]=0
    
    
    c = np.ones(max_1.shape)
    c[(max_1 - max_2)<=70] = 0
    c[auto_corr != max_1] = 0
    candidates = np.where(c == 1)   
    idx =  np.argmax(auto_corr[candidates])
    dy = candidates[0][idx] - x//2
    dx = candidates[1][idx] - y//2

    c = est_attenuation.est_attenuation(I_in, dx, dy)
    
    return[dx, dy, c]
    

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

    
    
    
    
    
    
    