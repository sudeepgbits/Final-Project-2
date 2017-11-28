# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 01:20:23 2017

@author: SUDEEP
"""
import numpy as np
import cv2


import scipy.ndimage.filters as nd_filters
import scipy.signal as spsig
import scipy as sp
from math import exp
import deghost


def get_patch(I, x, y, hw):
    #print('x value in patch = ', x )
    if ((x>hw) and (x < I.shape[1]-hw) and (y>hw) and (y< I.shape[0] - hw)):
        p = I[y-hw:y+hw, x-hw:x+hw]
    else :
        p = []
    
    return p

def est_attenuation(I, dx, dy):
    #cv2.imshow('image', I)
    #cv2.waitKey(0)
    num_features = 200
    I = (I - I.min()) / (I.max() - I.min())
   
    I = deghost.mat2image(I)
    cv2.imwrite('apples.png', I)
    
    #I_in=I_in.astype(np.float32)

    #I_in = cv2.cvtColor(I_in, cv2.COLOR_RGB2GRAY)
    I = cv2.imread('apples.png')

    #I = np.asarray(I)
    #cns = cv2.cornerHarris(I,2,3,0.04) 
    
    
    feat_detector = cv2.ORB(nfeatures=num_features)
    #image_1_kp, image_1_desc = feat_detector.detectAndCompute(I, None)
    image_1_kp, image_1_desc = feat_detector.detectAndCompute(I, None)
    
    

    cns = np.zeros((num_features,2), dtype=np.uint64)
    for i,image_1_kp in enumerate(image_1_kp):
        cns[i,:] = image_1_kp.pt
        
    
    #print(len(cns))  
    #cns = cv2.dilate(cns,None)
    #print('cns =', cns)
    hw = 18
    #m = np.array([])
    score = np.zeros((len(cns),1))
    atten = score * 0
    w = score * 0
    
    for i in range(0,len(cns)):
        
        p1 = get_patch(I, int(cns[i,0]), int(cns[i,1]), hw)
        p2 = get_patch(I, int(cns[i,0] + dx), int(cns[i,1] + dy), hw)
        
        p = np.concatenate((p1, p2), axis = 1)
        if i == 0:
            m = p
        else :
            m = np.concatenate((m, p), axis = 0)
        
        p1 = p1.flatten('F')
        p1 = np.reshape(p1, (len(p1),1))
        p2 = p2.flatten('F')
        p2 = np.reshape(p2, (len(p2),1))
        mean_p1 = np.mean(p1)
        p1 = p1 - mean_p1
        mean_p2 = np.mean(p2)
        p2 = p2 - mean_p2
        score[i] = sum(p1 * p2)/sum(p1 ** 2)**0.5/sum(p2 ** 2)**0.5
        atten[i] = (max(p2)-min(p2))/(max(p1)-min(p1))
        if (atten[i] < 1) and (atten[i] > 0):
            w[i] = exp(-1* score[i]/(2*0.2**2))
    
    c = sum(w * atten)/sum(w)
    
    #print c
    return c