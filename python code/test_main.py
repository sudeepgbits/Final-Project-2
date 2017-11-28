# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 23:31:31 2017

@author: SUDEEP
"""

import numpy as np
import cv2
import scipy as sp
import scipy.io as scio

class configs:
    dx = 15
    dy = 7
    c = 0.5
    padding = 0
    match_input = 0
    linear = 0
    h = 64
    w = 64
    num_px = 4096
    ch = 1
    non_negative = 1
    beta_factor = 2
    beta_i = 200
    dims = [64,64]
    delta = 1.00e-04
    p = 0.2
    use_lap = 1
    use_diagonal = 1
    use_lap2 = 1
    use_cross = 0
    niter = 20
    
    
    
    
    
    
import grad_irls
#import deghost
#import est_attenuation
def mat2image(input):
   
    return ((input * 255).astype(np.uint8))

print('loading apples..')
apples = scio.loadmat('apples.mat', squeeze_me=True)

print('initializing variables')
Iin = apples['I_in']
c = apples['c']
dx = apples['dx']
dy = apples['dy']

#configs = scio.loadmat('apples.mat', squeeze_me=True)
config1 = configs()



print('calling kernel estimate..')
grad_irls.grad_irls(Iin,config1)

grad_irls.dump_session(grad_irls.pkl)
#I2 = deghost.kernel_est(Iin)


#I1 = deghost.simple()

#I1_img = mat2image(I1)

#dx = 30
#dy = 0
#atten = est_attenuation.est_attenuation(Iin, dx, dy)
#cv2.imshow('image',I1_img)
#cv2.waitKey(0)
