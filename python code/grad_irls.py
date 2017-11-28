# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 01:24:57 2017

@author: SUDEEP
"""

import numpy as np
import cv2
from numpy.linalg import inv


import scipy.ndimage.filters as nd_filters
import scipy.signal as spsig
import scipy as sp
from math import exp
from scipy import sparse
import deghost
from scipy.sparse import spdiags
from scipy.fftpack import dst, idst

from math import cos
from math import pi

def get_k(h, w, dx, dy, c):
    all_ids = np.zeros((1,h*w), dtype=np.uint64)
    for i in range(0,h*w):
        all_ids[0,i] = i
    all_ids = np.reshape(all_ids, [h, w])
    self_ids=all_ids
    
    #circle shift
    negh_ids2 = np.roll(all_ids, [dx, 0], axis=0)
    negh_ids2 = np.roll(negh_ids2, [0, dy], axis=1)
    
    #ncircle shift
    negh_ids = np.roll(all_ids, [0, dy], axis=1)
    negh_ids = np.roll(negh_ids, [dx, 0], axis=0)
    negh_ids[:,0:dy] =0
    negh_ids[0:dx,:] =0

    ind = np.ones((h,w), dtype=np.uint64)
    indc = ind * c
    indc[negh_ids==0]=0
    
    #S_plus = sparse.csr_matrix(ind)
    
    S_plus = sparse.csr_matrix((np.asarray(ind.T).reshape(-1),(np.asarray(self_ids.T).reshape(-1),np.asarray(self_ids.T).reshape(-1))),shape=(np.size(ind),np.size(ind)))

    #S_minus = sparse.csr_matrix(indc)
    S_minus = sparse.csr_matrix((np.asarray(indc.T).reshape(-1),(np.asarray(self_ids.T).reshape(-1),np.asarray(negh_ids2.T).reshape(-1))),shape=(np.size(indc),np.size(indc)))

    A = S_plus + S_minus
    
    return[A, self_ids, negh_ids2, ind, indc]
    
    

def get_fx(h,w):
    all_ids = np.zeros((1,h*w), dtype=np.uint64)
    for i in range(0,h*w):
        all_ids[0,i] = i
    all_ids = np.reshape(all_ids, [h, w])
    self_ids=all_ids;
    negh_ids = np.roll(all_ids,[0, -1], axis=1)
    ind = np.ones((h,w), dtype=np.uint64)
    ind2 = ind
    ind2[:,-1] = 0
    #S_plus = sparse.csr_matrix(ind)
    S_plus = sparse.csr_matrix((np.asarray(ind.T).reshape(-1),(np.asarray(self_ids.T).reshape(-1),np.asarray(self_ids.T).reshape(-1))),shape=(np.size(ind),np.size(ind)))

    print('S_plus dimension =', np.shape(S_plus.toarray()))
    #S_minus = sparse.csr_matrix(ind2)
    S_minus = sparse.csr_matrix((np.asarray(ind2.T).reshape(-1),(np.asarray(self_ids.T).reshape(-1),np.asarray(negh_ids.T).reshape(-1))),shape=(np.size(ind2),np.size(ind2)))

    A=S_plus-S_minus
    
    return[A, self_ids, negh_ids, ind, ind2]
    

def get_fy(h,w):
    all_ids = np.zeros((1,h*w), dtype=np.uint64)
    for i in range(0,h*w):
        all_ids[0,i] = i
    all_ids = np.reshape(all_ids, [h, w])
    self_ids=all_ids;
    negh_ids = np.roll(all_ids,[-1, 0], axis=0)
    ind = np.ones((h,w), dtype=np.uint64)
    ind2 = ind
    ind2[-1,:] = 0
    #S_plus = sparse.csr_matrix(ind)
    S_plus = sparse.csr_matrix((np.asarray(ind.T).reshape(-1),(np.asarray(self_ids.T).reshape(-1),np.asarray(self_ids.T).reshape(-1))),shape=(np.size(ind),np.size(ind)))

    #S_minus = sparse.csr_matrix(ind2)
    S_minus = sparse.csr_matrix((np.asarray(ind2.T).reshape(-1),(np.asarray(self_ids.T).reshape(-1),np.asarray(negh_ids.T).reshape(-1))),shape=(np.size(ind2),np.size(ind2)))

    A=S_plus-S_minus
    
    return[A, self_ids, negh_ids, ind, ind2]
      
    
def get_fu(h,w):
    all_ids = np.zeros((1,h*w), dtype=np.uint64)
    for i in range(0,h*w):
        all_ids[0,i] = i
    all_ids = np.reshape(all_ids, [h, w])
    self_ids=all_ids;
    
    negh_ids = np.roll(all_ids, [1, 0], axis=0)
    negh_ids = np.roll(negh_ids, [0, -1], axis=1)
    
    ind = np.ones((h,w), dtype=np.uint64)
    ind2 = ind
    ind2[:,-1] = 0
    ind2[1,:] = 0
    #S_plus = sparse.csr_matrix(ind)
    S_plus = sparse.csr_matrix((np.asarray(ind.T).reshape(-1),(np.asarray(self_ids.T).reshape(-1),np.asarray(self_ids.T).reshape(-1))),shape=(np.size(ind),np.size(ind)))

    #S_minus = sparse.csr_matrix(ind2)
    S_minus = sparse.csr_matrix((np.asarray(ind2.T).reshape(-1),(np.asarray(self_ids.T).reshape(-1),np.asarray(negh_ids.T).reshape(-1))),shape=(np.size(ind2),np.size(ind2)))

    A=S_plus-S_minus
    
    return[A, self_ids, negh_ids, ind, ind2]
      
    
def get_fv(h,w):
    all_ids = np.zeros((1,h*w), dtype=np.uint64)
    for i in range(0,h*w):
        all_ids[0,i] = i
    all_ids = np.reshape(all_ids, [h, w])
    self_ids=all_ids;
    
    negh_ids = np.roll(all_ids, [-1, 0], axis=0)
    negh_ids = np.roll(negh_ids, [0, -1], axis=1)
    
    ind = np.ones((h,w), dtype=np.uint64)
    ind2 = ind
    ind2[:,-1] = 0
    ind2[-1,:] = 0
    #S_plus = sparse.csr_matrix(ind)
    S_plus = sparse.csr_matrix((np.asarray(ind.T).reshape(-1),(np.asarray(self_ids.T).reshape(-1),np.asarray(self_ids.T).reshape(-1))),shape=(np.size(ind),np.size(ind)))

    #S_minus = sparse.csr_matrix(ind2)
    S_minus = sparse.csr_matrix((np.asarray(ind2.T).reshape(-1),(np.asarray(self_ids.T).reshape(-1),np.asarray(negh_ids.T).reshape(-1))),shape=(np.size(ind2),np.size(ind2)))

    A=S_plus-S_minus
    
    return[A, self_ids, negh_ids, ind, ind2]
      
    
def get_lap(h,w):
    all_ids = np.zeros((1,h*w), dtype=np.uint64)
    for i in range(0,h*w):
        all_ids[0,i] = i
    all_ids = np.reshape(all_ids, [h, w])
    self_ids=all_ids;
    ind = np.ones((h,w), dtype=np.uint64)
    #S_plus = sparse.csr_matrix(4*ind)
    S_plus = sparse.csr_matrix((np.asarray(4*ind.T).reshape(-1),(np.asarray(self_ids.T).reshape(-1),np.asarray(self_ids.T).reshape(-1))),shape=(np.size(ind),np.size(ind)))

    negh_ids = np.roll(all_ids, [0, -1], axis=1)
    ind2 = ind
    ind2[:,-1] = 0
    S_minus_1 = sparse.csr_matrix((np.asarray(ind2.T).reshape(-1),(np.asarray(self_ids.T).reshape(-1),np.asarray(negh_ids.T).reshape(-1))),shape=(np.size(ind2),np.size(ind2)))

    negh_ids = np.roll(all_ids, [0, 1], axis=1)
    ind2 = ind
    ind2[:,0] = 0
    S_minus_2 = sparse.csr_matrix((np.asarray(ind2.T).reshape(-1),(np.asarray(self_ids.T).reshape(-1),np.asarray(negh_ids.T).reshape(-1))),shape=(np.size(ind2),np.size(ind2)))

    negh_ids = np.roll(all_ids, [-1, 0], axis=0)
    ind2 = ind
    ind2[-1,:] = 0
    S_minus_3 = sparse.csr_matrix((np.asarray(ind2.T).reshape(-1),(np.asarray(self_ids.T).reshape(-1),np.asarray(negh_ids.T).reshape(-1))),shape=(np.size(ind2),np.size(ind2)))

    negh_ids = np.roll(all_ids, [1, 0], axis=0)
    ind2 = ind
    ind2[0,:] = 0
    S_minus_4 = sparse.csr_matrix((np.asarray(ind2.T).reshape(-1),(np.asarray(self_ids.T).reshape(-1),np.asarray(negh_ids.T).reshape(-1))),shape=(np.size(ind2),np.size(ind2)))

    A = S_plus-S_minus_1 - S_minus_2 - S_minus_3 - S_minus_4
    
    
    
    
    return[A, self_ids, negh_ids, ind, ind2]
      



def irls_grad(I_x, tx, out_xi, mh, configs, mx, my,  mu, mv, mlap):
    p = configs.p
    num_px=configs.num_px
    out_x=out_xi
    print('mx shape =', np.shape(mx))
    if configs.use_cross:
        mcross = mx.dot(my)
    if configs.use_lap2:
        mx2 = mx.dot(mx)
        my2 = my.dot(my)
        
    for i in range(0,configs.niter):
        if (configs.delta == 'exp_fall'):
            delta = 0.01*exp(-(i-6)*0.4)
        else:
            delta=configs.delta
        
        out_x1 = np.reshape(out_x, (np.size(out_x),1), order='F')
        
        w1 = (abs(out_x1) **2 + delta)**(p/2-1)
        I_x1 = np.reshape(I_x, (np.size(I_x),1), order='F')
        print('hi')
        print('I_x1 shape =' + str(np.shape(I_x1)) + 'size =' + str(np.size(I_x1)))
        print('mh shape=', np.shape(mh))
        w2 = (abs((mh.dot((I_x1 - out_x1))))**2 + delta) ** (p/2-1)
        print('bye')
        
        data2= np.reshape(w1, (1,np.size(w1)), order='F')
        A1 = spdiags(data2, 0, num_px, num_px) 
        data2= np.reshape(w2, (1,np.size(w2)), order='F')
        
        A2_temp = np.dot(spdiags(data2, 0, num_px, num_px), mh)
        A2 = np.dot(np.transpose(mh), A2_temp)
        
        Atot = A1 + A2
        Ab = A2
        
        if configs.use_lap :
            out_x1 = np.reshape(out_x, (np.size(out_x),1), order='F')
            I_x1 = np.reshape(I_x, (np.size(I_x),1), order='F')
            w3 = (abs((np.dot(mx,out_x1)))**2 + delta)**(p/2-1)
            w4 = (abs((np.dot(my,out_x1)))**2 + delta)**(p/2-1)
            val_w5 = np.dot(mx,mh)
            val_w5 = np.dot(val_w5, (I_x1 - out_x1))
            w5 = (abs(val_w5)**2 + delta)**(p/2-1)
            val_w6 = np.dot(my,mh)
            val_w6 = np.dot(val_w6, (I_x1 - out_x1))
            w5 = (abs(val_w6)**2 + delta)**(p/2-1)
    
            A3 = np.dot(np.transpose(mx),spdiags_cal(w3,0,num_px,num_px))
            A3 = np.dot(A3,mx)
            
            A4 = np.dot(np.transpose(my),spdiags_cal(w4,0,num_px,num_px))
            A4 = np.dot(A4,my)
            
            A5 = np.dot(np.transpose(mx),spdiags_cal(w5,0,num_px,num_px))
            A5 = np.dot(A4,mx)
            
            A6 = np.dot(np.transpose(my),spdiags_cal(wy,0,num_px,num_px))
            A6 = np.dot(A6,my)
            
            A7 = np.dot(np.transpose(mh),A5+A6)
            A7 = np.dot(A6,mh)
            
            Atot = Atot+A3+A4+A7
            Ab = Ab + A7
            
            
        if configs.use_diagnoal:
            out_x1 = np.reshape(out_x, (np.size(out_x),1), order='F')
            I_x1 = np.reshape(I_x, (np.size(I_x),1), order='F')
            
            w8 = (abs(np.dot(mu,out_x1))**2 + delta)**(p/2-1)
            w9 = (abs(np.dot(mv,out_x1))**2 + delta)**(p/2-1)
            
            w10_val = np.dot(mu,mh)
            w10_val = np.dot(w10_val,(I_x1-out_x1))
            w10 = (abs(w10_val)**2 + delta)**(p/2-1)
            
            w11_val = np.dot(mv,mh)
            w11_val = np.dot(w11_val,(I_x1-out_x1))
            w11 = (abs(w11_val)**2 + delta)**(p/2-1)
            
            A8 = np.dot(np.transpose(mu),spdiags_cal(w8,0,num_px,num_px))
            A8 = np.dot(A8,mu)
            
            A9 = np.dot(np.transpose(mv),spdiags_cal(w9,0,num_px,num_px))
            A9 = np.dot(A9,mv)
            
            A10 = np.dot(np.transpose(mu),spdiags_cal(w10,0,num_px,num_px))
            A10 = np.dot(A10,mu)
            
            A11 = np.dot(np.transpose(mv),spdiags_cal(w11,0,num_px,num_px))
            A11 = np.dot(A11,mv)
            
            A12 = np.dot(np.transpose(mh),A10+A11)
            A12 = np.dot(A12,mh)
            
            Atot = Atot+A8+A9+A12
            Ab = Ab+A12
            
        if configs.use_lap2:
            out_x1 = np.reshape(out_x, (np.size(out_x),1), order='F')
            I_x1 = np.reshape(I_x, (np.size(I_x),1), order='F')
            
            w17 = (abs(np.dot(mx2,out_x1))**2 + delta)**(p/2-1)
            w18 = (abs(np.dot(my2,out_x1))**2 + delta)**(p/2-1)
            
            w19_val = np.dot(mx2,mh)
            w19_val = np.dot(w19_val,(I_x1-out_x1))
            w19 = (abs(w19_val)**2 + delta)**(p/2-1)
            
            w20_val = np.dot(my2,mh)
            w20_val = np.dot(w20_val,(I_x1-out_x1))
            w20 = (abs(w20_val)**2 + delta)**(p/2-1)
            
            A17 = np.dot(np.transpose(mx2),spdiags_cal(w17,0,num_px,num_px))
            A17 = np.dot(A17,mx2)
            
            A18 = np.dot(np.transpose(my2),spdiags_cal(w18,0,num_px,num_px))
            A18 = np.dot(A18,my2)
            
            A19 = np.dot(np.transpose(mx2),spdiags_cal(w19,0,num_px,num_px))
            A19 = np.dot(A19,mx2)
            
            A20 = np.dot(np.transpose(my2),spdiags_cal(w20,0,num_px,num_px))
            A20 = np.dot(A20,my2)
            
            A21 = np.dot(np.transpose(mh),A19+A20)
            A21 = np.dot(A21,mh)
            
            Atot=Atot+A17+A18+A21
            Ab=Ab+A21
            
        if configs.use_cross :
            out_x1 = np.reshape(out_x, (np.size(out_x),1), order='F')
            I_x1 = np.reshape(I_x, (np.size(I_x),1), order='F')
            w15 = (abs(np.dot(mcross,out_x1))**2 + delta)**(p/2-1)
            
            w16_val = np.dot(mcross,mh)
            w16_val = np.dot(w16_val,(I_x1-out_x1))
            w16 = (abs(w16_val)**2 + delta)**(p/2-1)
            
            A15 = np.dot(np.transpose(mcross),spdiags_cal(w15,0,num_px,num_px))
            A15 = np.dot(A15,mcross)
            
            A16 = np.dot(np.transpose(mh),np.transpose(mcross))
            A16 = np.dot(A16,spdiags_cal(w16,0,num_px,num_px))
            A16 = np.dot(A16,mcross)
            A16 = np.dot(A16,mh)
            
            Atot=Atot+A15+A16
            Ab=Ab+A16
        
        I_x1 = np.reshape(I_x, (np.size(I_x),1), order='F')
        out_x = (np.dot(Ab,I_x1))/Atot
       
        res = I_x1 - out_x
        
            
    out_x = np.reshape(out_x, configs.dims)
            
            
    return out_x
    
  
     

def grad_irls(I_in, configs):
    
    dx = configs.dx
    dy = configs.dy
    c = configs.c
    configs.dims=[np.shape(I_in)[0], np.shape(I_in)[1]]
    dims = configs.dims
    
    configs.delta= 1 * exp(-4)
    configs.p=0.2
    configs.use_lap=1
    configs.use_diagnoal=1
    configs.use_lap2=1
    configs.use_cross=0
    configs.niter=20
    
    [A, self_ids, negh_ids2, ind, indc] = get_k(configs.h, configs.w, dx, dy, c)
    
    mk = A
    #print('mk =', mk)
    #mh = inv(A)
    mh = sparse.csr_matrix(np.linalg.pinv(A.toarray()))
    
    
    print('calculating mx')
    [Ax, self_idsx, negh_idsx, indx, ind2x] = get_fx(configs.h, configs.w)
    mx = Ax
    
    print('calculating Ay')
    [Ay, self_idsy, negh_idsy, indy, ind2y] = get_fy(configs.h, configs.w)
    my = Ay
    
    print('calculating Au')
    [Au, self_idsu, negh_idsu, indu, ind2u] = get_fu(configs.h, configs.w)
    mu = Au
    
    print('calculating Av')
    [Av, self_idsv, negh_idsv, indv, ind2v] = get_fv(configs.h, configs.w)
    mv = Av
    
    print('calculating Alap')
    [Alap, self_idslap, negh_idslap, indlap, ind2lap] = get_lap(configs.h, configs.w)
    mlap = Alap
    
    k = configs.ch
    
    kernel = np.array([-1, 1])
    
    I_x = cv2.filter2D(I_in, -1, kernel, borderType=cv2.BORDER_DEFAULT)
    I_x = (I_x - I_x.min()) / (I_x.max() - I_x.min())
    
    
    kernel = np.array([[-1], [1]])
    I_y = cv2.filter2D(I_in, -1, kernel, borderType=cv2.BORDER_DEFAULT)
    I_y = (I_y - I_y.min()) / (I_y.max() - I_y.min())
    
    out_xi=I_x/2
    out_yi=I_y/2
    
    print('calculating out_x from irls_grad')
    out_x=irls_grad(I_x, [], out_xi, mh, configs, mx, my,  mu, mv, mlap)
    
    
    out_x1 = np.reshape(out_x, (np.size(out_x),1), order='F')
    I_x1 = np.reshape(I_x, (np.size(I_x),1), order='F')
    outr_x = reshape(np.dot(mh,(I_x1-out_x1)), dims)
    
    print('calculating out_y from irls_grad')
    out_y=irls_grad(I_y, [], out_yi, mh, configs, mx, my, mu, mv, mlap)
    out_y1 = np.reshape(out_y, (np.size(out_y),1), order='F')
    I_y1 = np.reshape(I_y, (np.size(I_y),1), order='F')
    outr_x = reshape(np.dot(mh,(I_y1-out_y1)), dims)
    
    
    print('poisson_solver_function.....1')
    I_t = poisson_solver_function(out_x, out_y, I_in)
    print('poisson_solver_function.....2')
    I_r = poisson_solver_function(outr_x, outr_y, I_in)
    
    return [I_t, I_r, configs]


def poisson_solver_function(gx,gy,boundary_image):
    [H,W] = np.shape(boundary_image)
    [H1,W1] = np.shape(gx)
    [H2,W2] = np.shape(gy)
    
    if not ((H1 == H2) or (H == H1) or (W1 == W2)):
        print('Size of gx,gy and boundary images is not same')
        exit
    
    gxx = np.zeros((H,W))
    gyy = np.zeros((H,W))
    f = np.zeros((H,W),dtype=np.uint64)
    
    j = np.zeros((1,H))
    for i in range(0,H):
        j[0,i] = i
    
    k = np.zeros((1,W))
    for i in range(0,W):
        k[0,i] = i
    
    gyy[j+1,k] = gy[j+1,k] - gy[j,k]
    gxx[j,k+1] = gx[j,k+1] - gx[j,k]
    f = gxx + gyy

    boundary_image[1:-1, 1:-1] = 0
    
    j = np.zeros((1,H))
    for i in range(2,H):
        j[0,i] = i
    
    k = np.zeros((1,W))
    for i in range(2,W):
        k[0,i] = i

    f_bp = np.zeros((H,W))
    
    for t in range (0,j):
        for l in range(0,k):
            f_bp[t,l] = -4*boundary_image[t,l] + boundary_image[t,l+1] + boundary_image[t,l-1] + boundary_image[t-1,k] + boundary_image[t+1,l]
    
    f1 = f - np.reshape(f_bp, (H,W))
    
    f2 = f1[2:-1, 2:-1]
    
    tt = dst(f2)
    f2sin = np.transpose(dst(np.transpose(tt)))


    u = np.zeros((1,W-1))
    v = np.zeros((1,H-1))
    
    for t in range(0,W-1):
        u[0,t] = t
    
    
    for t in range(0,H-1):
        v[0,t] = t
        
    x, y = np.meshgrid(u, v)
    denom = np.zeros((np.shape(x)))
    f3 = np.zeros((np.shape(x)))
    for q in range(0,np.shape(x)[0]):
        for w in range(0,np.shape(x)[0]):
            denom[q,w] = (2*cos(pi*x[q,w]/(W-1))-2) + (2*cos(pi*y[q,w]/(H-1)) - 2)
            f3[q,w] = f2sin[q,w]/denom[q,w]
    
    tt = idst(f3)
    
    img_tt = idst(np.transpose(tt))
    img_direct = boundary_image
    img_direct[2:-1,2:-1] = 0        
    img_direct[2:-1,2:-1] = img_tt
       
    return img_direct

def spdiags_cal(x, t, num_px, num_px2):
    
    data2= np.reshape(x, (1,np.size(x)), order='F')
    spdiageI = spdiags(data2, t, num_px, num_px2)
    
    return spdiageI       
    #### look at ignore first row first col etc...
    
    