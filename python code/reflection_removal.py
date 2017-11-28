import numpy as np
import scipy as sp
import scipy.io as spio

import cv2

def get_k(h, w, dx, dy, c):
    idx = np.arange((h*w)).reshape(w,h).transpose()
    idx_shift = np.roll(idx,(dy,dx),axis=(0,1))
    data = np.ones((h,w),dtype=np.float)
    data_c = c * np.ones((h,w),dtype=np.float)
    data_c[0:dy,:] = 0
    data_c[:,0:dx] = 0
    return sp.sparse.coo_matrix((data.ravel(),(idx.ravel(),idx.ravel())),shape=(h*w,h*w)) + sp.sparse.coo_matrix((data_c.ravel(),(idx.ravel(),idx_shift.ravel())),shape=(h*w,h*w))
    
def merge_patches(patch,h,w,psize):
    patches = patch.transpose().reshape((h-psize+1, w-psize+1, psize**2))
    out = np.zeros((h, w))
    k=0;
    for j in range(psize):
      for i in range(psize):
        out[i:i+h-psize+1, j:j+w-psize+1] = out[i:i+h-psize+1, j:j+w-psize+1] + patches[:,:,k]; 
        k=k+1;
    return out
def merge_two_patches(est_t, est_r, h, w, psize):
    t_merge = merge_patches(est_t, h, w, psize).ravel()
    r_merge = merge_patches(est_r, h, w, psize).ravel()
    return np.hstack([t_merge,r_merge]).reshape(-1,1)
    
def im2patches(img, psize):

    m,n = img.shape
    s0, s1 = img.strides    
    nrows = m-psize+1
    ncols = n-psize+1
    shp = psize,psize,nrows,ncols
    strd = s0,s1,s0,s1

    return np.lib.stride_tricks.as_strided(img, shape=shp, strides=strd).reshape(psize**2,-1)

def loggausspdf2(x,sigma):
    d = x.shape[0];
    R = np.linalg.cholesky(sigma).T
    B = np.linalg.solve(R,x)
    q = np.sum(B**2,axis=0) # quadratic term (M distance)
    c = d*np.log(2*np.pi)+2*np.sum(np.log(np.diag(R)));   # normalization constant
    y = -(c+q)/2;
    return y
def aprxMAPGMM(x,psize,noiseSD,h,w,GS):
    SigmaNoise = (noiseSD**2)*np.identity(psize);
    SigmaNoise1 = (noiseSD**2)*np.identity(psize**2);
    mean_x = np.mean(x,axis=0).reshape((1,-1))
    x = x - mean_x
    
    dim = GS['dim']+0
    nmodels = GS['nmodels'] + 0
    means = GS['means'] + 0
    covs = GS['covs'] + 0
    invcovs = GS['invcovs'] + 0
    mixweights = GS['mixweights'] + 0
    
    biased_covs = covs + np.tile(SigmaNoise.ravel().reshape(-1,1),[1,1,nmodels])
    PYZ = np.zeros((nmodels,x.shape[1]))
    for i in range(nmodels):
        PYZ[i,:] = np.log(mixweights[i]) + loggausspdf2(x,biased_covs[:,:,i])
    
    # find the most likely component for each patch
    ks = np.argmax(PYZ,axis=0)

    #and now perform weiner filtering
    Xhat = np.zeros(x.shape).reshape(dim,-1);
    for i in range(nmodels):
        inds = np.where(ks==i)
        Xhat[:,inds[0]] = np.linalg.solve((covs[:,:,i]+SigmaNoise1),(np.dot(covs[:,:,i],x[:,inds].reshape(dim,-1)) + np.dot(SigmaNoise1,np.tile(means[:,i].ravel().reshape(-1,1),[1,inds[0].size]))))

    return Xhat + mean_x
    
def patch_gmm(img,dx,dy,c):
    (h,w) = img.shape
    id_mat = sp.sparse.identity((h*w))
    k_mat = get_k(h, w, dx, dy, c)
    A = sp.sparse.hstack([id_mat,k_mat])
    
    lbda = 1e6;
    psize = 8;
    num_patches = (h-psize+1)*(w-psize+1)
    mask=merge_two_patches(np.ones((psize**2, num_patches)),np.ones((psize**2, num_patches)), h, w, psize)
    
    beta = 200
    beta_factor = 2
    mat = spio.loadmat('GSModel_8x8_200_2M_noDC_zeromean.mat', squeeze_me=True)
    GS = mat['GS']
    
    #TODO: irls
    I_t_i = np.zeros((h,w))
    I_r_i = np.zeros((h,w))
    
    est_t = im2patches(I_t_i, psize)
    est_r = im2patches(I_r_i, psize)
    
        
    for i in range(25):
        print 'Optimizine %d iter...\n'%i
        x0 = np.hstack([I_t_i.ravel(),I_r_i.ravel()]).reshape(-1,1)
        sum_piT_zi = merge_two_patches(est_t, est_r, h, w, psize);
        sum_zi_2 = np.linalg.norm(est_t.ravel())**2 + np.linalg.norm(est_r.ravel())**2;
        z = (lbda * A.T.tocsr().dot(img.ravel())).reshape(-1,1) +  (beta * sum_piT_zi); 
        def calc_func_value_and_gradient(x,*args):
            f = lbda * np.linalg.norm(A.dot(x) - img.ravel())**2 + beta*(sum(x.reshape(-1,1)*mask*x.reshape(-1,1) - 2 * x.reshape(-1,1)* sum_piT_zi.ravel().reshape(-1,1)) + sum_zi_2)
            g = 2*(lbda * (A.T.tocsr().dot(A.dot(x))).reshape(-1,1) + beta*(mask*x.reshape(-1,1)) - z)
            return f,g
        (out,f,d) = sp.optimize.fmin_l_bfgs_b(calc_func_value_and_gradient, x0,args=(A,mask),bounds=[(0,1) for i in range(h*w*2)],m=50,factr=1e4,pgtol=1e-8)
        
        out = out.reshape(h,w,2)
        I_t_i = out[:,:,0]
        I_r_i = out[:,:,1] 

        #Restore patches using the prior
        est_t = im2patches(I_t_i, psize);
        est_r = im2patches(I_r_i, psize);
        noiseSD=(1/beta)**0.5;
        
        est_t = aprxMAPGMM(est_t,psize,noiseSD,h,w, GS)
        est_r = aprxMAPGMM(est_r,psize,noiseSD,h,w, GS)

        beta = beta*beta_factor;
    return I_t_i,I_r_i

if __name__ == "__main__":
    mat = spio.loadmat('apples.mat', squeeze_me=True)

    I_in = mat['I_in'] # array
    dx = mat['dx'] # structure containing an array
    dy = mat['dy'] # array of structures
    c = mat['c']
    (h,w,c) = I_in.shape
    for i in range(c):
        I_t,I_r = patch_gmm(I_in[:,:,i],dx,dy,c)
        cv2.imwrite('trans'+str(i)+'.pgm',I_t)
        cv2.imwrite('reflc'+str(i)+'.pgm',I_t)
