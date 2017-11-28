# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 02:47:44 2017

@author: sg946668
"""

from scipy import sparse

h = 8
w = 8
ind = np.ones((h,w),dtype=np.uint64)
all_ids = np.zeros((h,w),dtype=np.uint64)
k = 0
for i in range(0,h):
    for j in range(0,w):
        all_ids[i,j] = k
        k = k+1
self_ids = all_ids
S_plus = sparse.csr_matrix((np.asarray(ind).reshape(-1),(np.asarray(self_ids).reshape(-1),np.asarray(self_ids).reshape(-1))),shape=(np.size(ind),np.size(ind)))

"""
new_ind = np.asarray(ind).reshape(-1)
new_all_ids = np.asarray(all_ids).reshape(-1)
S_plus = sparse.csr_matrix((new_ind,(new_all_ids,new_all_ids)),shape=(64,64))

S_plus2 = sparse.csr_matrix(ind)
"""