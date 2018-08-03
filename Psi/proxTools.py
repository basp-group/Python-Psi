'''
Created on 14 Mar 2018

@author: mjiang
ming.jiang@epfl.ch
'''

import numpy as np
import scipy.linalg as LA
import sys

def soft(alpha,th):
    '''
    soft-thresholding function
    
    @param alpha: wavelet coefficients
    
    @param th: threshold level
    
    @return: wavelet coefficients after soft threshold
    
    '''
    tmp = np.abs(alpha) - th
    tmp = (tmp + np.abs(tmp))/2.
    return np.sign(alpha) * tmp

def hard(alpha,th):
    '''
    hard-thresholding function
    
    @param alpha: wavelet coefficients 
    
    @param th: threshold level
    
    @return: wavelet coefficients after hard threshold
    
    '''
    return alpha * (alpha>0) 

def proj_sc(alpha, rad):
    '''
    scaling, projection on L2 norm
    
    @param alpha: coefficients to be processed
    
    @param rad: radius of the l2 ball
    
    @return: coefficients after projection
    
    '''
    return alpha * min(rad/LA.norm(alpha), 1)

def l21_norm(alpha, th, mode='soft'):
    '''
    Proximity operator for joint sparsity. l2-norm on rows.
    
    @param alpha: coefficients to be processed, matrix of size [M, N]
    
    @param th: threshold level, vector of size [M]
    
    @param mode: 'soft'-threshold or 'hard'-thresholding
    
    @return: coefficients after joint sparsity, l21-norm (after proximal operation)
    
    '''
    l2norm = LA.norm(alpha,axis=1)
    ind = (l2norm > sys.float_info.epsilon)
    if mode == 'soft':
        l2norm_th = soft(l2norm, th)
        return l2norm_th * alpha[ind]/l2norm[ind,np.newaxis], l2norm_th.sum()
    elif mode == 'hard':
        l2norm_th = hard(l2norm, th)
        return l2norm_th * alpha[ind]/l2norm[ind,np.newaxis], l2norm_th.sum()
    
def nuclear_norm(mat, th, mode='soft'):
    '''
    Proximity operator for nuclear norm.
    
    @param mat: input matrix of size [M, N]
    
    @param th: threshold level, vector of size [K], where K = min(M, N)
    
    @param mode: 'soft'-threshold or 'hard'-thresholding
    
    @return: output matrix, the main diagonal of the diagonal matrix (after proximal operation)
    
    '''
    U, s, Vh = LA.svd(mat, full_matrices=False) 
    if mode == 'soft':
        s1 = soft(s, th)
        S1 = np.diag(s1)
        S1[S1<0] = 0
        return np.dot(U, np.dot(S1, Vh)), s1
    elif mode == 'hard':
        s1 = hard(s, th)
        S1 = np.diag(s1)
        S1[S1<0] = 0
        return np.dot(U, np.dot(S1, Vh)), s1
    