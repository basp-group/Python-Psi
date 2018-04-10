'''
Created on 14 Mar 2018

@author: mjiang
ming.jiang@epfl.ch
'''

import numpy as np
import scipy.linalg as LA

def soft(alpha,th):
    '''
    soft-thresholding function
    
    @param alpha: wavelet coefficients
    
    @param th: threshold level
    
    @return: wavelet coefficients after soft threshold
    
    '''
    return np.sign(alpha) * (np.abs(alpha) - th)

def hard(alpha,th):
    '''
    hard-thresholding function
    
    @param alpha: wavelet coefficients 
    
    @param th: threshold level
    
    @return: wavelet coefficients after hard threshold
    
    '''
    coef = np.copy(alpha)
    (coef)[np.abs(alpha)<=th] = 0
    return coef 

def proj_sc(alpha, rad):
    '''
    scaling, projection on L2 norm
    
    @param alpha: coefficients to be processed
    
    @param rad: radius of the l2 ball
    
    @return: coefficients after projection
    
    '''
    return alpha * min(rad/LA.norm(alpha), 1)
    