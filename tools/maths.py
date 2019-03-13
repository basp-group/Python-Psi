"""
Created on Dec 1, 2017

@author: mjiang
ming.jiang@epfl.ch
"""

import numpy as np


def pow_method(A, At, im_size, tol, max_iter, verbose=False):
    """
    Computes the spectral radius (maximum eigen value) of the operator A
    
    :param A: function handle of direct operator
    :param At: function handle of adjoint operator
    :param im_size: size of the image
    :param tol: tolerance of the error, stopping criterion
    :param max_iter: max iteration
    :return: spectral radius of the operator
    """
    if len(im_size) == 2:
        x = np.random.randn(im_size[0], im_size[1])
    elif len(im_size) == 3:
        x = np.random.randn(im_size[0], im_size[1], im_size[2])
    x /= np.linalg.norm(x)
    init_val = 1
    
    for it in np.arange(max_iter):
        y = A(x)
        x = At(y)
        val = np.linalg.norm(x)
        rel_var = np.abs(val-init_val) / init_val
        if rel_var < tol:
            break
        init_val = val
        x /= val

        if verbose:
            print("Iteration: ", it, ", val: ", val)
        
    return val
