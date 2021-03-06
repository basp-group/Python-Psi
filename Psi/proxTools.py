"""
Created on 14 Mar 2018

@author: mjiang
ming.jiang@epfl.ch
"""

import numpy as np
import scipy.linalg as LA


def soft(alpha, th):
    """
    soft-thresholding function

    :param alpha: wavelet coefficients
    :param th: threshold level
    :return: wavelet coefficients after soft threshold
    """
    tmp = np.abs(alpha) - th
    tmp = (tmp + np.abs(tmp))/2.
    return np.sign(alpha) * tmp


def hard(alpha, th):
    """
    hard-thresholding function

    :param alpha: wavelet coefficients
    :param th: threshold level
    :return: wavelet coefficients after hard threshold
    """
    return alpha * (np.abs(alpha) > th)


def proj_sc(alpha, rad):
    """
    scaling, projection on L2 norm

    :param alpha: coefficients to be processed
    :param rad: radius of the l2 ball
    :return: coefficients after projection
    """
    return alpha * min(rad/LA.norm(alpha), 1)


def nuclear_norm(mat, th, mode='soft'):
    """
    Proximity operator for nuclear norm.

    :param mat: input matrix of size [M, N]
    :param th: threshold level, vector of size [K], where K = min(M, N)
    :param mode: 'soft'-threshold or 'hard'-thresholding
    :return: output matrix, the main diagonal of the diagonal matrix (after proximal operation)
    """
    U, s, Vh = LA.svd(mat, full_matrices=False)
    if mode == 'soft':
        s1 = soft(s, th)
        S1 = np.diag(s1)
        S1[S1 < 0] = 0
        return np.dot(U, np.dot(S1, Vh)), s1
    elif mode == 'hard':
        s1 = hard(s, th)
        S1 = np.diag(s1)
        S1[S1 < 0] = 0
        return np.dot(U, np.dot(S1, Vh)), s1


def l21_norm(alpha, th, mode='soft', axis=0):
    """
    Proximity operator for joint sparsity. l2-norm on the given axis.

    :param alpha: coefficients to be processed, matrix of size [M, N]
    :param th: threshold level, the size of th should be consistent with the size of the other axis of alpha,
                e.g. axis = 0, size(th) = N; axis = 1, size(th) = M
    :param mode: 'soft'-threshold or 'hard'-thresholding
    :return: coefficients after joint sparsity, l21-norm (after proximal operation)
    """
    import sys

    l2norm = LA.norm(alpha, axis=axis)
    alpha_l21 = np.copy(alpha)
    ind = (l2norm > sys.float_info.epsilon)
    if mode == 'soft':
        l2norm_th = soft(l2norm, th)
    elif mode == 'hard':
        l2norm_th = hard(l2norm, th)
    if axis == 0:
        # !Att: multiplication based on array broadcasting
        alpha_l21[:, ind] = np.multiply(l2norm_th[ind] / l2norm[ind], alpha[:, ind])
    elif axis == 1:
        # !Att: multiplication based on array broadcasting
        alpha_l21[ind] = np.multiply(l2norm_th[ind][:, np.newaxis] / l2norm[ind][:, np.newaxis], alpha[ind])
    return alpha_l21, l2norm_th.sum()


# Elliptical projection #
def proj_ellipse(y, alpha, pU, z0, epsilon, max_iter, min_iter, tol):
    """
    Elliptical projection solver

    :param y: data, complex vector [M]
    :param alpha: coefficients to be processed, complex vector [M]
    :param pU: preconditioning matrix, real vector [M]
    :param z0: initial guess
    :param epsilon: l2-ball epsilon
    :param max_iter: max iteration
    :param min_iter: min iteration
    :param tol: error tolerance, stopping criterion
    :return: coefficients after projection, real vector [M]
    """
    mu = 1. / (pU.max()**2)
    rel_err = 1.
    it = 0
    while (rel_err > tol and it < max_iter) or it < min_iter:
        grad = pU * (z0 - alpha)
        z = y + proj_sc(z0 - mu * grad - y, epsilon)
        rel_err = LA.norm(z - z0) / LA.norm(z)
        z0 = z
        it += 1
    return z
