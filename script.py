"""
Created on 11 Apr 2018

@author: mjiang
ming.jiang@epfl.ch
"""

import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from pynufft.nufft import NUFFT_cpu
import time
import pylab

from tools.radio import *
from tools.maths import pow_method
from Psi.opt import *
from reduction.fouReduction import *

# input data control #
get_image = True
gen_uv = True
natWeight = True

# parameters associated to Fourier reduction #
usingReduction = True

# Non-Negative Least Squares initialization #
nnls_init = True

# other parameters #
maskKernel = True  # economic G matrix, recommend to set True

if get_image:
    print('Read a reference image')
    # im = fits.getdata('cluster.fits')
    im = fits.getdata('data/M31_64.fits')
    im = im.squeeze()
Nd = np.shape(im)
N = Nd[0] * Nd[1]
visibSize = 5 * N
input_SNR = 40

# Get UV coverage
if gen_uv:
    print('uv coverage is being simulated')
    [u, v] = simulateCoverage(N, visibSize)  # u, v should be column vectors

    u = u[:, np.newaxis]
    v = v[:, np.newaxis]
    uv = np.hstack((u, v))
    fits.writeto('data/uv_FR.fits', uv, overwrite=True)
else:
    print('Read an available uv coverage file')
    uv = fits.getdata('data/uv_FR.fits')

# non-uniform FFT control, using the package pynufft #
Jd = (8, 8)  # neighbour size for nufft
osf = (2, 2)  # oversampling factor for nufft
Kd = (Nd[0] * osf[0], Nd[1] * osf[1])  # oversampled size for nufft

st = NUFFT_cpu()
st.plan(uv, Nd, Kd, Jd)

Dr = st.sn  # scaling matrix

# definition of some operators #
# Direct operator: A = F * Z * Dr
# Adjoint operator: A^T = Dr^T * Z^T * F^T
# Interpolation kernel G
# Adjoint interpolation kernel G^T
# Masked interpolation kernel Gm
# Masked adjoint interpolation kernel Gm^T
# Measurement operator Phi = G * A = G * F * Z * Dr
# Adjoint measurement operator Phi^t = A^T * G^T = Dr^T * Z^T * F^T * G^T
# Masked measurement operator Phim = Gm * A = Gm * F * Z * Dr
# Masked adjoint Measurement operator Phim^t = A^T * Gm^T = Dr^T * Z^T * F^T * Gm^T
# Mask of the values that have no contribution to the convolution mask_G

A, At, G, Gt, Gm, Gmt, _, _, _, _, mask_G = operators(st)

# simulated data control #
(y, yn, noise, sigma_noise) = util_gen_input_data(im, G, A, input_SNR)

# natural weighting #
if natWeight:
    nW = util_natWeight(sigma_noise, visibSize)
    if np.size(nW) == 1:
        G *= nW
        Gt *= nW
        Gm *= nW
        Gmt *= nW
        yn *= nW
        noise *= nW
    else:
        G = nW.dot(G)
        Gt = nW.dot(Gt)
        Gm = nW.dot(Gm)
        Gmt = nW.dot(Gmt)
        yn = nW.dot(yn)
        noise = nW.dot(noise)

Phi = lambda x: operatorPhi(x, G, A)  # measurement operator: Phi = G * A
Phi_t = lambda x: operatorPhit(x, Gt, At)
Phim = lambda x: operatorPhi(x, Gm, A, mask_G)  # masked measurement operator: Phim = Gm * A
Phim_t = lambda x: operatorPhit(x, Gmt, At, Kd, mask_G)

# SARA dictionary control #
wlt_basis = ['db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8',
             'self']  # wavelet basis to construct SARA dictionary
nlevel = 2  # wavelet decomposition level
sara = SARA(wlt_basis, nlevel, Nd[0], Nd[1])

# Fourier reduction step #
if usingReduction:
    print('Fourier dimensionality reduction is active')
    fouRedparam = redparam(thresholdstrategy='percent', threshold=.6, covmatfileexists=False,
                           covmatfilename='data/covmat.fits', fastCov=True)
    if maskKernel:
        fouRedparam.paddingsize = Kd
        fouRedparam.mask_G = mask_G
        Ipsf, S, d12, FIpsf, FIpsf_t = fourierReduction(Gm, Gmt, A, At, Nd, fouRedparam)
        # Reduced dataset Ry and reduced noise #
        ry = operatorR(yn, Phim_t, d12, S)
    else:
        Ipsf, S, d12, FIpsf, FIpsf_t = fourierReduction(G, Gt, A, At, Nd, fouRedparam)
        # Reduced dataset Ry and reduced noise #
        ry = operatorR(yn, Phi_t, d12, S)

# l2 ball bound control, very important for the optimization with constraint formulation #
"""
                Example for different settings
    l2_ball_definition='value', stopping_criterion='l2-ball-percentage', stop=1.01 
    l2_ball_definition='sigma', stopping_criterion='sigma', bound=2., stop=2. 
    l2_ball_definition='chi-percentile', stopping_criterion='chi-percentile', bound=0.99, stop=0.999 
"""
if not nnls_init:
    if usingReduction:
        if maskKernel:
            rn = operatorR(noise, Phim_t, d12, S)
        else:
            rn = operatorR(noise, Phi_t, d12, S)
        epsilon = LA.norm(rn)
        epsilons = 1.001 * epsilon
    else:
        l2ball = l2param(l2_ball_definition='sigma', stopping_criterion='sigma', bound=2, stop=2)
        if natWeight:
            epsilon, epsilons = util_gen_l2_bounds(yn, 1.0, l2ball)
        else:
            epsilon, epsilons = util_gen_l2_bounds(yn, sigma_noise, l2ball)
    print('epsilon='+str(epsilon))

# optimization parameters control #
if usingReduction:
    nu2 = pow_method(lambda x: operatorRPhi(x, Ipsf, d12, S, Nd), lambda x: operatorRPhit(x, Ipsf, d12, S, Nd), Nd,
                     1e-6, 200)  # Spectral radius of the measurement operator
else:
    if maskKernel:
        nu2 = pow_method(Phim, Phim_t, Nd, 1e-6, 200)
    else:
        nu2 = pow_method(Phi, Phi_t, Nd, 1e-6, 200)
print('nu2='+str(nu2))
fbparam = optparam(nu1=1.0, nu2=nu2, gamma=1.e-3, tau=0.49, max_iter=500, rel_obj=1.e-6,
                   use_reweight_steps=True, use_reweight_eps=False, reweight_begin=300, reweight_step=50,
                   reweight_times=4,
                   reweight_alpha=0.01, reweight_alpha_ff=0.5, reweight_rel_obj=1.e-6,
                   adapt_eps=nnls_init, adapt_eps_begin=100, adapt_eps_rel_obj=1.e-3,
                   mask_psf=False, precond=False)

# dirty image #
dirac2D = np.zeros(Nd)
dirac2D[Nd[0] // 2, Nd[1] // 2] = 1
PSF = Phi_t(Phi(dirac2D))
dirty = np.real(Phi_t(y)) / np.abs(PSF).max()

# run FB primal-dual algo #
print('Sparse recovery using Forward-Backward Primal-Dual')

# Initialization using NNLS #
if nnls_init:
    nnlsparam = optparam(nu2=nu2, max_iter=200, rel_obj=1.e-6)  # Initialization parameters control
    print('Initialization using Non-Negative Least Squares')

if usingReduction:
    if nnls_init:
        fbparam.initsol, epsilon = fb_nnls(ry, lambda x: operatorRPhi(x, Ipsf, d12, S, Nd),
                                           lambda x: operatorRPhit(x, Ipsf, d12, S, Nd), nnlsparam, FISTA=True)
        print('Initialization: ' + str(LA.norm(fbparam.initsol - im) / LA.norm(im)))
        print('Estimated epsilon via NNLS: ' + str(epsilon))
        epsilons = fbparam.adapt_eps_tol_out * epsilon
    imrec, l1normIter, l2normIter, relerrorIter = \
        forward_backward_primal_dual_fouRed(ry, d12, FIpsf, FIpsf_t, S, sara, epsilon, epsilons, fbparam)
else:
    if maskKernel:
        if nnls_init:
            fbparam.initsol, epsilon = fb_nnls(yn, Phim, Phim_t, nnlsparam, FISTA=True)
            print('Initialization: ' + str(LA.norm(fbparam.initsol - im) / LA.norm(im)))
            print('Estimated epsilon via NNLS: ' + str(epsilon))
            epsilons = fbparam.adapt_eps_tol_out * epsilon
        imrec, l1normIter, l2normIter, relerrorIter = \
            forward_backward_primal_dual(yn, A, At, Gm, Gmt, mask_G, sara, epsilon, epsilons, fbparam)
    else:
        if nnls_init:
            fbparam.initsol, epsilon = fb_nnls(yn, Phi, Phi_t, nnlsparam, FISTA=True)
            print('Initialization: ' + str(LA.norm(fbparam.initsol - im) / LA.norm(im)))
            print('Estimated epsilon via NNLS: ' + str(epsilon))
            epsilons = fbparam.adapt_eps_tol_out * epsilon
        mask_G[:] = True
        imrec, l1normIter, l2normIter, relerrorIter = \
            forward_backward_primal_dual(yn, A, At, G, Gt, mask_G, sara, epsilon, epsilons, fbparam)

print('Relative error of the reconstruction: ' + str(LA.norm(imrec - im) / LA.norm(im)))

# results #
plt.figure()
plt.imshow(im, cmap='gist_stern')
plt.title('Reference image')
plt.colorbar()

plt.figure()
plt.imshow(dirty, cmap='gist_stern')
plt.title('Dirty image')
plt.colorbar()

plt.figure()
plt.imshow(imrec, cmap='gist_stern')
plt.title('Reconstructed image')
plt.colorbar()

plt.figure()
plt.plot(np.arange(np.size(l1normIter)) + 1, l1normIter)
plt.title('Evolution of l1 norm')

plt.figure()
plt.plot(np.arange(np.size(l2normIter)) + 1, l2normIter)
plt.title('Evolution of l2 norm')

plt.figure()
plt.plot(np.arange(np.size(relerrorIter)) + 1, relerrorIter)
plt.title('Evolution of relative error')

pylab.show()
