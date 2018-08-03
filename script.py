'''
Created on 11 Apr 2018

@author: mjiang
ming.jiang@epfl.ch
'''

import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import scipy.io as sciio
from scipy.sparse import coo_matrix
from pynufft.nufft import NUFFT_cpu
import time
import pylab

from tools.radio import *
from tools.maths import pow_method
from Psi.opt import *
from reduction.fouReduction import * 

###### input data control ##########
get_image = True
gen_uv = False

###### parameters associated to Fourier reduction #######
usingReduction = False

####### other parameters ##########
maskKernel = True           # economic G matrix, recommend to set True 

if get_image:
    print('Read a reference image')
    # im = fits.getdata('cluster.fits')
    im = fits.getdata('data/M31_64.fits')
    im = im.squeeze()
Nd = np.shape(im)
N = Nd[0]*Nd[1]
visibSize = 5*N
input_SNR = 40

# Get UV coverage
if gen_uv:
    print('uv coverage is being simulated')
    [u,v] = simulateCoverage(N, visibSize)          # u, v should be column vectors
    
    u = u[:,np.newaxis]
    v = v[:,np.newaxis]
    uv = np.hstack((u,v))
    fits.writeto('data/uv_FR.fits', uv, overwrite=True)
else:
    print('Read an available uv coverage file')
    uv = fits.getdata('data/uv_FR.fits')

########## non-uniform FFT control, using the package pynufft ##########
Jd = (8,8)              # neighbour size for nufft
osf = (2,2)             # oversampling factor for nufft 
Kd = (Nd[0]*osf[0], Nd[1]*osf[1])          # oversampled size for nufft           

st = NUFFT_cpu()
st.plan(uv,Nd,Kd,Jd)

Dr = st.sn          # scaling matrix

########### definition of some operators #############
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

A, At, G, Gt, Gm, Gmt, Phi, Phi_t, Phim, Phim_t, mask_G = operators(st)

if usingReduction:
    if maskKernel:
        PhitPhi = lambda x: Phim_t(Phim(x))
    else:
        PhitPhi = lambda x: Phi_t(Phi(x))

######## simulated data control ##########
(y, yn) = util_gen_input_data(im, G, A, input_SNR)

############## SARA dictionary control ################
wlt_basis = ['db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'self']        # wavelet basis to construct SARA dictionary
nlevel = 2              # wavelet decomposition level
sara = SARA(wlt_basis, nlevel, Nd[0], Nd[1])

############# l2 ball bound control, very important for the optimization with constraint formulation ##########
####################################### Example for different settings ########################################
######### l2_ball_definition='value', stopping_criterion='l2-ball-percentage', stop=1.01 ######################
######### l2_ball_definition='sigma', stopping_criterion='sigma', bound=2., stop=2. ###########################
######### l2_ball_definition='chi-percentile', stopping_criterion='chi-percentile', bound=0.99, stop=0.999 ####
l2ball = l2param(l2_ball_definition='sigma', stopping_criterion='sigma', bound=2, stop=2)
epsilon, epsilons = util_gen_l2_bounds(yn, input_SNR, l2ball)
print('l2 ball bound = '+str(epsilon))

############# Fourier reduction step ##########
if usingReduction:
    print('Fourier dimensionality reduction is active')
    fouRedparam=redparam(thresholdstrategy='percent', threshold=0.5, covmatfileexists=True)
    if maskKernel:
        R, Rt, ry, rG, rGt, rPhi, rPhit = fourierReduction(yn, Gm, Gmt, Phim, Phim_t, Nd, fouRedparam)
    else:
        R, Rt, ry, rG, rGt, rPhi, rPhit = fourierReduction(yn, G, Gt, Phi, Phi_t, Nd, fouRedparam)

########### optimization parameters control ###############
if usingReduction:
    nu2 = pow_method(rPhi, rPhit, Nd, 1e-6, 200)             # Spectral radius of the measurement operator
else:
    if maskKernel:
        nu2 = pow_method(Phim, Phim_t, Nd, 1e-6, 200)
    else:
        nu2 = pow_method(Phi, Phi_t, Nd, 1e-6, 200)
fbparam = optparam(nu1=1.0,nu2=nu2,gamma=1.e-3,tau=0.49,max_iter=500, \
                   use_reweight_steps=True, use_reweight_eps=False, reweight_begin=300, reweight_step=50, reweight_times=4, \
                   reweight_alpha=0.01, reweight_alpha_ff=0.5)


############ dirty image ###############
dirty = np.real(At(Gt(yn)))

############## run FB primal-dual algo ##################
print('Sparse recovery using Forward-Backward Primal-Dual')
if usingReduction:
    imrec, l1normIter, l2normIter, relerrorIter = forward_backward_primal_dual(ry, A, At, rG, rGt, mask_G, sara, epsilon, epsilons, fbparam)
else:
    if maskKernel:
        imrec, l1normIter, l2normIter, relerrorIter = forward_backward_primal_dual(yn, A, At, Gm, Gmt, mask_G, sara, epsilon, epsilons, fbparam)
    else:
        mask_G[:] = True 
        imrec, l1normIter, l2normIter, relerrorIter = forward_backward_primal_dual(yn, A, At, G, Gt, mask_G, sara, epsilon, epsilons, fbparam)

##### results ####
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
plt.plot(np.arange(np.size(l1normIter))+1, l1normIter)
plt.title('Evolution of l1 norm')

plt.figure()
plt.plot(np.arange(np.size(l2normIter))+1, l2normIter)
plt.title('Evolution of l2 norm')

plt.figure()
plt.plot(np.arange(np.size(relerrorIter))+1,relerrorIter)
plt.title('Evolution of relative error')

pylab.show()