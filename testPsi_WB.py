'''
Created on 20 Mar 2018

@author: mjiang
ming.jiang@epfl.ch
'''

import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import pylab

from Psi.dict import *
from Psi.opt import *
from Psi.proxTools import *
from tools.radio import *
from tools.maths import pow_method
from pynufft.nufft import NUFFT_cpu

get_image = True
gen_uv = True
natWeight = True

# Non-Negative Least Squares initialization #
nnls_init = True

# Preconditioning #
precond = True

c = 60              # Number of bands
unds = 4
f = np.linspace(1, 2, c)
samp_rate = 0.3

# image loading #
if get_image:
    im = fits.getdata('data/W28_256.fits')           # reference image
    im = im.squeeze()
    imsize = im.shape                           # image size
N = imsize[0] * imsize[1]
input_SNR = 40

x0, X0 = generate_cube(im, f, emission=True)            # size: x0[L, Nx, Ny], X0[L, N]

fits.writeto('x0.fits', x0, overwrite=True)

f = f[::unds]
x0 = x0[::unds]
X0 = X0[::unds]

# simulated sampling control #
pattern = 'gaussian'        # pattern of simulated sampling, 'gaussian', 'uniform' or 'geometric'
holes = True
simuparam = sparam()    # object of paramters of simulated sampling
simuparam.N = imsize[0] * imsize[1]
simuparam.p = samp_rate
[uw, vw] = util_gen_sampling_pattern(pattern, holes, simuparam)

# non-uniform FFT control, using the package pynufft #
Jd = (8, 8)              # neighbour size for nufft
osf = (2, 2)             # oversampling factor for nufft
Kd = (imsize[0]*osf[0], imsize[1]*osf[1])          # oversampled size for nufft

uw /= 2
vw /= 2

aW = []
G = []
Gt = []
Gm = []
Gmt = []
mask_G = []
y = []
yn = []
epsilon = []
epsilons = []
nu2_ch = np.zeros(c//unds)

if nnls_init:
    nnlsparam = optparam(max_iter=200, rel_obj=1.e-6)  # Initialization parameters control
    print('Initialization using Non-Negative Least Squares')
else:
    # l2 ball bound parameter #
    """
                        Example for different settings
        l2_ball_definition='value', stopping_criterion='l2-ball-percentage', stop=1.01 
        l2_ball_definition='sigma', stopping_criterion='sigma', bound=2., stop=2. 
        l2_ball_definition='chi-percentile', stopping_criterion='chi-percentile', bound=0.99, stop=0.999 
    """
    l2ball = l2param(l2_ball_definition='sigma', stopping_criterion='sigma', bound=2, stop=2)

# optimization parameters control #
fbparam = optparam(nu0=1.0, nu1=1.0, gamma0=1., gamma=1.e-2, max_iter=500, rel_obj=1.e-6,
                   use_reweight_steps=True, use_reweight_eps=False, reweight_begin=300, reweight_step=50,
                   reweight_times=4,
                   reweight_alpha=0.01, reweight_alpha_ff=0.5, reweight_rel_obj=1.e-6,
                   adapt_eps=nnls_init, adapt_eps_begin=100, adapt_eps_rel_obj=1.e-3,
                   mask_psf=False, precond=False)

for i in np.arange(c//unds):
    u = f[i]/f[0] * uw
    v = f[i]/f[0] * vw
    uv = np.hstack((u[:, np.newaxis], v[:, np.newaxis]))

    st = NUFFT_cpu()
    st.plan(uv, imsize, Kd, Jd)

    if precond:
        tmp_aW = util_gen_preconditioning_matrix(u, v, Kd)

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

    A, At, tmp_G, tmp_Gt, tmp_Gm, tmp_Gmt, _, _, _, _, tmp_mask_G = operators(st)

    # simulated data control #
    if gen_uv:
        tmp_y, tmp_yn, _, sigma_noise = util_gen_input_data(x0[i], tmp_G, A, input_SNR)

    # natural weighting #
    if natWeight:
        nW = util_natWeight(sigma_noise, np.size(tmp_yn))
        if np.size(nW) == 1:
            tmp_G *= nW
            tmp_Gt *= nW
            tmp_Gm *= nW
            tmp_Gmt *= nW
            tmp_yn *= nW
        else:
            tmp_G = nW.dot(tmp_G)
            tmp_Gt = nW.dot(tmp_Gt)
            tmp_Gm = nW.dot(tmp_Gm)
            tmp_Gmt = nW.dot(tmp_Gmt)
            tmp_yn = nW.dot(tmp_yn)

    # l2 ball bound control, very important for the optimization with constraint formulation #
    if nnls_init:
        Phim = lambda x: operatorPhi(x, tmp_Gm, A, tmp_mask_G)
        Phim_t = lambda x: operatorPhit(x, tmp_Gmt, At, Kd, tmp_mask_G)
        nu2_ch[i] = pow_method(Phim, Phim_t, imsize, 1e-6, 200, verbose=True)
        nnlsparam.nu2 = nu2_ch[i]
        _, tmp_epsilon = fb_nnls(tmp_yn, Phim, Phim_t, nnlsparam, FISTA=True)
        tmp_epsilons = fbparam.adapt_eps_tol_out * tmp_epsilon
        print('Channel ' + str(i) + ': estimated epsilon via NNLS: ' + str(tmp_epsilon))
    else:
        if natWeight:
            tmp_epsilon, tmp_epsilons = util_gen_l2_bounds(tmp_yn, 1.0, l2ball)
        else:
            tmp_epsilon, tmp_epsilons = util_gen_l2_bounds(tmp_yn, sigma_noise, l2ball)
        print('Channel ' + str(i) + ': estimated epsilon via NNLS: ' + str(tmp_epsilon))

    aW.append(tmp_aW.flatten())
    G.append(tmp_G)
    Gt.append(tmp_Gt)
    Gm.append(tmp_Gm)
    Gmt.append(tmp_Gmt)
    mask_G.append(tmp_mask_G)
    y.append(tmp_y.flatten())
    yn.append(tmp_yn.flatten())
    epsilon.append(tmp_epsilon)
    epsilons.append(tmp_epsilons)

if precond:
    aW = np.array(aW)       # aW of size [L, M], where L represents band and M represent visibility
else:
    aW = None
y = np.array(y)        # y and yn of size [L, M], where L represents band and M represent visibility
yn = np.array(yn)
mask_G = np.array(mask_G)
epsilon = np.array(epsilon)
epsilons = np.array(epsilons)
print('epsilon=', epsilon)

Phim3 = lambda x: operatorPhi3(x, Gm, A, mask_G)  # masked measurement operator: Phim = Gm * A
Phim_t3 = lambda x: operatorPhit3(x, Gmt, At, Kd, mask_G)

# SARA dictionary control #
wlt_basis = ['db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8',
             'self']        # wavelet basis to construct SARA dictionary
nlevel = 3              # wavelet decomposition level
hypersara = hyperSARA(wlt_basis, nlevel, imsize[0], imsize[1], c//unds)

if precond:
    nu2 = pow_method(lambda x: np.sqrt(aW) * Phim3(x), lambda x: Phim_t3(np.sqrt(aW) * x), (c//unds, imsize[0], imsize[1]), 1e-6, 200, verbose=True)
    fbparam.precond = True
    print('Preconditioning active, nu2=' + str(nu2))
else:
    nu2 = pow_method(Phim3, Phim_t3, (c//unds, imsize[0], imsize[1]), 1e-6, 200, verbose=True)             # Spectral radius of the measurement operator
    print('nu2='+str(nu2))
fbparam.nu2 = nu2

# run wide-band primal-dual algo #
imrec, nuclearnormIter, l21normIter, l2normIter, relerrorIter = \
    wide_band_primal_dual(yn, A, At, Gm, Gmt, mask_G, hypersara, epsilon, epsilons, fbparam, precondMat=aW)

snr_res = 20 * np.log10(LA.norm(x0)/LA.norm(imrec - x0))
print('Reconstruction snr=', snr_res)
fits.writeto('imrec.fits', imrec, overwrite=True)

# Results #
plt.figure()
plt.plot(np.arange(np.size(nuclearnormIter))+1, nuclearnormIter)
plt.title('Evolution of nuclear norm')

plt.figure()
plt.plot(np.arange(np.size(l21normIter))+1, l21normIter)
plt.title('Evolution of l21 norm')

plt.figure()
plt.plot(np.arange(np.size(l2normIter))+1, l2normIter)
plt.title('Evolution of l2 norm')

plt.figure()
plt.plot(np.arange(np.size(relerrorIter))+1, relerrorIter)
plt.title('Evolution of relative error')

pylab.show()
