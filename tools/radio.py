"""
Created on Nov 24, 2017

@author: mjiang
ming.jiang@epfl.ch
"""

import numpy as np
import scipy.fftpack as scifft


# Simulation parameters #
class sparam(object):
    """
    Simulation parameters
    """
    def __init__(self, N=256, p=0.5, hole_number=100, hole_prob=0.1, hole_size=np.pi/60,
                 fpartition=np.array([-np.pi, np.pi]), sigma=np.pi/4, sigma_holes=np.pi/3):
        self.N = N  # number of pixels
        self.p = p  # ratio of visibility to number of pixels
        self.hole_number = hole_number  # number of holes to introduce for "gaussian holes"
        self.hole_prob = hole_prob  # probability of single pixel hole for "gaussian + missing pixels"
        self.hole_size = hole_size  # size of the missing frequency data
        self.fpartition = fpartition  # partition (symetrically) of the data to nodes (frequency ranges)
        self.sigma = sigma  # variance of the gaussion over continous frequency
        self.sigma_holes = sigma_holes  # variance of the gaussion for the holes


# L2-ball parameters #
class l2param(object):
    """
    l2 ball parameters

    signature of the initialization
            l2param(l2_ball_definition, stopping_criterion, bound, stop)

    l2_ball_definition:
        1 - 'value'
        2 - 'sigma'
        3 - 'chi-percentile'
    stopping_criterion:
        1 - 'l2-ball-percentage'
        2 - 'sigma'
        3 - 'chi-percentile'
    bound, stop are paramters corresponding to l2_ball_definition and stopping_criterion respectively
    """
    
    def __init__(self, l2_ball_definition, stopping_criterion, bound=0.99, stop=1.0001):
        self.l2_ball_definition = l2_ball_definition
        self.stopping_criterion = stopping_criterion
        self.bound = bound
        self.stop = stop


def simulateCoverage(num_pxl, num_vis, pattern='gaussian', holes=True):
    """
    Simulate uv coverage
    """
    
    sprm = sparam()
    sprm.N = num_pxl
    sprm.p = np.ceil(np.double(num_vis)/num_pxl)        
    u, v = util_gen_sampling_pattern(pattern, holes, sprm)
         
    return u, v  


def util_gen_sampling_pattern(pattern, holes, sprm):
    """
    Generate a sampling pattern according to the given simulation parameter.
    This function is also valid to generate a sampling pattern with holes.

    :param pattern: 'gaussian', 'uniform' or 'geometric'
    :param holes: boolean, whether to generate holes in the pattern
    :param sprm: object of parameters for the simulation, more details to see the class "sparam"
    :return: (u,v) sampling pattern
    """
    
    from scipy.stats import norm 
    
    # Check simulation parameters
    if not hasattr(sprm, 'N'):
        sprm.N = 256
        print('Default the number of pixels is set to ' + str(sprm.N))
    if not hasattr(sprm, 'p'):
        sprm.p = 2
        print('Default ratio of visibility to number of pixels is set to ' + str(sprm.p))
    if not hasattr(sprm, 'fpartition'):
        sprm.fpartition = np.array([np.pi])
        print('Default symmetric partition is set to ' + str(sprm.fpartition))
    if not hasattr(sprm, 'hole_number'):
        sprm.hole_number = 10
        print('Default the number of holes is set to ' + str(sprm.hole_number))
    if not hasattr(sprm, 'sigma_holes'):
        sprm.sigma_holes = np.pi/2
        print('Default sigma of holes is set to ' + str(sprm.sigma_holes))
    if not hasattr(sprm, 'hole_size'):
        sprm.hole_size = np.pi/8
        print('Default the size of holes is set to ' + str(sprm.hole_size))
    if not hasattr(sprm, 'sigma'):
        sprm.sigma = np.pi/12
        print('Default sigma of the distribution is set to ' + str(sprm.sigma))
    
    Nh = sprm.hole_number
    sigma_h = sprm.sigma_holes
    hu = np.array([])
    hv = np.array([])
    
    # generate holes in the coverage
    if holes:
        print('Generate ' + str(Nh) + ' holes in the sampling pattern')
        while len(hu) < Nh:
            uv = -np.pi + 2 * np.pi * np.random.rand(2, 1)
            if norm.pdf(0, 0, sigma_h) * np.random.rand(1) > norm.pdf(np.linalg.norm(uv), 0, sigma_h):
                hu = np.append(hu, uv[0])
                hv = np.append(hv, uv[1])
        
    # generate points outside the holes
    sigma_m = sprm.sigma
    Nm = int(sprm.p * sprm.N)
    hs = sprm.hole_size
    print('Generate ' + str(Nm) + ' frequency points')
    
    u = np.array([])
    v = np.array([])           
       
    while len(u) < Nm:
        Nmextra = int(Nm - len(u))
        if pattern == 'gaussian':
            us = sigma_m * np.random.randn(int(1.1 * Nmextra))
            vs = sigma_m * np.random.randn(int(1.1 * Nmextra))
        elif pattern == 'uniform':
            us = -np.pi + 2 * np.pi * np.random.rand(int(1.1 * Nmextra))
            vs = -np.pi + 2 * np.pi * np.random.rand(int(1.1 * Nmextra))
        elif pattern == 'geometric':
            us = np.random.geometric(sigma_m, int(1.1 * Nmextra))
            vs = np.random.geometric(sigma_m, int(1.1 * Nmextra))
        # discard points outside (-pi,pi)x(-pi,pi)
        sf1 = np.where((us < np.pi) & (us > -np.pi))[0]
        sf2 = np.where((vs < np.pi) & (vs > -np.pi))[0]
        sf = list(set(sf1).intersection(sf2))

        if holes:
            for k in np.arange(Nh):
                # discard points inside the holes
                sf1 = np.where((us[sf] < hu[k] + hs) & (us[sf] > hu[k] - hs))[0]
                sf2 = np.where((vs[sf] < hu[k] + hs) & (vs[sf] > hu[k] - hs))[0]
                sfh = list(set(sf1).intersection(sf2))
                sf = list(set(sf) - set(sfh))

        sf = sf[:Nmextra]
        if np.size(sf) > 0: 
            u = np.append(u, us[sf])
            v = np.append(v, vs[sf])
        print(str(len(u)) + ' of ' + str(Nm))
         
    print('Sampling pattern is done!')
    return u, v


def util_gen_input_data(im, G, A, input_snr):
    """
    Generate simulated data

    :param im: reference image
    :param G: convolutional kernel, matrix
    :param A: function handle of the direct operator
    :param input_snr: SNR in dB
    :return: y, Ideal observation of the reference image (using non-uniform FFT)
            yn, noisy observation
            noise, additional Gaussian noise
            sigma_noise: standard deviation of the noise
    """
    
    import scipy.linalg as LA
    
    Av = A(im)              # Av = F * Z * Dr * Im
    y = G.dot(Av.reshape(np.size(Av), 1))            # y = G * Av
    N = np.size(y)
    
    sigma_noise = 10**(-input_snr/20.) * LA.norm(y)/np.sqrt(N)
    n = (np.random.randn(N, 1) + 1j * np.random.randn(N, 1)) * sigma_noise / np.sqrt(2)
    
    yn = y + n
    
    return y, yn, n, sigma_noise


def util_gen_l2_bounds(y, sigma_noise, param):
    """
    Generate l2 ball bound and l2 ball stopping criterion

    :param y: input data, complex vector
    :param sigma_noise: standard deviation of the input noise, defined by
            snr(dB) = 20 * log10(sqrt(mean (s^2)) / std(noise))
    :param param: object of l2 ball parameters, more details to see the class "l2param"
    :return: l2 ball bound, l2 ball stopping criterion
    """
    
    import scipy.linalg as LA
    from scipy.stats import chi2

    N = np.size(y)
    
    if param.l2_ball_definition == 'value' or param.l2_ball_definition == 1:
        epsilon = LA.norm(np.sqrt(np.size(y))) * sigma_noise
        
    if param.l2_ball_definition == 'sigma' or param.l2_ball_definition == 2:
        s1 = param.bound        
        epsilon = np.sqrt(N + s1*np.sqrt(2*N)) * sigma_noise
    
    if param.l2_ball_definition == 'chi-percentile' or param.l2_ball_definition == 3:
        p1 = param.bound
        # isf(p, N) is inverse of survival function of chi2 distribution with N degrees of freedom 
        # to compute the minimum x so that the probability is no more than p
        epsilon = np.sqrt(chi2.isf(1-p1, N)) * sigma_noise 
        
    if param.stopping_criterion == 'l2-ball-percentage' or param.stopping_criterion == 1:
        epsilons = epsilon * param.stop
    
    if param.stopping_criterion == 'sigma' or param.stopping_criterion == 2:
        s2 = param.stop
        epsilons = np.sqrt(N + s2*np.sqrt(2*N)) * sigma_noise
        
    if param.stopping_criterion == 'chi-percentile' or param.stopping_criterion == 3:
        p2 = param.stop
        epsilons = np.sqrt(chi2.isf(1-p2, N)) * sigma_noise         
        
    return epsilon, epsilons
        

def operatorA(im, st, imsize, paddingsize):
    """
    This function implements the operator A = F * Z * Dr

    :param im: input with imsize
    :param st: structure of nufft
    :param imsize: tuple of the image size
    :param paddingsize: tuple of the padding size
    :return: output with the padding size, vector
    """
    
    im = im.reshape(imsize)
    im = st.sn * im.astype('complex')           # Scaling (Dr)
    IM = scifft.fft2(im, paddingsize)     # Oversampled Fourier transform
    IM = IM.reshape(np.size(IM), 1)
    return IM


def operatorAt(IMC, st, imsize, paddingsize):
    """
    This function implements the adjoint operator At = Dr'Z'F'

    :param IMC: input with the padding size, vector
    :param st: structure of nufft
    :param imsize: tuple of the image size
    :param paddingsize: tuple of the oversampled image size
    :return: output with the image size, matrix
    """
    
    imc = scifft.ifft2(IMC.reshape(paddingsize))     # F'
    im = st.sn.conj() * imc[:imsize[0], :imsize[1]]         # Dr'Z'F'
    return im


def operatorPhi(im, G, A, M=None):
    """
    This function implements the operator Phi = G * A

    :param im: input image
    :param G: convolution kernel, matrix
    :param A: function handle of direct operator F * Z
    :param M: mask of the values that have no contribution to the convolution
    :return: visibilities, column complex vector
    """
    spec = A(im)
    if M is not None:
        spec = spec[M]
    vis = G.dot(spec)
    return vis


def operatorPhit(vis, Gt, At, paddingsize=None, M=None):
    """
    This function implements the operator Phi*T = A^T * G^T

    :param vis: input visibilities
    :param Gt: adjoint convolution kernel, matrix
    :param At: function handle of adjoint operator Z^T * F^T
    :param paddingsize: tuple of the oversampled image size, mandatory if M is not None
    :param M: mask of the values that have no contribution to the convolution
    :return: image, real matrix
    """
    protospec = Gt.dot(vis)
    if M is not None:
        protospec1 = np.zeros((paddingsize[0]*paddingsize[1], 1)).astype('complex')
        protospec1[M] = protospec
        protospec = protospec1
    im = np.real(At(protospec))
    return im


def operatorPhi3(im, G, A, M=None):
    """
    This function implements the operator Phi = G * A

    :param im: input image cube [L, Nx, Ny]
    :param G: list of convolution kernels
    :param A: function handle of direct operator F * Z
    :param M: list of masks of the values that have no contribution to the convolution
    :return: visibilities, column complex matrix [L, M]
    """
    L = len(G)
    vis = []
    for i in np.arange(L):
        spec = A(im[i])
        if M is not None:
            spec = spec[M[i]]
        tmp_vis = G[i].dot(spec)
        vis.append(tmp_vis.flatten())
    vis = np.array(vis)
    return vis


def operatorPhit3(vis, Gt, At, paddingsize=None, M=None):
    """
    This function implements the operator Phi*T = A^T * G^T

    :param vis: input visibilities [L, N]
    :param Gt: list of adjoint convolution kernels
    :param At: function handle of adjoint operator Z^T * F^T
    :param paddingsize: tuple of the oversampled image size, mandatory if M is not None
    :param M: list of masks of the values that have no contribution to the convolution
    :return: image, real cube [L, Nx, Ny]
    """
    L = len(Gt)
    im = []
    for i in np.arange(L):
        protospec = Gt[i].dot(vis[i][:, np.newaxis])
        if M is not None:
            protospec1 = np.zeros((paddingsize[0]*paddingsize[1], 1)).astype('complex')
            protospec1[M[i]] = protospec
            protospec = protospec1
        tmp_im = np.real(At(protospec))
        im.append(tmp_im)
    im = np.array(im)
    return im


def operators(st):
    kernel = st.sp
    imsize = st.Nd
    Kd = st.Kd
    
#     mask_G = np.any(kernel.toarray(),axis=0)        # mask of the values that have no contribution to the convolution 
#     kernel_m = kernel[:,mask_G]                   # convolution kernel after masking non-contributing values 

    G = kernel.tocsr()
    Gt = G.conj().T
    # faster computation and more economic storage
    mask_G = np.array((np.sum(np.abs(G), axis=0))).squeeze().astype('bool')
    Gm = kernel.tocsc()[:, mask_G]
    Gmt = Gm.conj().T
    
    np.abs(kernel).sign()
    
    A = lambda x: operatorA(x, st, imsize, Kd)          # direct operator: F * Z * Dr
    At = lambda x: operatorAt(x, st, imsize, Kd)           # adjoint operator: Dr^T * Z^T * F^T

    Phi = lambda x: operatorPhi(x, G, A)                # measurement operator: Phi = G * A
    Phi_t = lambda x: operatorPhit(x, Gt, At)
    
    Phim = lambda x: operatorPhi(x, Gm, A, mask_G)         # masked measurement operator: Phim = Gm * A
    Phim_t = lambda x: operatorPhit(x, Gmt, At, Kd, mask_G)
    
    return A, At, G, Gt, Gm, Gmt, Phi, Phi_t, Phim, Phim_t, mask_G


def guessmatrix(operator, M, N, diagonly=True):
    """
    Compute the covariance matrix by applying a given operator (F*Phi^T*Phi) on different delta functions
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse import csc_matrix

    if diagonly:
        maxnonzeros = min(M, N)
        operdiag = np.zeros(maxnonzeros, dtype='complex')
    else:
        matrix = csc_matrix((M, N))             # economic storage
        
    for i in np.arange(N):
        deltacol = coo_matrix(([1], ([i], [0])), shape=(N, 1))
        currcol = operator(deltacol.toarray()).flatten()
        if diagonly:
            if i > maxnonzeros:
                break
            operdiag[i] = currcol[i]
        else:
            matrix[:, i] = currcol[:, np.newaxis]
            
    if diagonly:
        matrix = coo_matrix((operdiag, (np.arange(maxnonzeros), np.arange(maxnonzeros))), shape=(M, N))
    
    return matrix


def util_natWeight(sigma_noise, M):

    from scipy.sparse import coo_matrix

    if np.size(sigma_noise) == 1:
        nW = 1./sigma_noise
    elif np.size(sigma_noise) != M:
        raise Exception('Dimension error of the natural weighting matrix ')
    else:
        nW = coo_matrix((sigma_noise, (np.arange(M), np.arange(M))), shape=(M, M))

    return nW


def generate_cube(im, f, emission=True):

    import matplotlib.pyplot as plt

    im = (im + np.abs(im)) / 2
    im /= im.max()
    Nx, Ny = im.shape

    # Generate sources
    col = np.array([22, 3, 192, 180, 191, 218, 148, 173, 81, 141, 104])
    row = np.array([154, 128, 97, 77, 183, 174, 207, 188, 210, 139, 137])
    r = np.array([18, 47, 30, 53, 21, 48, 15, 40, 17, 35, 71])

    nS = np.size(col)
    S = np.zeros((nS + 1, Nx * Ny))

    foreground = np.zeros((Nx, Ny))

    for i in np.arange(nS):
        [X, Y] = np.meshgrid(np.arange(Nx) - row[i], np.arange(Ny) - col[i])
        mask = (X**2 + Y**2) < r[i]**2
        mask = mask.astype('bool')
        if i == 9:
            mask9 = ~mask
        if i == 11:
            mask10 = mask
            mask = mask9 & mask10
        s = im * mask
        s[s < 5.e-3] = 0
        S[i] = s.flatten()
        foreground += s

    # Background source
    background = im - foreground
    background[background < 0] = 0
    S[-1] = background.flatten()

    nS += 1

    # Build spectra
    alpha = np.array([0.3, 0.3, 0.5, 0.1, 0.3, 0.5, 0.1, 0.3, 0.5, 0.1, -10, 0.5])
    beta = np.array([6, 6, 6, 5, 5, 5, 4, 4, 4, 3, -10, 5])
    c = np.size(f)

    H = np.zeros((c, nS))
    HI = np.ones((c, nS))

    k = 2
    for i in np.arange(nS):
        HI[k, i] = 2
        HI[-k, i] = 1.5
        HI[-k-1, i] = 2
        HI[-k-2, i] = 1.5
        HI[-k-3, i] = 2
        HI[-k-4, i] = 1.5
        k += 3

    plt.figure()

    for i in [10, 0, 1, 2, 4, 5, 6, 7, 9, 11]:
        H[:, i] = (f/f[0]) ** (-alpha[i] + beta[i] * np.log(f/f[0]))
        if emission:
            H[:, i] *= HI[:, i]
        plt.plot(H[:, i])

    plt.show()

    # Simulation of data cube
    X0 = H.dot(S)
    X0[X0 < 0] = 0

    x0 = np.reshape(X0, (c, Nx, Ny))

    return x0, X0


def util_gen_preconditioning_matrix(u, v, imsize, uniform_weight_sub_pixels=1):
    Nox = imsize[0]
    Noy = imsize[1]
    aWw = np.ones((np.size(u), 1))

    uedges = np.linspace(-np.pi, np.pi, uniform_weight_sub_pixels * Nox + 1)
    vedges = np.linspace(-np.pi, np.pi, uniform_weight_sub_pixels * Noy + 1)

    h, _, _, = np.histogram2d(u, v, [uedges, vedges])
    histu = np.sum(h, axis=1).astype('int')   # histogram of u
    histv = np.sum(h, axis=0).astype('int')   # histogram of v
    argu = np.argsort(u)        # indices of ascending u
    argv = np.argsort(v)        # indices of ascending v

    indu = 0
    for nbu in histu:
        indu1 = indu + nbu
        setu = set(argu[indu:indu1])    # indices of current bin of u
        indv = 0
        indu = indu1
        if not setu:        # empty set
            continue
        for nbv in histv:
            indv1 = indv + nbv
            setv = set(argv[indv:indv1])    # indices of current bin of v
            ind = list(setu.intersection(setv))     # common index in both bins of u and v
            indv = indv1
            if not setv:        # empty set
                continue
            aWw[ind] = len(ind)
    return 1./aWw
