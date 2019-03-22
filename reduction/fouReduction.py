"""
Created on Nov 24, 2017

@author: mjiang
ming.jiang@epfl.ch
"""

import numpy as np
import scipy.fftpack as scifft

from tools.radio import guessmatrix

FT2 = lambda x: scifft.fftshift(scifft.fft2(scifft.ifftshift(x)))
IFT2 = lambda x: scifft.fftshift(scifft.ifft2(scifft.ifftshift(x)))


class redparam(object):
    """
    Fourier reduction parameters
    """

    def __init__(self, thresholdstrategy, threshold, covmatfileexists=False,
                 covmatfilename='data/covmat.mtx', fastCov=True):
        self.thresholdstrategy = thresholdstrategy  # 'value', 'percent' or 'estimate'
        self.threshold = threshold
        # threshold value should be consistent with the thresholdstrategy, e.g. 1.0e-4 or 0.9(90%)
        self.covmatfileexists = covmatfileexists
        # whether covariance matrix is available. For the first run, this should be False.
        self.covmatfilename = covmatfilename  # covariance matrix file name
        self.fastCov = fastCov  # fast covariance matrix computation through psf
        self.paddingsize = None  # oversampled image size
        self.mask_G = None  # mask


def operatorR(x, Phi_t, Sigma, S):
    # the embeddding operator R = \sigma * S * F * Phi^T = \sigma * S * F * Dr^T * Z^T * F^-1 * G^T
    im = FT2(np.real(Phi_t(x)))
    x_red = Sigma.flatten() * im.flatten()[S]
    x_red = x_red.reshape((np.size(x_red), 1))      # make sure of a column vector
    return x_red


def operatorRt(x_red, imsize, Phi, Sigma, S):
    # the adjoint non-masked embeddding operator R^T = Phi * F^T * S^T * \sigma = G * F * Z * F^T * S^T * \sigma
    x = np.zeros(imsize).astype('complex').flatten()
    x_red = x_red.flatten() * Sigma.flatten()
    x[S] = x_red
    x = Phi(np.real(IFT2(x.reshape(imsize))))
    return x


def embedOperators(Phi, Phi_t, Sigma, S, imsize):
    # embeding operators r = F * Phi^T, r^T = Phi * F^T
    R = lambda x: operatorR(x, Phi_t, Sigma, S)
    Rt = lambda x: operatorRt(x, imsize, Phi, Sigma, S)
    return R, Rt


def operatorRPhi(x, Ipsf, Sigma, S, imsize):
    # New reduced measurement operator: Sigma * S * F * Ipsf
    tmp = FT2(Ipsf(x.reshape(imsize)))
    y = Sigma.flatten() * tmp.flatten()[S]
    y = y.reshape((np.size(y), 1))      # make sure of a column vector
    return y


def operatorRPhit(x, Ipsf, Sigma, S, imsize):
    # Adjoint of the new reduced measurement operator: Ipsf * F^T * S * Sigma
    tmp = np.zeros(imsize).astype('complex').flatten()
    tmp[S] = Sigma.flatten() * x.flatten()
    tmp = tmp.reshape(imsize)
    y = Ipsf(IFT2(tmp))
    return y


def operatorIpsf(x, A, At, H, paddingsize=None, M=None):
    """
    This function implements the operator Ipsf = Phi^T * Phi = At * H * A
    :param x: input image
    :param A: function handle of direct operator F * Z
    :param At: function handle of adjoint operator Z^T * F^T
    :param H: holographic matrix
    :param paddingsize: tuple of the oversampled image size, mandatory if M is not None
    :param M: mask of the values that have no contribution to the convolution
    :return: image convolved with psf
    """
    tmp = A(np.real(x))
    if M is not None:
        tmp = tmp[M, 0]
    tmp1 = H.dot(tmp)
    if M is not None:
        tmp2 = np.zeros((paddingsize[0]*paddingsize[1], 1)).astype('complex')
        tmp2[M, 0] = tmp1
        tmp1 = tmp2
    y = np.real(At(tmp1))
    return y


def fourierReduction(G, Gt, A, At, Nd, redparam):
    """
    Fourier reduction method
    Assuming
            yn = Phi x + n, where Phi = G * A = G * F * Z

            Phi is the measurement operator
            G is the interpolation kernel, which is a sparse matrix
            F is the FFT operation
            Z is the zero-padding operation

    Suppose an embedding operator R such that
            R(yn) = R(Phi) x + R(n), where R(yn) has much lower dimensionality than yn

    :param G: function handle of the interpolation kernel

    :param Gt: function handle of the adjoint interpolation kernel

    :param A: function handle of the oversampled Fourier transform: F * Z

    :param At: function handle of the downsampled inverser Fourier transform: Z^T * F^T

    :param Nd: image size, tuple

    :param redparam: object of reduction parameters, it contains:
            redparam.thresholdstrategy: strategy to reduce the data dimensionality
                            'value' or 1: singular values will be thresholded by a predefined value
                            'percent' or 2: keep only the largest singular values according to a predefined percentage
                            'estimate' or 3: estimated from dirty image: gamma * sigma_noise / ||dirty||_2, gamma is
                            given as threshold level.

            redparam.threshold: if thresholdstrategy is 'value', threshold is a predefined value
                      if thresholdstrategy is 'percent', threshold is a percentage
                      if thresholdstrategy is 'estimate', threshold is a value of gamma

            redparam.covmatfileexists: flag used to read available covariance matrix
                                        associated to the operator F * Phi^T

    :return: Ipsf, function handle of psf
             S, selection vector for the singular values selection, bool
             d12, reduced inverse singular values
             FIpsf, function handle of F * Ipsf
             FIpsf_t, function handle of Ipsf * F^T
    """

    import astropy.io.fits as fits

    if not hasattr(redparam, 'thresholdstrategy'):
        redparam.thresholdstrategy = 'percent'
        print('Default reduction strategy is set to ' + redparam.thresholdstrategy)
    if redparam.thresholdstrategy == 'estimate' or redparam.thresholdstrategy == 3:
        if hasattr(redparam, 'x2'):
            im2 = redparam.x2
        elif hasattr(redparam, 'dirty2'):
            im2 = redparam.dirty2
        else:
            raise Exception('Groud trouth image or dirty image is missing for the estimation of the threshold')
        if not hasattr(redparam, 'sigma_noise') and not hasattr(redparam, 'noise'):
            raise Exception('Standard deviation of noise or noise matrix is missing '
                            'for the estimation of the threshold')
    if not hasattr(redparam, 'threshold'):
        redparam.threshold = 0.5
        print('Default reduction level is set to ' + str(redparam.threshold * 100) + '%')
    if not hasattr(redparam, 'covmatfileexists'):
        redparam.covmatfileexists = False
        print('Default covariance matrix availability is set to ' + str(redparam.covmatfileexists))
    if not hasattr(redparam, 'fastCov'):
        redparam.fastCov = True
        print('The fast covariance matrix computation is set to ' + str(redparam.fastCov))
    covmatfileexists = redparam.covmatfileexists
    thresholdstrategy = redparam.thresholdstrategy
    threshold = redparam.threshold
    fastCov = redparam.fastCov

    N = Nd[0] * Nd[1]

    H = Gt.dot(G)  # holographic matrix
    Ipsf = lambda x: operatorIpsf(x, A, At, H, redparam.paddingsize, redparam.mask_G)

    ### Parameter estimation  ###
    if covmatfileexists:
        d = fits.getdata(redparam.covmatfilename)
    else:
        if fastCov:
            dirac2D = np.zeros(Nd)
            dirac2D[Nd[0] // 2, Nd[1] // 2] = 1
            PSF = Ipsf(dirac2D)
            covariancemat = np.abs(np.real(FT2(PSF)).flatten())     # real part of F * Ipsf
            d = covariancemat.flatten()
        else:
            covoperator = lambda x: FT2(Ipsf(IFT2(x.reshape(Nd))))  # F * Phi^T * Phi * F^-1
            # As the embeding operator R = \sigma * F * Phi^T, when applied to the noise n,
            # it is necessary to compute the covariance of R * R^T so as to study the statistic of the new noise Rn
            covariancemat = np.abs(np.real(guessmatrix(covoperator, N, N, diagonly=True)))
            d = covariancemat.diagonal()
        fits.writeto(redparam.covmatfilename, d, overwrite=True)
    # Thresholding of singular values
    d1 = np.copy(d)
    if thresholdstrategy == 'value' or thresholdstrategy == 1:
        ind = d < threshold
        d1[ind] = 0
        print('The threshold is ' + str(threshold))
    if thresholdstrategy == 'percent' or thresholdstrategy == 2:
        dsort = np.sort(d)
        val = dsort[int(np.size(d) * (1. - threshold))]
        ind = d < val
        d1[ind] = 0
        print('Keep only ' + str(100 * threshold) + '% data')
    if thresholdstrategy == 'estimate' or thresholdstrategy == 3:
        if hasattr(redparam, 'sigma_nosie'):
            nb_vis = np.size(G, axis=0)
            noise = redparam.sigma_nosie / np.sqrt(2) * (nb_vis + 1j * nb_vis)
        else:
            noise = redparam.noise
        rn = FT2(At(Gt.dot(noise)))
        th = threshold * np.std(rn) / im2
        ind = d < th
        d1[ind] = 0
        print('The threshold is ' + str(th))

    # the mask S
    S = d1.astype('bool')
    d1 = d1[S]
    d1 = d1[:, np.newaxis]
    if not d1.all():
        d1[d1 < 1.e-10] = 1.e-10  # avoid division by zero
    d12 = 1. / np.sqrt(d1)

    FIpsf = lambda x: np.reshape(FT2(Ipsf(x)), newshape=(N, 1))  # F * Phi^T * Phi, image -> vect
    FIpsf_t = lambda x: Ipsf(IFT2(x.reshape(Nd)))  # Phi^T * Phi * F^T, vect -> image

    return Ipsf, S, d12, FIpsf, FIpsf_t

###########################################################################################################
###########################################################################################################
