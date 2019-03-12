"""
Created on 5 Feb 2018

@author: mjiang
ming.jiang@epfl.ch
"""

import numpy as np
import scipy.linalg as LA

from Psi.dict import *
from Psi.proxTools import *
from tools.maths import *


# Optimization parameters #
class optparam(object):
    """
    Optimization parameters
    """
    def __init__(self, positivity=True, nu1=1., nu2=1., gamma=1.e-3, tau=0.49, rel_obj=1.e-3, max_iter=200,
                 lambda0=1., lambda1=1., lambda2=1., omega1=1., omega2=1., 
                 weights=1., global_stop_bound=True, use_reweight_steps=False, use_reweight_eps=False, 
                 reweight_begin=100, reweight_step=50, reweight_times=5, reweight_rel_obj=1.e-4,
                 reweight_min_steps_rel_obj=100, reweight_alpha=0.01, reweight_alpha_ff=0.5,
                 adapt_eps=False, adapt_eps_begin=50, adapt_eps_rel_obj=1.e-3, adapt_eps_step=100,
                 adapt_eps_tol_in=0.99, adapt_eps_tol_out=1.01, adapt_eps_change_percentage=0.5*(np.sqrt(5)-1),
                 mask_psf=False,
                 precond=False):
        
        self.positivity = positivity
        self.nu1 = nu1  # bound on the norm of the operator Psi
        self.nu2 = nu2  # bound on the norm of the operator A*G
        self.gamma = gamma  # convergence parameter L1 (soft th parameter)
        self.tau = tau  # forward descent step size
        self.rel_obj = rel_obj  # stopping criterion
        self.max_iter = max_iter  # max number of iterations
        self.lambda0 = lambda0  # relaxation step for primal update
        self.lambda1 = lambda1  # relaxation step for L1 dual update
        self.lambda2 = lambda2  # relaxation step for L2 dual update
#         self.sol_steps = [inf]  # saves images at the given iterations
        self.omega1 = omega1
        self.omega2 = omega2
        
        self.global_stop_bound = global_stop_bound  # global stopping bound for reweight scheme
        self.weights = weights  # weights matrix
        self.use_reweight_steps = use_reweight_steps  # reweighted scheme based on reweight steps
        self.use_reweight_eps = use_reweight_eps  # reweighted scheme based on the precision of the solution
        self.reweight_begin = reweight_begin  # the beginning of the reweighted scheme
        self.reweight_step = reweight_step  # number of iteration for each reweighted scheme
        self.reweight_times = reweight_times  # number of reweighted schemes
        self.reweight_rel_obj = reweight_rel_obj  # criterion for performing reweighting
        self.reweight_min_steps_rel_obj = reweight_min_steps_rel_obj 
        self.reweight_alpha = reweight_alpha  # coefficient in the reweight update function
        self.reweight_alpha_ff = reweight_alpha_ff  # factor to update the coefficient in the reweight update function

        self.adapt_eps = adapt_eps  # adaptive l2 bound
        self.adapt_eps_begin = adapt_eps_begin  # the beginning of the epsilon update
        self.adapt_eps_rel_obj = adapt_eps_rel_obj  # bound on the relative change of the solution
        self.adapt_eps_step = adapt_eps_step  # number of iteration for consecutive epslion updates
        self.adapt_eps_tol_in = adapt_eps_tol_in  # tolerance inside the l2 ball
        self.adapt_eps_tol_out = adapt_eps_tol_out  # tolerance outside the l2 ball
        self.adapt_eps_change_percentage = adapt_eps_change_percentage
        # the weight of the update w.r.t the l2 norm of the residual data

        self.mask_psf = mask_psf

        self.precond = precond


# Solve Non-Negative Least Squares using Forward-Backward #
def fb_nnls(y, A, At, param, FISTA=True):
    """"
    Solve non negative least squares problem:

    min  0.5*||y- A x||_2^2 s.t. x >= 0
    """
    if hasattr(param, 'initsol'):                # positivity constraint
        prev_sol = param.initsol
        prev_res = y - A(prev_sol)
        prev_epsilon = LA.norm(prev_res)
        grad = At(prev_res)
    else:
        prev_res = y
        prev_epsilon = LA.norm(prev_res)
        grad = At(prev_res)
        prev_sol = np.real(np.zeros_like(grad))

    xhat = prev_sol
    mu = 1./param.nu2
    t = 1

    for _ in np.arange(param.max_iter):

        sol = np.real(xhat + mu * grad)

        sol[sol < 0] = 0

        if FISTA:
            told = t
            t = (1. + np.sqrt(1. + 4 * told**2))/2.
            xhat = sol + (told - 1)/t * (sol - prev_sol)
        else:
            xhat = sol

        res = y - A(sol)
        epsilon = LA.norm(res)

        if epsilon == 0:
            rel_error = 1
        else:
            rel_error = LA.norm(epsilon - prev_epsilon)/prev_epsilon

        if rel_error < param.rel_obj:
            break

        grad = At(res)
        prev_sol = sol
        prev_epsilon = epsilon

    print('Non-Negative Least Squares is done')
    print('Residual norm is '+str(epsilon))
    return sol, epsilon


# Forward Backward Primal Dual method #
def forward_backward_primal_dual(y, A, At, G, Gt, mask_G, SARA, epsilon, epsilons, param,
                                 maskPsfMat=None, precondMat=None):
    """
    min ||Psit x||_1 s.t. ||y - \Phi x||_2 <= epsilons
    
    :param y: input data, complex vector of size [M]
    :param A: function handle, direct operator F * Z * Dr
    :param At: function handle, adjoint operator of A
    :param G: interpolation kernel, the kernel matrix is of size [M,No]
    :param Gt: adjoint interpolation kernel
    :param mask_G: mask of the values that have no contribution to the convolution, boolean vector of size [No],
        No is equal to the size of the oversampling
    :param SARA: object of sara dictionary
    :param epsilon: global l2 bound
    :param epsilons: stop criterion of the global L2 bound, slightly bigger than epsilon
    :param param: object of optimization parameters, more details to see the class "optparam"
    :param maskPsfMat: mask for the psf, boolean array of size [Nx, Ny], default is None
    :param precondMat: preconditioning matrix, real array of size [Nx, Ny], default is None
    :return: recovered image, real array of size [Nx, Ny]

    :TODO preconditioning
    """

    K = np.size(y)
    P = SARA.lenbasis 
    No = np.size(mask_G)        # total size of the oversampling
    
    Nx, Ny = SARA.Nx, SARA.Ny
    N = Nx * Ny
    
    if hasattr(param, 'positivity'):                # positivity constraint
        positivity = param.positivity
    else:
        positivity = True

    if hasattr(param, 'initsol'):                   # initialization of final solution x
        xsol = param.initsol
    else:
        xsol = np.zeros((Nx, Ny))

    if hasattr(param, 'adapt_eps'):                   # adaptive l2 bound
        adapt_eps = param.adapt_eps
    else:
        adapt_eps = False

    if hasattr(param, 'mask_psf'):                # mask of psf
        mask_psf = param.mask_psf
    else:
        mask_psf = False

    if hasattr(param, 'precond'):                # preconditioning
        precond = param.precond
    else:
        precond = False

    v1 = np.zeros((P, N))                           # initialization of L1 dual variable
    r1 = np.copy(v1)
    vy1 = np.copy(v1)
    u1 = np.zeros((P, Nx, Ny))
    norm1 = np.zeros(P)
    
    v2 = np.zeros((K, 1))                           # initialization of L2 dual variable
    
    # initial variables in the primal gradient step
    g1 = np.zeros_like(xsol)
    g2 = np.zeros_like(xsol)

    tau = param.tau             # step size for the primal
    if np.size(param.nu1) == 1:
        param.nu1 = np.ones(P) * param.nu1
    sigma1 = 1./param.nu1       # step size for the L1 dual update
    sigma2 = 1./param.nu2       # step size for the L2 dual update
    
    # Reweight scheme #
    weights = np.ones_like(v1)     # weight matrix
    reweight_alpha = param.reweight_alpha           # used for weight update
    reweight_alpha_ff = param.reweight_alpha_ff     # used for weight update
    
    omega1 = param.omega1          # omega sizes
    omega2 = param.omega2
    
    gamma = param.gamma         # threshold
    
    # relaxation parameters
    lambda0 = param.lambda0
    lambda1 = param.lambda1
    lambda2 = param.lambda2
    
    eps_it = 0
    reweight_step_count = 0
    reweight_last_step_iter = 0
    eps_over = 0
    eps_under = 0
    l1normIter = np.zeros(param.max_iter)
    l2normIter = np.zeros(param.max_iter)
    relerrorIter = np.zeros(param.max_iter)
    
    for it in np.arange(param.max_iter):
        
        # primal update #
        ysol = xsol - tau*(omega1 * g1 + omega2 * g2)
        if mask_psf:                        # mask invalid region of the image
            ysol = ysol[maskPsfMat]
        if positivity:
            ysol[ysol <= 0] = 0            # Positivity constraint. Att: min(-param.im0, 0) in the initial Matlab code!
        prev_xsol = np.copy(xsol)
        xsol += lambda0 * (ysol - xsol)
        
        # compute relative error
        norm_prevsol = LA.norm(prev_xsol)
        norm_sol = LA.norm(xsol - prev_xsol)
        if norm_prevsol == 0 or norm_sol == 0:
            rel_error = 1
        else:
            rel_error = norm_sol/norm_prevsol
        
        prev_xsol = 2*ysol - prev_xsol
        
        # L1 dual variable update #
        # Update for all bases, parallelable #
        for k in np.arange(P):
            if SARA.basis[k] == 'self':                           # processing for 'self' base, including normalization
                r1[k] = SARA.Psit[k](prev_xsol)/np.sqrt(P)
            else:                                                 # processing for other basis, including normalization
                r1[k] = SARA.coef2vec(SARA.Psit[k](prev_xsol))/np.sqrt(P)
            vy1[k] = v1[k] + r1[k] - soft(v1[k] + r1[k], gamma * weights[k] / sigma1[k])
            v1[k] = v1[k] + lambda1 * (vy1[k] - v1[k])
            if SARA.basis[k] == 'self':                           # processing for 'self' base, including normalization
                u1[k] = SARA.Psi[k](weights[k] * v1[k])/np.sqrt(P)
            else:                                                 # processing for other basis, including normalization
                u1[k] = SARA.Psi[k](SARA.vec2coef(weights[k] * v1[k]))/np.sqrt(P)
                
            norm1[k] = np.abs(r1[k]).sum()          # local L1 norm of current solution
                
        # L2 dual variable update #
        ns = A(prev_xsol)                   # non gridded measurements of current solution
        ns = ns[mask_G]
        
        r2 = G.dot(ns)
        vy2 = v2 + r2 - y - proj_sc(v2 + r2 - y, epsilon)
        v2 = v2 + lambda2 * (vy2 - v2)
        u2 = Gt.dot(v2)
                
        if np.abs(y).sum() == 0:
            u2[:] = 0
            r2[:] = 0

        norm2 = LA.norm(r2 - y)             # norm of residual
        
        # Adjust the l2 bound #
        if adapt_eps and it + 1 >= param.adapt_eps_begin and rel_error < param.adapt_eps_rel_obj \
                and it + 1 >= eps_it + param.adapt_eps_step:
            if norm2 < param.adapt_eps_tol_in * epsilon:                              # Current l2 ball over-estimated
                epsilon = norm2 + (1 - param.adapt_eps_change_percentage) * (epsilon - norm2)
                epsilons = param.adapt_eps_tol_out * epsilon                          # Update stopping epsilon
                eps_over += 1
                eps_it = it
                print('Epsilon under-estimate, update to ' + str(epsilon))
            elif norm2 > param.adapt_eps_tol_out * epsilon:                           # Current l2 ball under-estimated
                epsilon = epsilon + param.adapt_eps_change_percentage * (norm2 - epsilon)
                epsilons = param.adapt_eps_tol_out * epsilon                          # Update stopping epsilon
                eps_under += 1
                eps_it = it
                print('Epsilon over-estimate, update to ' + str(epsilon))

        # primal gradient update #
        g1 = np.zeros(np.shape(xsol))
        
        for k in np.arange(P):
            g1 += sigma1[k] * u1[k]
        
        uu = np.zeros((No, 1)).astype('complex')
        uu[mask_G] = u2    
        g2 = np.real(At(sigma2 * uu))
        
        l1normIter[it] = norm1.sum()
        l2normIter[it] = norm2
        relerrorIter[it] = rel_error
        
        print('Iteration: ' + str(it+1))
        print('Epsilon: ' + str(epsilon))
        print('L1 norm: ' + str(norm1.sum()))
        print('Residual: ' + str(norm2))
        print('Relative error: ' + str(rel_error))
        
        # weight update #
        if (param.use_reweight_steps and reweight_step_count < param.reweight_times
            and it+1-param.reweight_begin == param.reweight_step * reweight_step_count) or \
            (param.use_reweight_eps and norm2 <= epsilons
             and param.reweight_min_steps_rel_obj < it - reweight_last_step_iter
             and rel_error < param.reweight_rel_obj):
            # Update for all bases, parallelable #
            # \alpha =  weight = \alpha / (\alpha + |wt|)
            for k in np.arange(P): 
                if SARA.basis[k] == 'self': 
                    weights[k] = reweight_alpha / (reweight_alpha + abs(SARA.Psit[k](xsol)/np.sqrt(P)))
                else:              
                    weights[k] = reweight_alpha / (reweight_alpha + abs(SARA.coef2vec(SARA.Psit[k](xsol))/np.sqrt(P)))
    
            reweight_alpha = reweight_alpha_ff * reweight_alpha
            weights_vec = weights.flatten()
            # Compute optimal sigma1 according to the spectral radius of the operator Psi * W
            sigma1 = 1/SARA.power_method(1e-8, 200, weights_vec) * np.ones(P)
            reweight_step_count += 1
            reweight_last_step_iter = it
            print('Reweighted scheme number: '+str(reweight_step_count))
            
        # global stopping criteria #
        if rel_error < param.rel_obj and (norm2 <= param.adapt_eps_tol_out * epsilon) \
                and (norm2 >= param.adapt_eps_tol_in * epsilon):
            break
        
    xsol[xsol <= 0] = 0
    l1normIter = l1normIter[:it+1]
    l2normIter = l2normIter[:it+1]
    relerrorIter = relerrorIter[:it+1]

    if adapt_eps:
        print('Upward L2 ball bound '+str(eps_under)+' times')
        print('Downward L2 ball bound '+str(eps_over)+' times')

    return xsol, l1normIter, l2normIter, relerrorIter


def forward_backward_primal_dual_fouRed(y, d12, FIpsf, FIpsf_t, S, SARA, epsilon, epsilons, param,
                                        maskPsfMat=None, precondMat=None):
    """
    min ||Psit x||_1 s.t. ||y - \Phi x||_2 <= epsilons

    :param y: input data, complex vector of size [M]
    :param d12: inverse of singular values, vector of size [N0], N0 is the dimension after reduction
    :param FIpsf: function handle of F * Ipsf = F * Phi^t * Phi
    :param FIpsf_t: function handle of adjoint FIpsf
    :param S: selection vector to reduce the dimensionality, boolean vector of size [N],
    :param SARA: object of sara dictionary
    :param epsilon: global l2 bound
    :param epsilons: stop criterion of the global L2 bound, slightly bigger than epsilon
    :param param: object of optimization parameters, more details to see the class "optparam"
    :param maskPsfMat: mask for the psf, boolean array of size [Nx, Ny], default is None
    :param precondMat: preconditioning matrix, real array of size [Nx, Ny], default is None
    :return: recovered image, real array of size [Nx, Ny]

    :TODO preconditioning
    """

    K = np.size(y)
    P = SARA.lenbasis

    Nx, Ny = SARA.Nx, SARA.Ny
    N = Nx * Ny

    if hasattr(param, 'positivity'):  # positivity constraint
        positivity = param.positivity
    else:
        positivity = True

    if hasattr(param, 'initsol'):  # initialization of final solution x
        xsol = param.initsol
    else:
        xsol = np.zeros((Nx, Ny))

    if hasattr(param, 'adapt_eps'):  # adaptive l2 bound
        adapt_eps = param.adapt_eps
    else:
        adapt_eps = False

    if hasattr(param, 'mask_psf'):                # mask of psf
        mask_psf = param.mask_psf
    else:
        mask_psf = False

    if hasattr(param, 'precond'):                # preconditioning
        precond = param.precond
    else:
        precond = False

    v1 = np.zeros((P, N))  # initialization of L1 dual variable
    r1 = np.copy(v1)
    vy1 = np.copy(v1)
    u1 = np.zeros((P, Nx, Ny))
    norm1 = np.zeros(P)

    v2 = np.zeros((K, 1))  # initialization of L2 dual variable

    # initial variables in the primal gradient step
    g1 = np.zeros_like(xsol)
    g2 = np.zeros_like(xsol)

    tau = param.tau  # step size for the primal
    if np.size(param.nu1) == 1:
        param.nu1 = np.ones(P) * param.nu1
    sigma1 = 1. / param.nu1  # step size for the L1 dual update
    sigma2 = 1. / param.nu2  # step size for the L2 dual update

    # Reweight scheme #
    weights = np.ones_like(v1)  # weight matrix
    reweight_alpha = param.reweight_alpha  # used for weight update
    reweight_alpha_ff = param.reweight_alpha_ff  # # used for weight update

    omega1 = param.omega1  # omega sizes
    omega2 = param.omega2

    gamma = param.gamma  # threshold

    # relaxation parameters
    lambda0 = param.lambda0
    lambda1 = param.lambda1
    lambda2 = param.lambda2

    eps_it = 0
    reweight_step_count = 0
    reweight_last_step_iter = 0
    eps_over = 0
    eps_under = 0
    l1normIter = np.zeros(param.max_iter)
    l2normIter = np.zeros(param.max_iter)
    relerrorIter = np.zeros(param.max_iter)

    for it in np.arange(param.max_iter):

        # primal update #
        ysol = xsol - tau * (omega1 * g1 + omega2 * g2)
        if mask_psf:                        # mask invalid region of the image
            ysol = ysol[maskPsfMat]
        if positivity:
            ysol[ysol <= 0] = 0  # Positivity constraint. Att: min(-param.im0, 0) in the initial Matlab code!
        prev_xsol = np.copy(xsol)
        xsol += lambda0 * (ysol - xsol)

        # compute relative error
        norm_prevsol = LA.norm(prev_xsol)
        norm_sol = LA.norm(xsol - prev_xsol)
        if norm_prevsol == 0 or norm_sol == 0:
            rel_error = 1
        else:
            rel_error = norm_sol / norm_prevsol

        prev_xsol = 2 * ysol - prev_xsol

        # L1 dual variable update #
        # Update for all bases, parallelable #
        for k in np.arange(P):
            if SARA.basis[k] == 'self':  # processing for 'self' base, including normalization
                r1[k] = SARA.Psit[k](prev_xsol) / np.sqrt(P)
            else:  # processing for other basis, including normalization
                r1[k] = SARA.coef2vec(SARA.Psit[k](prev_xsol)) / np.sqrt(P)
            vy1[k] = v1[k] + r1[k] - soft(v1[k] + r1[k], gamma * weights[k] / sigma1[k])
            v1[k] = v1[k] + lambda1 * (vy1[k] - v1[k])
            if SARA.basis[k] == 'self':  # processing for 'self' base, including normalization
                u1[k] = SARA.Psi[k](weights[k] * v1[k]) / np.sqrt(P)
            else:  # processing for other basis, including normalization
                u1[k] = SARA.Psi[k](SARA.vec2coef(weights[k] * v1[k])) / np.sqrt(P)

            norm1[k] = np.abs(r1[k]).sum()  # local L1 norm of current solution

        # L2 dual variable update #
        ns = FIpsf(prev_xsol)[S]  # non gridded measurements of current solution

        r2 = d12 * ns
        vy2 = v2 + r2 - y - proj_sc(v2 + r2 - y, epsilon)
        v2 = v2 + lambda2 * (vy2 - v2)
        u2 = d12 * v2

        if np.abs(y).sum() == 0:
            u2[:] = 0
            r2[:] = 0

        norm2 = LA.norm(r2 - y)  # norm of residual

        # Adjust the l2 bound #
        if adapt_eps and it + 1 >= param.adapt_eps_begin and rel_error < param.adapt_eps_rel_obj \
                and it + 1 >= eps_it + param.adapt_eps_step:
            if norm2 < param.adapt_eps_tol_in * epsilon:  # Current l2 ball over-estimated
                epsilon = norm2 + (1 - param.adapt_eps_change_percentage) * (epsilon - norm2)
                epsilons = param.adapt_eps_tol_out * epsilon  # Update stopping epsilon
                eps_over += 1
                eps_it = it
                print('Epsilon under-estimate, update to ' + str(epsilon))
            elif norm2 > param.adapt_eps_tol_out * epsilon:  # Current l2 ball under-estimated
                epsilon = epsilon + param.adapt_eps_change_percentage * (norm2 - epsilon)
                epsilons = param.adapt_eps_tol_out * epsilon  # Update stopping epsilon
                eps_under += 1
                eps_it = it
                print('Epsilon over-estimate, update to ' + str(epsilon))

        # primal gradient update #
        g1 = np.zeros(np.shape(xsol))

        for k in np.arange(P):
            g1 += sigma1[k] * u1[k]

        uu = np.zeros((N, 1)).astype('complex')
        uu[S] = u2
        g2 = np.real(FIpsf_t(sigma2 * uu))

        l1normIter[it] = norm1.sum()
        l2normIter[it] = norm2
        relerrorIter[it] = rel_error

        print('Iteration: ' + str(it + 1))
        print('Epsilon: ' + str(epsilon))
        print('L1 norm: ' + str(norm1.sum()))
        print('Residual: ' + str(norm2))
        print('Relative error: ' + str(rel_error))

        # weight update #
        if (param.use_reweight_steps and reweight_step_count < param.reweight_times
            and it + 1 - param.reweight_begin == param.reweight_step * reweight_step_count) or \
            (param.use_reweight_eps and norm2 <= epsilons
             and param.reweight_min_steps_rel_obj < it - reweight_last_step_iter
             and rel_error < param.reweight_rel_obj):
            # Update for all bases, parallelable #
            # \alpha =  weight = \alpha / (\alpha + |wt|)
            for k in np.arange(P):
                if SARA.basis[k] == 'self':
                    weights[k] = reweight_alpha / (reweight_alpha + abs(SARA.Psit[k](xsol) / np.sqrt(P)))
                else:
                    weights[k] = reweight_alpha / (reweight_alpha + abs(SARA.coef2vec(SARA.Psit[k](xsol)) / np.sqrt(P)))

            reweight_alpha = reweight_alpha_ff * reweight_alpha
            weights_vec = weights.flatten()
            # Compute optimal sigma1 according to the spectral radius of the operator Psi * W
            sigma1 = 1 / SARA.power_method(1e-8, 200, weights_vec) * np.ones(P)
            reweight_step_count += 1
            reweight_last_step_iter = it
            print('Reweighted scheme number: ' + str(reweight_step_count))

        # global stopping criteria #
        if rel_error < param.rel_obj and (norm2 <= param.adapt_eps_tol_out * epsilon) and (
                norm2 >= param.adapt_eps_tol_in * epsilon):
            break

    xsol[xsol <= 0] = 0
    l1normIter = l1normIter[:it + 1]
    l2normIter = l2normIter[:it + 1]
    relerrorIter = relerrorIter[:it + 1]

    if adapt_eps:
        print('Upward L2 ball bound ' + str(eps_under) + ' times')
        print('Downward L2 ball bound ' + str(eps_over) + ' times')

    return xsol, l1normIter, l2normIter, relerrorIter
        
            
# Wide-band imaging based on primal-dual method #
def wide_band_primal_dual(y, A, At, G, Gt, mask_G, HyperSARA, epsilon, epsilons, param):
    """
    min ||Psit x||_2,1 + ||x||_* s.t. ||y - \Phi x||_2 <= epsilons

    :param y: input data, complex matrix of size [L, M], where columns represent bands
    :param A: function handle, direct operator F * Z * Dr
    :param At: function handle, adjoint operator of A
    :param G: wide-band interpolation kernel, cube of size [L, M, No]
    :param Gt: adjoint wide-band interpolation kernels
    :param mask_G: mask of the values that have no contribution to the convolution,
        boolean matrix of size [L, No] where each row is a vector of size [No], No is equal to the size of the
        oversampling
    :param HyperSARA: object of HyperSARA dictionary
    :param epsilon: vector of size [L] representing global l2 bound
    :param epsilons: vector of size [L] representing stop criterion of the global L2 bound, slightly bigger than epsilon
    :param param: object of optimization parameters, more details to see the class "optparam"
    :return: recovered image, real array of size [L, Nx, Ny]
    """

    K, L = np.shape(y)
    P = HyperSARA.lenbasis
    No = np.shape(mask_G)[0]        # total size of the oversampling

    Nx, Ny = HyperSARA.Nx, HyperSARA.Ny
    N = Nx * Ny

    if hasattr(param, 'positivity'):  # positivity constraint
        positivity = param.positivity
    else:
        positivity = True

    if hasattr(param, 'initsol'):  # initialization of final solution x
        xsol = param.initsol
    else:
        xsol = np.zeros((L, Nx, Ny))

    if hasattr(param, 'adapt_eps'):  # adaptive l2 bound
        adapt_eps = param.adapt_eps
    else:
        adapt_eps = False

    if hasattr(param, 'mask_psf'):                # mask of psf
        mask_psf = param.mask_psf
    else:
        mask_psf = False

    if hasattr(param, 'precond'):                # preconditioning
        precond = param.precond
    else:
        precond = False

    v0 = np.zeros((L, N))                           # initialization of nuclear norm dual variable
    v1 = np.zeros((P, L, N))                           # initialization of L21 dual variable

    v2 = np.zeros((L, K))                           # initialization of L2 dual variable
    r2 = np.copy(v2)
    vy2 = np.copy(v2)

    # initial variables in the primal gradient step
    g0 = np.zeros_like(xsol)
    g1 = np.zeros_like(xsol)
    g2 = np.zeros_like(xsol)

    sigma0 = 1./param.nu0       # step size for the nuclear norm dual update
    sigma1 = 1./param.nu1       # step size for the L21 dual update
    sigma2 = 1./param.nu2       # step size for the L2 dual update

    tau = 0.99/(sigma0*param.nu0 + sigma1*param.nu1 + sigma2*param.nu2)     # step size for the primal

    omega0 = sigma0          # omega sizes
    omega1 = sigma1
    omega2 = sigma2

    # Reweight scheme #
    weights0 = np.ones((L, 1))       # weights matrix for nuclear-norm term
    weights1 = np.ones_like(v1)     # weight matrix for l21-norm term
    reweight_alpha = param.reweight_alpha           # used for weight update
    reweight_alpha_ff = param.reweight_alpha_ff     # used for weight update

    gamma0 = param.gamma0  # threshold
    gamma = param.gamma         # threshold
    kappa0 = gamma0/sigma0
    kappa1 = gamma/sigma1

    # relaxation parameters
    lambda0 = param.lambda0
    lambda1 = param.lambda1
    lambda2 = param.lambda2
    lambda3 = param.lambda3

    reweight_step_count = 0
    reweight_last_step_iter = 0
    nuclearnormIter = np.zeros(param.max_iter)
    l21normIter = np.zeros(param.max_iter, P)
    l2normIter = np.zeros(param.max_iter)
    relerrorIter = np.zeros(param.max_iter)

    for it in np.arange(param.max_iter):

        # primal update #
        ysol = xsol - tau*(omega0 * g0 + omega1 * g1 + omega2 * g2)
        ysol[ysol <= 0] = 0              # Positivity constraint. Att: min(-param.im0, 0) in the initial Matlab code!
        prev_xsol = np.copy(xsol)
        xsol += lambda0 * (ysol - xsol)

        # compute relative error
        norm_prevsol = LA.norm(prev_xsol)
        if norm_prevsol == 0:
            rel_error = 1
        else:
            rel_error = LA.norm(xsol - prev_xsol)/norm_prevsol

        prev_xsol = 2*ysol - prev_xsol

        # Nuclear norm dual variable update #
        prev_xsol_mat = prev_xsol.reshape((L, N))
        tmp, s0 = nuclear_norm(v0 + prev_xsol_mat, kappa0 * weights0)
        nuclearnormIter[it] = np.abs(s0).sum()            # nuclear norm (l1-norm of the diagonal)
        v0 = v0 + lambda1 * (prev_xsol_mat - tmp)
        g0 = np.reshape(v0, (L, Nx, Ny))

        # L21 dual variable update #
        # Update for all bases, parallelable #
        r1 = HyperSARA.Psit3(prev_xsol)
        vy1, l21normIter[it] = l21_norm(v1 + r1, kappa1 * weights1)
        v1 = v1 + lambda2 * (r1 - vy1)
        g1 = sigma1 * HyperSARA.Psi3(v1)

        # L2 dual variable update #
        u2 = []
        for l in np.arange(L):
            ns = A(prev_xsol[l])                   # non gridded measurements of current solution
            ns = ns[mask_G[l]]

            r2[l] = G[l].dot(ns).flatten()
            vy2[l] = v2[l] + r2[l] - y[l] - proj_sc(v2[l] + r2[l] - y[l], epsilon)  # ! should be row vectors
            v2[l] = v2[l] + lambda3 * (vy2[l] - v2[l])
            u2.append(Gt[l].dot(v2[l, np.newaxis].T))

        if np.abs(y).sum() == 0:
            u2 = []
            r2[:] = 0

        norm2 = LA.norm(r2 - y)             # norm of residual

        # primal gradient update #
        for l in np.arange(L):
            uu = np.zeros((No, 1)).astype('complex')
            uu[mask_G[l]] = u2[l]
            g2[l] = np.real(At(sigma2 * uu))

        l2normIter[it] = norm2
        relerrorIter[it] = rel_error

        print('Iteration: ' + str(it+1))
        print('Nuclear norm: ' + str(nuclearnormIter[it]))
        print('L21 norm: ' + str(l21normIter[it]))
        print('Residual: ' + str(norm2))
        print('Relative error: ' + str(rel_error))

        # weight update #
        if (param.use_reweight_steps and reweight_step_count < param.reweight_times
            and it + 1 - param.reweight_begin == param.reweight_step * reweight_step_count) or \
                (param.use_reweight_eps and norm2 <= epsilons
                 and param.reweight_min_steps_rel_obj < it - reweight_last_step_iter
                 and rel_error < param.reweight_rel_obj):
            # Update for nuclear-norm weights #
            # \alpha =  weight = \alpha / (\alpha + |S|)
            weights0 = reweight_alpha / (reweight_alpha + np.abs(s0))
            # Update for all bases, parallelable #
            # \alpha =  weight = \alpha / (\alpha + |wt|)
            weights1 = reweight_alpha / (reweight_alpha + abs(HyperSARA.Psit3(xsol)))
            reweight_alpha = reweight_alpha_ff * reweight_alpha
            # Compute optimal sigma1 according to the spectral radius of the operator Psi * W
            sigma1 = 1 / HyperSARA.power_method(1e-8, 200, weights1)
            reweight_step_count += 1
            reweight_last_step_iter = it
            print('Reweighted scheme number: '+str(reweight_step_count))

        # global stopping criteria #
        if rel_error < param.reweight_rel_obj and ((param.global_stop_bound and (norm2 <= epsilon))
                                                   or (not param.global_stop_bound and norm2 <= epsilon)):
            break

    xsol[xsol <= 0] = 0
    l21normIter = l21normIter[:it+1]
    l2normIter = l2normIter[:it+1]
    relerrorIter = relerrorIter[:it+1]

    return xsol, l21normIter, l2normIter, relerrorIter
