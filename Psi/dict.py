"""
Created on 27 Feb 2018

@author: mjiang
ming.jiang@epfl.ch
"""

import numpy as np
import pywt


class SARA(object):
    """
    Class of SARA dictionary

    @requires: pywt package

    @note:
    To create a SARA object, using
            obj = SARA(basis, nlevel, Nx, Ny)
    where
        basis is a list of names of orthogonal wavelets, in addition, keyword 'self' represents the image itself,
            e.g. basis = ['db1', 'db2', 'self']
        nlevel is an integer (no smaller than 1) representing the level of the decomposition,
            e.g. nlevel = 2
        Nx, Ny are integers representing the image size (rows, cols)
            e.g. Nx = 16, Ny = 8 means an image of 16x8

    List of attributes:
    self.basis, self.lenbasis, self.nlevel, self.Nx, self.Ny, self.Psi, self.Psit, self.bookkeeping

    List of methods:
    saradec2, sararec2, coef2vec, vec2coef, power_method
    """
    
    def __init__(self, basis, nlevel, Nx, Ny):
        self.basis = basis
        self.lenbasis = len(basis)
        self.nlevel = nlevel
        self.Nx = Nx
        self.Ny = Ny
        self.saradec2()
        self.sararec2()
        self.bookkeeping = []
        
    def saradec2(self):
        """
        The decomposition operator self.Psit (function handle) of the SARA dictionary is obtained.
        Attention: Periodization mode is used in wavelet decomposition!

        @requires: pywt package
        """
        self.Psit = []
        for i in np.arange(self.lenbasis):
            f = 'lambda x :'
            if self.basis[i] == 'self':
                f = '%s x.flatten()' % f
            else:
                f = "%s pywt.wavedec2(x, '%s', 'periodization', %d)" % (f, self.basis[i], self.nlevel)
            self.Psit.append(eval(f))
            
    def sararec2(self):
        """
        The adjoint (reconstruction) operator self.Psi (function handle) of the SARA dictionary is obtained.
        Attention: Periodization mode is used in wavelet reconstruction!

        @requires: pywt package
        """
        self.Psi = []
        for i in np.arange(self.lenbasis):
            f = 'lambda x :'
            if self.basis[i] == 'self':
                f = '%s np.reshape(x, (%d, %d))' % (f, self.Nx, self.Ny)
            else:
                f = "%s pywt.waverec2(x, '%s', 'periodization')" % (f, self.basis[i])
            self.Psi.append(eval(f))
            
    def coef2vec(self, coef):
        """
        Convert wavelet coefficients to an array-type vector, inverse operation of vec2coef.

        The initial wavelet coefficients are stocked in a list as follows:
            [cAn, (cHm, cVn, cDn), ..., (cH1, cV1, cD1)],
        and each element is a 2D array.
        After the conversion, the returned vector is as follows:
        [cAn.flatten(), cHn.flatten(), cVn.flatten(), cDn.flatten(), ...,cH1.flatten(), cV1.flatten(), cD1.flatten()].
        """
        vec = []
        self.bookkeeping = []
        for ele in coef:
            if type(ele) == tuple:
                self.bookkeeping.append((np.shape(ele[0])))
                for wavcoef in ele:
                    vec = np.append(vec, wavcoef.flatten())
            else:  
                self.bookkeeping.append((np.shape(ele)))
                vec = np.append(vec, ele.flatten())
        self.bookkeeping.append((self.Nx, self.Ny))            
        return vec   
    
    def vec2coef(self, vec):
        """
        Convert an array-type vector to wavelet coefficients, inverse operation of coef2vec.

        The initial vector is stocked in a 1D array as follows:
        [cAn.flatten(), cHn.flatten(), cVn.flatten(), cDn.flatten(), ..., cH1.flatten(), cV1.flatten(), cD1.flatten()].
        After the conversion, the returned wavelet coefficient is in the form of the list as follows:
            [cAn, (cHm, cVn, cDn), ..., (cH1, cV1, cD1)],
        and each element is a 2D array. This list can be passed as the argument in pywt.waverec2.
        """
        ind = self.bookkeeping[0][0] * self.bookkeeping[0][1] 
        coef = [np.reshape(vec[:ind], self.bookkeeping[0])]
        for ele in self.bookkeeping[1:-1]:
            indnext = ele[0] * ele[1]
            coef.append((np.reshape(vec[ind:ind+indnext], ele), 
                         np.reshape(vec[ind+indnext:ind+2*indnext], ele), 
                         np.reshape(vec[ind+2*indnext:ind+3*indnext], ele)))
            ind += 3*indnext
            
        return coef
    
    def power_method(self, tol, max_iter, weights=1):
        """
        Built-in power method in SARA. Compute the spectral radius of Psi weighted by a matrix weights.

        Init val <- ||x||
        Loop
            x <- x / val
            x' <- Psi W W^T Psit x
            val <- ||x - x'|| / ||x||
        return val
        """
        x = np.random.randn(self.Nx, self.Ny)
        x = x/np.linalg.norm(x, 'fro')
        init_val = 1
        
        for _ in np.arange(max_iter):
            # Apply W Psit
            y = []
            for j in np.arange(self.lenbasis):
                if self.basis[j] == 'self':
                    y = np.append(y, self.Psit[j](x)/np.sqrt(self.lenbasis))
                else:
                    y = np.append(y, self.coef2vec(self.Psit[j](x))/np.sqrt(self.lenbasis))
            y *= weights
            
            # Apply Psi W
            y *= weights
            y = np.reshape(y, (self.lenbasis, int(np.size(y)/self.lenbasis)))
            
            x = np.zeros((self.Nx, self.Ny))
            for j in np.arange(self.lenbasis):
                if self.basis[j] == 'self':     
                    x += self.Psi[j](y[j])/np.sqrt(self.lenbasis)
                else:
                    x += self.Psi[j](self.vec2coef(y[j]))/np.sqrt(self.lenbasis)
                    
            val = np.linalg.norm(x, 'fro')
            rel_var = np.abs(val-init_val) / init_val
            if rel_var < tol:
                break
            init_val = val
            x /= val
            
        return val


class hyperSARA(SARA):
    """
    Class of HyperSARA dictionary

    @requires: pywt package

    @note:
    To create a SARA object, using
            obj = SARA(basis, nlevel, Nx, Ny, L)
    where
        basis is a list of names of orthogonal wavelets, in addition, keyword 'self' represents the image itself,
            e.g. basis = ['db1', 'db2', 'self']
        nlevel is an integer (no smaller than 1) representing the level of the decomposition,
            e.g. nlevel = 2
        Nx, Ny, L are integers representing the wide-band image size (rows, cols, bands)
            e.g. Nx = 16, Ny = 8, L = 4 means an cube of 16x8x4

    List of attributes:
    self.basis, self.lenbasis, self.nlevel, self.Nx, self.Ny, self.L,
    self.Psi, self.Psit, self.Psi3, self.Psit3, self.bookkeeping

    List of methods:
    saradec2, sararec2, coef2vec, vec2coef, power_method
    """

    def __init__(self, basis, nlevel, Nx, Ny, L):
        super(SARA, self).__init__(self, basis, nlevel, Nx, Ny)
        self.L = L
        self.Psi3 = lambda x: self.__rec3(x)
        self.Psit3 = lambda x: self.__dec3(x)

    def __rec3(self, alpha):
        """
        :param alpha: hyperSARA coefficients of size [P, L, Nx, Ny]
        :return: wide-band image of size [L, Nx, Ny]
        """
        x = np.zeros((self.L, self.Nx, self.Ny))
        for l in np.arange(self.L):
            for k in np.arange(self.lenbasis):
                if self.basis[k] == 'self':
                    tmp = self.Psi[k](alpha[k, l]) / np.sqrt(self.lenbasis)
                else:
                    tmp = self.Psi[k](self.vec2coef(alpha[k, l])) / np.sqrt(self.lenbasis)
                x[l] += tmp
        return x

    def __dec3(self, x):
        """
        :param x: wide-band image of size [L, Nx, Ny]
        :return: hyperSARA coefficients of size [P, L, Nx, Ny]
        """
        alpha = np.zeros((self.lenbasis, self.L, self.Nx * self.Ny))
        for k in np.arange(self.lenbasis):
            for l in np.arange(self.L):
                if self.basis[k] == 'self':
                    alpha[k, l] = self.Psit[k](x[l]) / np.sqrt(self.lenbasis)
                else:
                    alpha[k, l] = self.coef2vec(self.Psit[k](x[l])) / np.sqrt(self.lenbasis)
        return alpha

    def power_method(self, tol, max_iter, weights=1):
        """
        Overwrite SARA.power_method
        """
        x = np.random.randn(self.L, self.Nx, self.Ny)
        x = x / np.linalg.norm(x, 'fro')
        init_val = 1

        for _ in np.arange(max_iter):
            # Apply W Psit3
            y = self.Psit3(x)
            y *= weights

            # Apply Psi3 W
            y *= weights
            x = self.Psi3(y)

            val = np.linalg.norm(x, 'fro')
            rel_var = np.abs(val - init_val) / init_val
            if rel_var < tol:
                break
            init_val = val
            x /= val

        return val

