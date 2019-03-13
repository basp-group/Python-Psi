"""
Created on 13.03.19

@author: mjiang
ming.jiang@epfl.ch
"""

import numpy as np
import astropy.io.fits as fits
from Psi.dict import *


im = fits.getdata('data/W28_256.fits')
imsize = im.shape
c = 2
unds = 1

cube = np.zeros((c, imsize[0], imsize[1]))
# SARA dictionary control #
wlt_basis = ['db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8',
             'self']        # wavelet basis to construct SARA dictionary
nlevel = 3              # wavelet decomposition level

sara = SARA(wlt_basis, nlevel, imsize[0], imsize[1])
hypersara = hyperSARA(wlt_basis, nlevel, imsize[0], imsize[1], c//unds)

weights = 2 * np.ones((len(wlt_basis), imsize[0] * imsize[1]))
nu = sara.power_method(1e-8, 200, weights.flatten())

nu1 = hypersara.power_method(1e-8, 200, weights)

print(nu1)