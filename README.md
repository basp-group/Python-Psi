# Python Psi
Python Psi package is a python version of sparse imaging tool, which benefits from Forward-Backward Primal-Dual optimization algorithm. It is initially desinged for radio interferometric imaging, but it can be plugged into other applications.

In the context of radio interferometry, the measurement operator is obtained by using non-uniform FFT. However, other measurement operators can also be compatible.

Python Psi package also contains the functionality of Fourier dimensionality reduction, aiming at reducing raw data by an approximation of SVD decomposition in the context of radio interferometry.

## Contents
1. [Getting Started](#star)
1. [Introduction of methods](#meth)
1. [Running the tests](#test)
1. [Authors](#auth)
1. [References](#ref)

<a name="star"></a>
## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Python 3.4 or later

Numpy 

Scipy

Astropy

[pynufft](https://github.com/jyhmiinlin/pynufft), require Python 3.4 or later
```
pip3 install pynufft
```

[PyWavelets](https://pywavelets.readthedocs.io/en/latest/#)

```
pip3 install PyWavelets
```

### Installing

Just download the package and save it on your computer

<a name="test"></a>
## Running the tests

- test file *testReduction.py* for Fourier dimensionality reduction is available, all useful data for the test are located in the subdosser *data/*
- test file *testPsi.py* is available, all useful data for the test are located in the subdosser *data/*
- *script.py* is the main script for users, where users can activate or not Fourier dimensionality reduction.

<a name='meth'></a>
## Introduction of methods
### For Fourier dimensionality reduction
Consider an inverse Fourier (e.g. interferometry and MRI) imaging problem:
> yn = \Phi x + n

yn is the raw data in the form of complex column vector.

\Phi is called measurement operator, associated to a non-uniform fft. More precisely, \Phi = GA = GFZ, where G is the interpolation kernel, F is the FFT and Z is the zero padding. Attention, Fourier components are not fully sampled in many applications, \Phi is therefore an under-determined operator.

x is the vectorized image (column vector).

n is the noise.

In order to reduce the dimenality of the raw data, an (linear) embedding operator is applied to both sides of the imaging equation:

> R(yn) = R(\Phi) x + R(n)

The basic requirement of the operator R is that it should conserve the statistical properties of the noise. Therefore, the covariance of the R should be quasi-diagonal. According to the article "[A Fourier dimensionality reduction model for big data interferometric imaging](https://arxiv.org/abs/1609.02097)", having demonstrated that F\Phi^t\PhiF^t is quasi-diagonal, authors propose that

R = SF\Phi^t,

where S is a selection operation to select significant singular values. 

### Solution of inverse problem (SARA)
Recall the inverse imaging problem:
> yn = \Phi x + n,

If using Fourier dimensionality reduction, the inverse problem is substituted by
> R(yn) = R(\Phi) x + R(n).

This problem is ill-posed since the sampling pattern is not complete. To solve this ill-posed problem, sparsity constraint is imposed. Consider a constraint analysis sparsity formulation:

> min ||\Psit x||_1 s.t. ||yn - \Phi x||_2 <= epsilons,

where \Psi is the sparse representation of x, which is a concatenation of orthogonal wavelets.

This convex optimization problem is solved by Forward-Backward Primal-Dual algorithm. Due to the issue of the biais caused by l_1 norm, a reweighted l_1 scheme to alleviate the biais is integrated in the algorithm.

<a name="auth"></a>

## Authors

* **Ming Jiang [ming.jiang@epfl.ch](mailto:ming.jiang@epfl.ch)**

<a name="ref"></a>
## References
- Forward-Backward Primal-Dual - *Initial work by Alexandru Onose et al.* - [Scalable splitting algorithms for big-data interferometric imaging in the SKA era](https://academic.oup.com/mnras/article/462/4/4314/2589458), Mon. Not. Roy. Astron. Soc., 462(4):4314-4335, 2016
- Fourier dimenonality reduction - *Initial work by Vijay Kartik et al.* - [A Fourier dimensionality reduction model for big data interferometric imaging](https://academic.oup.com/mnras/article/468/2/2382/3061880), Mon. Not. Roy. Astron. Soc., 480(2):2382-2400, 2017.
- HyperSARA - *Initial work by Abdullah Abdulaziz et al.* - [Wideband Super-resolution Imaging in Radio Interferometry via Low Rankness and Joint Average Sparsity Models (HyperSARA)](https://academic.oup.com/mnras/article/489/1/1230/5543954), Mon. Not. Roy. Astron. Soc., 489(1):1230-1248, Oct. 2019.
