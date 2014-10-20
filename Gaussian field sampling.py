#!/usr/bin/env python

"""
AUTHOR : Rohit Tripathy

DATE : 10/16/2014

This is a program to generate realizations of a two dimensional Gaussian field.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def se_cov_func(x1, x2, s, l):
    """
    This is a function that takes in 2 2 dimensional vectors as input and computes the
    covariance matrix K for each pair of input variables. The squared exponential
    kernel is used and its variance and length scale is passed to the function
    as parameters
    
    Params:
    1. x1
    type : ndarray
    
    2. x2
    type : ndarray
    
    3. s
    type : scalar(float)
    
    4. l
    type : scalar(float)
    
    Returns:
    k
    type : scalar(float)
    """
    x1 = x1.reshape(2,1)
    x2 = x2.reshape(2,1)
    #Compute r
    r = x2 - x1
    r_mod = r[0][0] ** 2 + r[1][0] ** 2
    k = (s ** 2) * math.exp(- r_mod / (2 * l * l))
    return k

def cov_matrix(X):
    """
    This function takes in 2 arrays and computes the covariance matrix using the
    squared exponential kernel
    
    Params:
    1. x
    type : ndarray
    2. y
    type : ndarray
    
    Returns:
    k
    type : ndarray
    """
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in xrange(n):
        for j in xrange(i, n):
            K[j, i] = se_cov_func(X[i, :], X[j, :], 1., 0.06)
            K[i, j] = K[j, i]
    return K



def sample_random_field(X1, X2):
    """
    This function generates a realisation of a Gaussian Random field in two
    dimensions.
    
    Params:
    1. x1
    type : ndarray
    2. y1
    type : ndarray
    
    Returns :
    z
    type : ndrray
    """
    X = np.hstack([X1.flatten()[:, None], X2.flatten()[:, None]])
    n = X.shape[0]
    K = cov_matrix(X) + 1e-2
    import scipy.linalg
    w, V = scipy.linalg.eigh(K)
    I = np.argsort(w)[::-1]
    for i in xrange(10):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(X1, X2, V[:, I[i]].reshape(X1.shape),
                rstride = 1, cstride = 1, alpha = 0.4,cmap = plt.cm.jet)
        plt.xlabel('$X_1$')
        plt.ylabel('$X_2$')
        plt.savefig('eig_' + str(i).zfill(2) + '.png')
        plt.clf()
    quit()
    L = np.linalg.cholesky(K)
    z = np.random.randn(n, 1)
    y = np.dot(L, z)
    return y.reshape(X1.shape)
    
    
    
if __name__ == '__main__':
    x1 = np.linspace(0, 1, 32)
    x2 = np.linspace(0, 1, 32)
    X1, X2 = np.meshgrid(x1, x2)
    for i in xrange(10):
        Y = sample_random_field(X1, X2)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(X1, X2, Y, rstride = 1, cstride = 1, alpha = 0.4,cmap = plt.cm.jet)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig('sample_2d_' + str(i).zfill(2) + '.png')
        plt.clf()
    
