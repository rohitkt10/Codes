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

def cov_matrix(x, y):
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
    x = x.flatten()
    y = y.flatten()
    n = len(x)
    x = np.mat(x)
    y = np.mat(y)
    x = np.transpose(x)
    y = np.transpose(y)
    p = np.hstack((x, y))
    k = np.zeros(shape = (n, n))
    count = 0
    for i in xrange(n):
        for j in xrange(n):
            k[j, i] = se_cov_func(p[i, :], p[j, :], 0.5, 100)
    return k
def sample_random_field(x1, y1):
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
    n = len(x1)
    k = cov_matrix(x1, y1)
    z = np.zeros( shape = (n, n))
    for i in xrange(n):
        for j in xrange(n):
            z[i, j] = np.random.normal(0, k[i, j])
    return z
    
    
    
if __name__ == '__main__':
    x1 = np.linspace(0, 1, 30)
    x2 = np.linspace(0, 1, 30)
    x, y = np.meshgrid(x1, x2)
    z = np.zeros(shape = (10, 10))
    z = sample_random_field(x1, x2)
    """
    #compute the covariance matrix
    for i in xrange(32):
        for j xrange(32):
            p1 = [[x[i, j], y[i, j]]]
            p2 = 
            cov = cov_matrix([x[i, j], y[i, j]])
    cov = cov_matrix()
    """
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, z, rstride = 1, cstride = 1, alpha = 0.4,cmap = plt.cm.jet)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    