#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import math as m

"""
AUTHOR: ROHIT TRIPATHY

DATE: 10/05/2014

The following program describes a function to draw a random sample
from a multivariate normal distribution of dimensionality d.

Two examples are also provided - a sample drawn from a distribution
of dimensionality d=2 and d=4
"""

#Define a function to check if a matrix is square or not.
def is_square(a):
    """
    This function returns a Boolean value True if the matrix A is square.
    Otherwise it returns false.
    """
    val = True
    b = a.T
    n1 = len(a)
    n2 = len(b)
    if(n1 == n2):
        return True
    else:
        return False

#Define a function to check if the input matrix is positive definite
def is_positive_definite(a):
    n = len(a)
    eigval, eigvect = np.linalg.eig(a)
    is_positive = True
    for i in range(n):
        if(eigval[i] <= 0):
            is_positive = False
            break
    return is_positive

    
#Define a function to compute the Cholesky Decomposition of a matrix
def cholesky(a):
    """
    This is a function to perform the Cholesky Decomposition of a given
    matrix A. The function takes in a square matrix as input and returns
    the lower diagonal matrix L.
    
    PRECONDITIONS:
    
    1.The matrix A must be a square matrix.
    2.The matrix A must be non singular i.e. it must have a non zero
    determinant.
    3. The matrix A must be positive definite i.e it should have all non-zero
    eigen values.
    """
    
    #First assert that the input matrix is a square matrix
    assert is_square(a) == True
    
    #Assert that the matrix is singular i.e. has non zero determinant
    deter = np.linalg.det(a)
    assert deter != 0
    
    #Assert that the matrix is positive definite i.e. has all positive eigen
    #values
    assert is_positive_definite(a) == True
    
    n = len(a)
    #Initialize a nXn matrix
    l = np.zeros(shape=(n, n))
    
    #Now compute the lower diagonal matrix for the Cholesky Decomposition
    for k in range(n):
        if(k == 0):
            l[0][0] = m.sqrt(a[0][0])
            for i in range(1, n):
                l[i][0] = a[i][0] / l[0][0]
        elif(k != n-1):
            sum = 0.0
            for j in range(k):
                sum += (l[k][j] * l[k][j])
            l[k][k] = m.sqrt(a[k][k] - sum)
            for i in range(k+1, n):
                sum = 0.0
                for j in range(k):
                    sum += l[i][j] * l[k][j]
                l[i][k] = (a[i][k] - sum) / l[k][k]
        else:
            sum = 0.0
            for j in range(k):
                sum += l[k][j] * l[k][j]
            l[k][k] = m.sqrt(a[k][k] - sum)
    return l

#define a function to draw samples from a multivariate Gaussian
def sample_multi_norm(mu, sigma, d=2):
    """
    This is a function to draw a random sample from a multivariate Gaussian
    given its mean matrix(mu) and covariance matrix(sigma).
    
    The principle involved is as follows:
    if z =[ z1 z2 ... zn] is a random vector of n random variables which are
    distributed according to the standard normal distribution N(0, 1), then
    the random vector given by X = LZ + mu is also a multivariate Gaussian
    distributed as N(mu, sigma) where sigma = L.Lt ( Lt = transpose of L).
    The matrix L can be computed by performing a Cholesky decomposition of the
    matrix sigma.
    The vector mu and the covariance matrix sigma are passed to the function as
    arguments.
    
    Then random samples can be draw from the multivariate normal distribution X.
    
    The argument d represents the dimensionality of the multivariate normal
    distribution. The default dimensionality is 2 i.e. bivariate.
    """
    
    n = len(sigma)
    m = len(mu)
    #Make necessary assertions about the covariance matrix
    assert is_square(sigma) == True
    assert is_positive_definite(sigma) == True
    assert n == d
    assert m == n
    
    #Initialize the l matrix
    l = np.zeros(shape = (n,n))
    
    #Compute L matrix by performing Cholesky Decomposition of sigma
    l = cholesky(sigma)
    
    #Define the multivariate standard normal vector
    z = np.zeros(d)
    for i in range(d):
        z[i] = np.random.normal(0, 1)
    z = np.mat(z)
    z = z.transpose()
    
    #Draw a sample from X
    x = np.dot(l, z) + mu
    
    #Return the randomly drawn sample from the multivariate distribution
    return x

if __name__ == '__main__':
    #A couple of examples
    #Sample drawm from a bivariate distribution
    mu = np.array([[2.5], [1.5]])
    cov = np.array([[0.5, 0], [0, 1.2]])
    x = sample_multi_norm(mu, cov, 2)
    print "\nA sample drawn from a multivariate distribution with dimensionality 2:\n"
    print x
    
    #Sample drawm from a distribution of d = 4
    mu = np.array([[2.5], [1.5], [2.5], [3.0]])
    cov = np.array([[0.5,  0, 0, 0], [0, 1.0, 0.0, 0.0], [0, 0, 1.3, 0], [0, 0, 0, 0.8]])
    x = sample_multi_norm(mu, cov, 4)
    print "\nA sample drawn from a multivariate distribution with dimensionality 2:\n"
    print x