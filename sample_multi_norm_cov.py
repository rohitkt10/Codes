#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import math as m

"""
AUTHOR: ROHIT TRIPATHY

DATE: 10/05/2014

The following program describes a function to draw a random sample
from a multivariate normal distribution of dimensionality d, and a
function to compute the covariance matrix for an n vector according
to the squared exponential kernel.

Two examples are also provided for sampling of data - a sample
drawn from a distribution of dimensionality d=2 and d=4.

Then a sample space of 100 vector points is computed and its
covariance function is computed according to the squared
exponential kernel.
A contour plot of the results are plotted.

Different random samples are plotted for the same value of s & l.
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
def cholesky(a, l):
    """
    This is a function to perform the Cholesky Decomposition of a given
    matrix A. The function takes in a square matrix as input and returns
    the lower diagonal matrix L.
    
    PRECONDITIONS:
    
    1.The matrix A must be a square matrix.
    2.The matrix A must be hermitian i.e. the matrix and its conjugate
    transpose must be equal.
    3. The matrix A must be positive definite i.e it should have all non-zero
    eigen values.
    """
    
    #First assert that the input matrix is a square matrix
    assert is_square(a) == True
    
    #Assert that the matrix is hermitian
    conj_tr = a.copy()
    conj_tr = np.mat(conj_tr)
    conj = conj_tr.H
    assert (conj == conj_tr).all()
    
    #Assert that the matrix is positive definite i.e. has all positive eigen
    #values
    assert is_positive_definite(a) == True
    
    n = len(a)
    #Initialize a nXn matrix
    #l = np.zeros(shape=(n, n))
    l[:] = 0.
    
    #Now compute the lower diagonal matrix for the Cholesky Decomposition
    for k in xrange(n):
        if(k == 0):
            l[0][0] = m.sqrt(a[0][0])
            for i in xrange(1, n):
                l[i][0] = a[i][0] / l[0][0]
        elif(k != n-1):
            sum = 0.0
            for j in xrange(k):
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
    
    :param mu:  The mean of the multivariate normal.
    :type mu:   :class:`numpy.ndarray`
    :param sigma:  The covariance matrix.
    :type sigma:   :class:`numpy.ndarray`
    :returns:   A sample from the multivaraite normal.
    :rtype:     :class:`numpy.ndarray`
    
    Example:
    >>> x = sample_multi_norm(mu, sigma)
    
    Here is a list of a few things:
        + One
        + Two
        
    .. pre::
        
        These are the preconditions.
        
    .. warning::
    
        Here is a warning.
        
    .. note::
    
        Here is a note.
        
    The samples are taken using the cholesky of :math:`\Sigma`:
    
    .. math::
    
        \Sigma = L L^T.
    """
    sigma = np.array(sigma)
    mu = np.array(mu)
    #Make necessary assertions about the covariance matrix
    assert is_square(sigma) == True
    #assert is_positive_definite(sigma) == True
    assert sigma.shape[0] == sigma.shape[1]
    assert sigma.shape[0] == mu.shape[0]
    n = sigma.shape[0]
    
    #Initialize the l matrix
    #l = np.zeros(shape = (n,n))
    l = np.ndarray((n, n))
    
    #Compute L matrix by performing Cholesky Decomposition of sigma
    cholesky(sigma, l)
    
    #Define the multivariate standard normal vector
    z = np.zeros(d)
    for i in range(d):
        z[i] = np.random.normal(0, 1)
    z = z.reshape(mu.shape)
    #z = np.mat(z)
    #z = z.transpose()
    
    #Draw a sample from X
    x = np.dot(l, z) + mu
    
    #Return the randomly drawn sample from the multivariate distribution
    return x

def se_cov_func(x, s, l):
    """
    This is a function that takes in an n-vector as input and computes the
    covariance matrix K for each pair of input variables. The squared exponential
    kernel is used and its variance and length scale is passed to the function
    as parameters
    """
    n = len(x)
    k = np.zeros(shape = (n, n))
    for i in xrange(n):
        for j in xrange(i, n):
            arg = -(((x[i] - x[j]) * (x[i] - x[j])) / (2 * l * l))
            k[i, j] = (s * s) * m.exp(arg)
            k[j, i] = k[i, j]
    return k

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
    print "\nA sample drawn from a multivariate distribution with dimensionality 4:\n"
    print x
    
    #Drawing a sample from a distribution of d = 100
    x = np.linspace(0, 100, 100)
    x = np.mat(x)
    x = x.transpose()
    s = 0.50
    l =1.5
    k = se_cov_func(x, s, l)
    mu = np.zeros(100)
    mu = np.mat(mu)
    mu = mu.transpose()
    
    #Plot 5 different random samples for the given value of s and l.
    for i in range(5):
        f = sample_multi_norm(mu, k, 100)
        plt.plot(x, f)
    plt.plot(label = '$l = 3.0,\ s = 1.5$')
    plt.xlabel("x vector")
    plt.ylabel("Random Samples")
    plt.title("Sample Plot")
    plt.text(30, 1.25, '5 samples with $l = 3.0,\ s = 1.5$', fontsize=15)
    #plt.show()
    plt.savefig('test1.png')
