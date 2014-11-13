#!/usr/bin/env python

"""
AUTHOR : Rohit tripathy

DATE : 11/09/2014

The following is a program to generate samples from a 2-D
Gaussian Random field by using the Karhunen - Loeve
Expansion.

"""
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D


class Covariance(object):

    def __init__(self, num_dim):
        """
        num_dim
        """
        self.num_dim = num_dim

    def __call__(self, X1, X2=None):
        """
        Return covariance matrix.
        """
        if X2 is None:
            X2 = X1
        K = np.zeros((X1.shape[0], X2.shape[0]))
        for i in xrange(X1.shape[0]):
            for j in xrange(X2.shape[0]):
                K[i, j] = self.cov_func(X1[i, :], X2[j, :])
        return K

    def cov_func(self, x1, x2):
        """
        Return cov.
        """
        raise NotImplementedError('Implement me!')


class SECovariance(Covariance):

    def __init__(self, num_dim, s=1., ell=0.1):
        super(SECovariance, self).__init__(num_dim)
        self.s = s
        self.ell = ell

    def cov_func(self, x1, x2):
        r = x2 - x1
        r_mod = r[0] ** 2 + r[1] ** 2
        k = (self.s ** 2) * math.exp(- r_mod / (2 * self.ell ** 2))
        return k


class ExpCovariance(Covariance):

    def __init__(self, num_dim, s=1., ell=0.1):
        #super(ExpCovariance, self).__init__(num_dim)
        Covariance.__init__(self, num_dim)
        self.s = s
        self.ell = ell

    def cov_func(self, x1, x2):
        r = x2 - x1
        r_mod = np.sum(np.abs(r))
        k = (self.s ** 2) * math.exp(- r_mod / (self.ell))
        return k


class Karhunen_Loeve(object):
    """
    This a class which defines methods to represent a 2-D
    Gaussian by means of a Karhunen- Loeve expansion.
    
    Methods:
    __init__() -> Constructor
    se_cov_func() -> Defines the squared exponential covariance
                     function
    
    cov_matrix() -> Generates the covariance matrix
    
    eigen_decomposition() -> Obtain Eigen Values and Eigen Vectors
                             of the Covariance Matrix.
    
    kl_expansion() -> Repesentation of the Random field using
                      the KL Expansion
    
    visualize() -> Generate surface plots for the Random Field.
    """
    x = None
    y = None
    X = None
    Y = None
    s = None
    l = None
    K = None
    Lambda = None
    points = None
    V = None
    kl = None
    
    def __init__(self, x1, y1, s1, l1, num_xi=0.99, cov_func=SECovariance(2)):
        """
        Initialize an instance of the Karhunen_Loeve class.
        
        Params:
        x1 :: type : integer
        y1 :: type : integer
        s1 :: type : float
        l1 :: type : float
        """
        self.cov_func = cov_func
        self.num_xi = num_xi
        self.x = np.linspace(0, 1, x1)
        self.y = np.linspace(0, 1, y1)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.s = s1
        self.l = l1
        self.K = np.zeros(shape = (x1 * y1, x1 * y1))
        self.points = np.hstack([self.X.flatten()[:, None,], self.Y.flatten()[:, None]])
        self.K = self.cov_func(self.points)
        self.eigen_decomposition()
    
    def eigen_decomposition(self):
        """
        Generate the Eigen Values and Eigen Vectors of
        the Covariance Matrix.
        """
        w, V = linalg.eigh(self.K)
        c = w[::-1]
        if isinstance(self.num_xi, float):
            percent_energy = np.cumsum(c) /  np.sum(c)
            self.num_xi = np.arange(c.shape[0])[percent_energy < self.num_xi][-1] # num_xi changes
        self.Lambda = w[::-1][:self.num_xi]
        self.V = V[:, ::-1][:, :self.num_xi]

    def __call__(self, xi):
        """
        Evaluate the KLE at ``xi``.
        """
        assert xi.shape[0] <= self.Lambda.shape[0]
        r = xi.shape[0]
        num_xi = xi.shape[0]
        sqrt_lambda = np.sqrt(self.Lambda[:num_xi])
        Phi = self.V[:, :num_xi]
        return np.dot(Phi, xi * sqrt_lambda)

    def sample(self):
        """
        Samples from the random field defined by the KLE.
        """
        return self(np.random.randn(self.num_xi)).reshape(self.X.shape)

    def sample_and_plot(self):
        """
        Generate surface plots for the Random field.
        """
        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        ax.plot_surface(self.X, self.Y, self.sample(), cmap = plt.cm.jet, rstride = 2, cstride = 2, linewidth = 1)
        plt.show()

    def visualize_eigenfunction(self, i):
        """
        Visualizes eigenfunction i.
        """
        phi_i = self.V[:, i].reshape(self.X.shape)
        ax = plt.axes(projection = '3d')
        ax.plot_surface(self.X, self.Y, phi_i, cmap = plt.cm.jet, rstride = 2, cstride = 2, linewidth = 1)
        plt.show()
        

if __name__=='__main__':
    
    #define an instance of the Karhunen_Loeve class.
    k1 = Karhunen_Loeve(32, 32, 3., 0.1, cov_func=ExpCovariance(2))
    for i in xrange(10):
        k1.sample_and_plot()
