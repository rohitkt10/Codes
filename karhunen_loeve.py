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

class Karhunen_Loeve:
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
    
    def __init__(self, x1, y1, s1, l1):
        """
        Initialize an instance of the Karhunen_Loeve class.
        
        Params:
        x1 :: type : integer
        y1 :: type : integer
        s1 :: type : float
        l1 :: type : float
        """
        self.x = np.linspace(0, 1, x1)
        self.y = np.linspace(0, 1, y1)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.X = self.X.flatten()
        self.Y = self.Y.flatten()
        self.s = s1
        self.l = l1
        self.K = np.zeros(shape = (x1 * y1, x1 * y1))
        self.points = np.zeros(shape = (x1 * y1, 2))
        self.points[:, 0] = self.X
        self.points[:, 1] = self.Y
        self.Lambda = np.zeros(x1 * y1)
        self.V = np.zeros(shape = (x1 * y1, x1 * y1))
        self.kl = np.zeros(len(self.Lambda))
        self.cov_matrix()
        self.eigen_decomposition()
        self.kl_expansion()

    def se_cov_func(self, x1, x2):
        """
        Define the squared exponential kernel.
        
        Params:
        x1 :: type : ndarray
        x2 :: type : ndarray
        
        Returns:
        k  :: type : float
        """
        r = x2 - x1
        r_mod = r[0] ** 2 + r[1] ** 2
        k = (self.s ** 2) * math.exp(- r_mod / (2 * self.l * self.l))
        return k
    
    def cov_matrix(self):
        """
        Generate the covariance matrix.
        """
        for i in xrange(len(self.x) * len(self.y)):
            for j in xrange(i, len(self.x) * len(self.y)):
                self.K[i, j] = self.se_cov_func(self.points[i, :], self.points[j, :])
                self.K[j, i] = self.K[i, j]
    
    def eigen_decomposition(self):
        """
        Generate the Eigen Values and Eigen Vectors of
        the Covariance Matrix.
        """
        self.Lambda, self.V = linalg.eigh(self.K)
    
    def kl_expansion(self):
        """
        Use Karhunen Loeve Expansion to represent the Random field.
        The Gaussian field is assumed to be centered i.e.
        $\mu$ = 0
        """
        z = np.zeros(shape = (1, len(self.points)))
        z = np.random.randn(1, len(self.points))
        self.kl = np.zeros(len(self.points))
        sqrt_Lambda = np.sqrt(self.Lambda)
        for i in xrange(len(self.Lambda)):
            self.kl[i] = np.dot(z, sqrt_Lambda[i] * self.V[:, i])

    def visualize(self):
        """
        Generate surface plots for the Random field.
        """
        self.X = np.reshape(self.X, (len(self.x), len(self.y)))
        self.Y = np.reshape(self.Y, (len(self.x), len(self.y)))
        self.kl = np.reshape(self.kl, (len(self.x), len(self.y)))
        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        ax.plot_surface(self.X, self.Y, self.kl, cmap = plt.cm.jet, rstride = 2, cstride = 2, linewidth = 1)
        plt.show()
        

if __name__=='__main__':
    
    #define an instance of the Karhunen_Loeve class.
    k1 = Karhunen_Loeve(32, 32, 3., 0.06)
    k1.visualize()
