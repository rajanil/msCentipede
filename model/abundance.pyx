
# base libraries
import pdb

# numerical libraries
import numpy as np
cimport numpy as np
from numpy cimport ndarray
from scipy.special import gammaln, digamma, polygamma

# custom libraries
import optimizer
import utils

"""
cdef extern from "abundance.h":
    void alpha_function( double* x, double* f, double* left, double* total, double* zeta, double* tau, const double zetasum, long N, long R, long J )
    void alpha_gradient( double* x, double* Df, double* left, double* total, double* zeta, double* tau, const double zetasum, long N, long R, long J )
    void alpha_hessian( double* x, double* Hf, double* left, double* total, double* zeta, double* tau, const double zetasum, long N, long R, long J )
"""

cdef class Alpha:
    """
    Class to store and update (M-step) the parameter `alpha` in negative 
    binomial part of the msCentipede model. There is a separate parameter
    for bound and unbound states, for each replicate.

    Arguments
        R : int
        number of replicate measurements

    """

    def __cinit__(self, long R):

        self.R = R
        self.value = np.random.rand(self.R,2)*10

    cdef update(self, zeta, Omega omega):
        """Update the estimates of parameter `alpha` in the model.
        """

        cdef ndarray zetaestim, constant, xo, G, h, x_final

        # auxillary variables
        zetasum = np.sum(zeta.value)

        # initialize optimization variables
        xo = self.value.ravel()

        # set constraints for optimization variables
        G = np.diag(-1 * np.ones((2*self.R,), dtype=float))
        h = np.zeros((2*self.R,1), dtype=float)

        args = dict([('G',G),('h',h),('omega',omega),('zeta',zeta),('zetasum',zetasum)])

        # call optimizer
        x_final = optimizer.optimize(xo, self.function_gradient, self.function_gradient_hessian, args)
        self.value = x_final.reshape(self.R,2)

    cdef tuple function_gradient(self, ndarray[np.float64_t, ndim=1] x, dict args):
        """Computes the likelihood function (only terms that contain 
            `alpha`), and its gradient
        """

        cdef long r
        cdef double f, zetasum
        cdef Omega omega
        cdef ndarray Df, xzeta

        zeta = args['zeta']
        omega = args['omega']
        zetasum = args['zetasum']

        f = 0
        Df = np.zeros((2*omega.R,), dtype='float')

        for r from 0 <= r < omega.R:
            xzeta = zeta.total[:,r:r+1] + x[2*r:2*r+2]
            f = f + np.sum(np.sum(gammaln(xzeta) * zeta.value, 0) - \
                gammaln(x[2*r:2*r+2]) * zetasum + \
                zetasum * utils.nplog(omega.value[r]) * x[2*r:2*r+2])
            Df[2*r:2*r+2] = np.sum(digamma(xzeta) * zeta.value, 0) - \
                            digamma(x[2*r:2*r+2]) * zetasum + \
                            zetasum * utils.nplog(omega.value[r])

        return f, Df

    cdef tuple function_gradient_hessian(self, ndarray[np.float64_t, ndim=1] x, dict args):
        """Computes part of the likelihood function that has
        terms containing `alpha`, and its gradient and hessian
        """

        cdef long r
        cdef double f, zetasum
        cdef Omega omega
        cdef ndarray Df, Hf, xzeta

        zeta = args['zeta']
        omega = args['omega']
        zetasum = args['zetasum']
        
        f = 0
        Df = np.zeros((2*omega.R,), dtype='float')
        Hf = np.zeros((2*omega.R,), dtype='float')

        for r from 0 <= r < omega.R:
            xzeta = zeta.total[:,r:r+1] + x[2*r:2*r+2]
            f = f + np.sum(np.sum(gammaln(xzeta) * zeta.value, 0) - \
                gammaln(x[2*r:2*r+2]) * zetasum + \
                zetasum * utils.nplog(omega.value[r]) * x[2*r:2*r+2])
            Df[2*r:2*r+2] = np.sum(digamma(xzeta) * zeta.value, 0) - \
                            digamma(x[2*r:2*r+2]) * zetasum + \
                            zetasum * utils.nplog(omega.value[r])
            Hf[2*r:2*r+2] = np.sum(polygamma(1, xzeta) * zeta.value, 0) \
                - polygamma(1, x[2*r:2*r+2]) * zetasum
       
        Hf = np.diag(Hf)

        return f, Df, Hf


cdef class Omega:
    """
    Class to store and update (M-step) the parameter `omega` in negative 
    binomial part of the msCentipede model. There is a separate parameter
    for bound and unbound states, for each replicate.

    Arguments
        R : int
        number of replicate measurements

    """

    def __cinit__(self, long R):

        self.R = R
        self.value = np.random.rand(self.R,2)
        self.value[:,1] = self.value[:,1]/100

    cdef update(self, zeta, Alpha alpha):
        """Update the estimates of parameter `omega` in the model.
        """

        cdef long r
        cdef ndarray numerator, denominator, value

        numerator = np.sum(zeta.value,0) * alpha.value
        denominator = np.array([np.sum(zeta.value * (value + zeta.total[:,r:r+1]), 0) \
            for r,value in enumerate(alpha.value)])
        self.value = numerator / denominator
