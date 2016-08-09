
# base libraries
import pdb

# numerical libraries
import numpy as np
cimport numpy as np
from numpy cimport ndarray

# custom libraries
import optimizer
import utils

cdef extern from "footprint.h":
    void pi_function( double* x, double* f, double* left, double* total, double* zeta, double* tau, const double zetasum, long N, long R, long J )
    void pi_gradient( double* x, double* Df, double* left, double* total, double* zeta, double* tau, const double zetasum, long N, long R, long J )
    void pi_hessian( double* x, double* Hf, double* left, double* total, double* zeta, double* tau, const double zetasum, long N, long R, long J )
    void tau_function( double* x, double* f, double* left, double* total, double* zeta, double* pi, const double zetasum, long N, long R, long J )
    void tau_gradient( double* x, double* Df, double* left, double* total, double* zeta, double* pi, const double zetasum, long N, long R, long J )
    void tau_hessian( double* x, double* Hf, double* left, double* total, double* zeta, double* pi, const double zetasum, long N, long R, long J )

cdef class Pi:
    """
    Class to store and update (M-step) the parameter `p` in the
    msCentipede model. It is also used for the parameter `p_o` in
    the msCentipede_flexbg model.

    Arguments
        J : int
            number of scales

    """

    def __cinit__(self, long J):

        self.J = J
        self.L = 2**(self.J+1)-1

        # initializing parameters for each scale,
        # concatenated horizontally
        self.value = np.random.rand(self.L)

    cdef update(self, data, zeta, Tau tau):
        """Update the estimates of parameter in the model.
        """

        cdef dict args
        cdef double zetasum
        cdef ndarray xo, G, h

        # initialize optimization variable
        xo = self.value.copy()

        # set constraints for optimization variable
        G = np.vstack((np.diag(-1*np.ones((self.L,), dtype=float)), \
                np.diag(np.ones((self.L,), dtype=float))))
        h = np.vstack((np.zeros((self.L,1), dtype=float), \
                np.ones((self.L,1), dtype=float)))

        # auxillary variables
        zetasum = np.sum(zeta.value)

        # create dictionary of arguments
        args = dict([('G',G),('h',h),('data',data),('zeta',zeta),('tau',tau),('zetasum',zetasum)])

        # call optimizer
        self.value = optimizer.optimize(xo, self.function_gradient, self.function_gradient_hessian, args)

    cdef tuple function_gradient(self, ndarray[np.float64_t, ndim=1] x, dict args):

        """Computes the likelihood function (only terms that 
        contain `pi`) and its gradient
        """

        cdef long N, R, J
        cdef ndarray f, Df, left, total, zeta, tau

        N = args['data'].N
        R = args['data'].R
        J = args['data'].J
        left = args['data'].left
        total = args['data'].total
        zeta = args['zeta'].value
        tau = args['tau'].value
        zetasum = args['zetasum']

        f = np.array([0.])
        pi_function(<double*> x.data, <double*> f.data, \
                    <double*> left.data, <double*> total.data, \
                    <double*> zeta.data, <double*> tau.data, zetasum, \
                    N, R, J)

        Df = np.zeros((1,self.L), dtype=float)
        pi_gradient(<double*> x.data, <double*> Df.data, \
                    <double*> left.data, <double*> total.data, \
                    <double*> zeta.data, <double*> tau.data, zetasum, \
                    N, R, J)
        
        return f, Df

    cdef tuple function_gradient_hessian(self, ndarray[np.float64_t, ndim=1] x, dict args):

        """Computes the likelihood function (only terms that 
        contain `pi`), its gradient, and hessian
        """

        cdef long N, R, J
        cdef ndarray left, total, zeta, tau
        cdef ndarray f, Df, Hf

        N = args['data'].N
        R = args['data'].R
        J = args['data'].J
        left = args['data'].left
        total = args['data'].total
        zeta = args['zeta'].value
        tau = args['tau'].value
        zetasum = args['zetasum']

        f = np.array([0.])
        pi_function(<double*> x.data, <double*> f.data, \
                    <double*> left.data, <double*> total.data, \
                    <double*> zeta.data, <double*> tau.data, zetasum, \
                    N, R, J)

        Df = np.zeros((1,self.L), dtype=float)
        pi_gradient(<double*> x.data, <double*> Df.data, \
                    <double*> left.data, <double*> total.data, \
                    <double*> zeta.data, <double*> tau.data, zetasum, \
                    N, R, J)
        
        Hf = np.zeros((self.L, self.L), dtype=float)
        pi_hessian(<double*> x.data, <double*> Hf.data, \
                   <double*> left.data, <double*> total.data, \
                   <double*> zeta.data, <double*> tau.data, zetasum, \
                   N, R, J)

        return f, Df, Hf

cdef class Tau:
    """
    Class to store and update (M-step) the parameter `tau` in the
    msCentipede model. It is also used for the parameter `tau_o` in
    the msCentipede-flexbg model.

    Arguments
        J : int
        number of scales

    """

    def __cinit__(self, long J):

        self.J = J
        self.value = 10*np.random.rand(self.J)

    cdef update(self, data, zeta, Pi pi):
        """Update the estimates of the parameter in the model.
        """

        cdef dict args
        cdef double zetasum
        cdef ndarray xo, G, h

        # initialize optimization variables
        xo = self.value.copy()

        # set constraints for optimization variables
        G = np.diag(-1 * np.ones((self.J,), dtype=float))
        h = np.zeros((self.J,1), dtype=float)

        # auxillary variables
        zetasum = np.sum(zeta.value)

        # create args dictionary
        args = dict([('G',G),('h',h),('data',data),('zeta',zeta),('pi',pi),('zetasum',zetasum)])

        # call optimizer
        self.value = optimizer.optimize(xo, self.function_gradient, self.function_gradient_hessian, args)

    cdef tuple function_gradient(self, ndarray[np.float64_t, ndim=1] x, dict args):
        """Computes the likelihood function (only
        terms that contain `tau`) and its gradient.
        """

        cdef long N, R, J
        cdef ndarray left, total, zeta, pi
        cdef ndarray f, Df

        N = args['data'].N
        R = args['data'].R
        J = args['data'].J
        left = args['data'].left
        total = args['data'].total
        zeta = args['zeta'].value
        pi = args['pi'].value
        zetasum = args['zetasum']

        f = np.array([0.])
        tau_function(<double*> x.data, <double*> f.data, \
                    <double*> left.data, <double*> total.data, \
                    <double*> zeta.data, <double*> pi.data, zetasum, \
                    N, R, J)

        Df = np.zeros((1,self.J), dtype=float)
        tau_gradient(<double*> x.data, <double*> Df.data, \
                    <double*> left.data, <double*> total.data, \
                    <double*> zeta.data, <double*> pi.data, zetasum, \
                    N, R, J )
        
        return f, Df

    cdef tuple function_gradient_hessian(self, ndarray[np.float64_t, ndim=1] x, dict args):
        """Computes the likelihood function (only
        terms that contain `tau`), its gradient, and hessian.
        """

        cdef long N, R, J
        cdef ndarray left, total, zeta, pi
        cdef ndarray f, Df, Hf

        N = args['data'].N
        R = args['data'].R
        J = args['data'].J
        left = args['data'].left
        total = args['data'].total
        zeta = args['zeta'].value
        pi = args['pi'].value
        zetasum = args['zetasum']

        f = np.array([0.])
        tau_function(<double*> x.data, <double*> f.data, \
                    <double*> left.data, <double*> total.data, \
                    <double*> zeta.data, <double*> pi.data, zetasum, \
                    N, R, J)

        Df = np.zeros((1,self.J), dtype=float)
        tau_gradient(<double*> x.data, <double*> Df.data, \
                    <double*> left.data, <double*> total.data, \
                    <double*> zeta.data, <double*> pi.data, zetasum, \
                    N, R, J )
        
        Hf = np.zeros((self.J, self.J), dtype=float)
        tau_hessian(<double*> x.data, <double*> Hf.data, \
                   <double*> left.data, <double*> total.data, \
                   <double*> zeta.data, <double*> pi.data, zetasum, \
                   N, R, J)

        return f, Df, Hf
