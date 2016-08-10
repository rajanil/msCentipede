
# base libraries
import pdb

# numerical libraries
import numpy as np
cimport numpy as np
from numpy cimport ndarray

# custom libraries
import optimizer
import utils

cdef class Beta:
    """
    Class to store and update (M-step) the parameter `beta` in the logistic
    function in the prior of the msCentipede model.

    Arguments
        scores : array
        an array of scores for each motif instance. these could include
        PWM score, conservation score, a measure of various histone
        modifications, outputs from other algorithms, etc.

    """

    def __cinit__(self, ndarray[np.float64_t, ndim=2] scores):
    
        self.S = scores.shape[1]
        self.value = np.random.rand(self.S)

    cdef update(self, ndarray[np.float64_t, ndim=2] scores, zeta):
        """Update the estimates of parameter `beta` in the model.
        """

        # initial value
        xo = self.value.copy()

        # construct args dictionary
        args = dict([('scores',scores),('zeta',zeta)])

        self.value = optimizer.optimize(xo, self.function_gradient, self.function_gradient_hessian, args)

    cdef tuple function_gradient(self, ndarray[np.float64_t, ndim=1] x, dict args):
        """Computes the likelihood function (only terms that 
        contain `beta`) and its gradient.
        """

        cdef double f
        cdef ndarray scores, zeta, arg, Df

        scores = args['scores']
        zeta = args['zeta'].value

        arg = utils.insum(x*scores, [1])
        
        f = np.sum(arg * zeta - utils.nplog(1 + np.exp(arg)))
        
        Df = np.sum(scores * (zeta - utils.logistic(-arg)),0)
        
        return f, Df

    cdef tuple function_gradient_hessian(self, ndarray[np.float64_t, ndim=1] x, dict args):
        """Computes the likelihood function (only terms that
        contain `beta`), its gradient, and its hessian.
        """

        cdef double f
        cdef ndarray scores, zeta, arg, Df, Hf, larg

        scores = args['scores']
        zeta = args['zeta'].value

        arg = utils.insum(x * scores,[1])
        
        f = np.sum(arg * zeta - utils.nplog(1 + np.exp(arg)))
        
        Df = np.sum(scores * (zeta - utils.logistic(-arg)),0)
        
        larg = scores * utils.logistic(arg) * utils.logistic(-arg)
        Hf = np.dot(scores.T, larg)
        
        return f, Df, Hf
