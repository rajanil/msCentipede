
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
        self.value = np.hstack([-10*np.random.rand(1), np.random.rand(self.S-1)])

    cdef update(self, ndarray[np.float64_t, ndim=2] scores, zeta):
        """Update the estimates of parameter `beta` in the model.
        """

        # initial value
        xo = self.value.copy()

        # construct args dictionary
        args = dict([('scores',scores),('zeta',zeta)])

        try:
            self.value = optimizer.optimize(xo, self.function_gradient, self.function_gradient_hessian, args)
        except:
            pass

    @staticmethod
    cdef tuple function_gradient(ndarray[np.float64_t, ndim=1] x, dict args):
        """Computes the likelihood function (only terms that 
        contain `beta`) and its gradient.
        """

        cdef ndarray f, Df, scores, zeta, arg, pidx, nidx

        scores = args['scores']
        zeta = args['zeta'].value

        arg = utils.insum(x*scores, [1])

        pidx = arg>=0
        nidx = arg<0
        f = -1 * (np.sum(zeta[nidx]*arg[nidx],0) - \
                  np.sum((1-zeta[pidx])*arg[pidx],0) - \
                  np.sum(utils.nplog(1+np.exp(-1*np.abs(arg))),0))
        
        Df = -1 * utils.outsum(scores * (zeta - utils.logistic(-arg)))

        return f, Df

    @staticmethod
    cdef tuple function_gradient_hessian(ndarray[np.float64_t, ndim=1] x, dict args):
        """Computes the likelihood function (only terms that
        contain `beta`), its gradient, and its hessian.
        """

        cdef ndarray f, Df, Hf, scores, zeta, arg, larg, pidx, nidx

        scores = args['scores']
        zeta = args['zeta'].value

        arg = utils.insum(x * scores,[1])
        
        pidx = arg>=0
        nidx = arg<0
        f = -1 * (np.sum(zeta[nidx]*arg[nidx],0) - \
                  np.sum((1-zeta[pidx])*arg[pidx],0) - \
                  np.sum(utils.nplog(1+np.exp(-1*np.abs(arg))),0))
        
        Df = -1 * utils.outsum(scores * (zeta - utils.logistic(-arg)))
        
        larg = scores * utils.logistic(arg) * utils.logistic(-arg)
        Hf = np.dot(scores.T, larg)

        return f, Df, Hf
