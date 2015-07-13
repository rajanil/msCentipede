import numpy as np
cimport numpy as np
import cvxopt as cvx
from cvxopt import solvers
from scipy.special import digamma, gammaln, polygamma
import sys, time, math, pdb

# suppress optimizer output
solvers.options['show_progress'] = False

# defining some constants
EPS = np.finfo(np.double).tiny
MAX = np.finfo(np.double).max

# defining some simple functions
logistic = lambda x: 1./(1+np.exp(x))
insum = lambda x,axes: np.apply_over_axes(np.sum,x,axes)

def nplog(x):
    """Compute the natural logarithm, handling very
    small floats appropriately.

    """
    try:
        x[x<EPS] = EPS
    except TypeError:
        x = max([x,EPS])
    return np.log(x)


cdef class Data:
    """
    A data structure to store a multiscale representation of
    chromatin accessibility read counts across `N` genomic windows of
    length `L` in `R` replicates.

    Arguments
        reads : array

    """

    def __cinit__(self):

        self.N = 0
        self.L = 0
        self.R = 0
        self.J = 0
        self.value = dict()
        self.total = dict()

    cdef transform_to_multiscale(self, np.ndarray[np.float64_t, ndim=3] reads):
        """Transform a vector of read counts
        into a multiscale representation.
        
        .. note::
            See msCentipede manual for more details.   

        """

        cdef long k, j, size

        self.N = reads.shape[0]
        self.L = reads.shape[1]
        self.R = reads.shape[2]
        self.J = math.frexp(self.L)[1]-1
        for j from 0 <= j < self.J:
            size = self.L/(2**(j+1))
            self.total[j] = np.array([reads[:,k*size:(k+2)*size,:].sum(1) for k in xrange(0,2**(j+1),2)]).T
            self.value[j] = np.array([reads[:,k*size:(k+1)*size,:].sum(1) for k in xrange(0,2**(j+1),2)]).T

    def inverse_transform(self):
        """Transform a multiscale representation of the data or parameters,
        into vector representation.

        """

        if self.data:
            profile = np.array([val for k in xrange(2**self.J) \
                for val in [self.value[self.J-1][k][0],self.value[self.J-1][k][1]-self.value[self.J-1][k][0]]])
        else:
            profile = np.array([1])
            for j in xrange(self.J):
                profile = np.array([p for val in profile for p in [val,val]])
                vals = np.array([i for v in self.value[j] for i in [v,1-v]])
                profile = vals*profile

        return profile

    def copy(self):
        """ Create a copy of the class instance
        """

        cdef long j

        newcopy = Data()
        newcopy.J = self.J
        newcopy.N = self.N
        newcopy.L = self.L
        newcopy.R = self.R
        for j from 0 <= j < self.J:
            newcopy.value[j] = self.value[j]
            newcopy.total[j] = self.total[j]

        return newcopy


cdef class Zeta:
    """
    Inference class to store and update (E-step) the posterior
    probability that a transcription factor is bound to a motif
    instance.

    Arguments
        data : Data
        totalreads : array

    """

    def __cinit__(self, np.ndarray[np.float64_t, ndim=2] totalreads, long N, bool infer):

        cdef np.ndarray order, indices

        self.N = N
        self.total = totalreads

        if infer:
            self.prior_log_odds = np.zeros((self.N,1), dtype=float)
            self.footprint_log_likelihood_ratio = np.zeros((self.N,1), dtype=float)
            self.total_log_likelihood_ratio = np.zeros((self.N,1), dtype=float)
            self.posterior_log_odds = np.zeros((self.N,1), dtype=float)
        else:
            self.estim = np.zeros((self.N, 2),dtype=float)
            order = np.argsort(self.total.sum(1))
            indices = order[:self.N/2]
            self.estim[indices,1:] = -MAX
            indices = order[self.N/2:]
            self.estim[indices,1:] = MAX
            self.estim = np.exp(self.estim - np.max(self.estim,1).reshape(self.N,1))
            self.estim = self.estim / insum(self.estim,[1])

    cdef update(self, Data data, np.ndarray[np.float64_t, ndim=2] scores, \
        Pi pi, Tau tau, Alpha alpha, Beta beta, Omega omega, \
        Pi pi_null, Tau tau_null, str model):

        cdef long j
        cdef np.ndarray[np.float64_t, ndim=2] footprint_logodds, prior_logodds, negbin_logodds
        cdef Data lhoodA, lhoodB

        footprint_logodds = np.zeros((self.N,1), dtype=float)
        lhoodA, lhoodB = compute_footprint_likelihood(data, pi, tau, pi_null, tau_null, model)

        for j from 0 <= j < data.J:
            footprint_logodds += insum(lhoodA.value[j] - lhoodB.value[j],[1])

        prior_logodds = insum(beta.estim * scores, [1])
        negbin_logodds = insum(gammaln(self.total + alpha.estim.T[1]) \
                - gammaln(self.total + alpha.estim.T[0]) \
                + gammaln(alpha.estim.T[0]) - gammaln(alpha.estim.T[1]) \
                + alpha.estim.T[1] * nplog(omega.estim.T[1]) - alpha.estim.T[0] * nplog(omega.estim.T[0]) \
                + self.total * (nplog(1 - omega.estim.T[1]) - nplog(1 - omega.estim.T[0])),[1])

        self.estim[:,1:] = prior_logodds + footprint_logodds + negbin_logodds
        self.estim[:,0] = 0.
        self.estim[self.estim==np.inf] = MAX
        self.estim = np.exp(self.estim-np.max(self.estim,1).reshape(self.N,1))
        self.estim = self.estim/insum(self.estim,[1])

    cdef infer(self, Data data, np.ndarray[np.float64_t, ndim=2] scores, \
        Pi pi, Tau tau, Alpha alpha, Beta beta, Omega omega, \
        Pi pi_null, Tau tau_null, str model):

        cdef long j
        cdef Data lhoodA, lhoodB

        lhoodA, lhoodB = compute_footprint_likelihood(data, pi, tau, pi_null, tau_null, model)

        for j from 0 <= j < data.J:
            self.footprint_log_likelihood_ratio += insum(lhoodA.value[j] - lhoodB.value[j],[1])
        self.footprint_log_likelihood_ratio = self.footprint_log_likelihood_ratio / np.log(10)

        self.prior_log_odds = insum(beta.estim * scores, [1]) / np.log(10)

        self.total_log_likelihood_ratio = insum(gammaln(self.total + alpha.estim.T[1]) \
            - gammaln(self.total + alpha.estim.T[0]) \
            + gammaln(alpha.estim.T[0]) - gammaln(alpha.estim.T[1]) \
            + alpha.estim.T[1] * nplog(omega.estim.T[1]) - alpha.estim.T[0] * nplog(omega.estim.T[0]) \
            + self.total * (nplog(1 - omega.estim.T[1]) - nplog(1 - omega.estim.T[0])),[1])
        self.total_log_likelihood_ratio = self.total_log_likelihood_ratio / np.log(10)

        self.posterior_log_odds = self.prior_log_odds \
            + self.footprint_log_likelihood_ratio \
            + self.total_log_likelihood_ratio


cdef class Pi:
    """
    Class to store and update (M-step) the parameter `p` in the
    msCentipede model. It is also used for the parameter `p_o` in
    the msCentipede-flexbg model.

    Arguments
        J : int
        number of scales

    """

    def __cinit__(self, long J):

        cdef long j
        self.J = J
        self.value = dict()
        for j from 0 <= j < self.J:
            self.value[j] = np.random.rand(2**j)

    def update(self, Data data, Zeta zeta, Tau tau):
        """Update the estimates of parameter `p` (and `p_o`) in the model.
        """

        def function_gradient(np.ndarray[np.float64_t, ndim=1] x, dict kwargs):

            """Computes part of the likelihood function that has
            terms containing `pi`, along with its gradient
            """

            cdef Data data
            cdef Zeta zeta
            cdef Tau tau
            cdef long j, J, r, left, right
            cdef double f
            cdef np.ndarray func, F, Df, df, alpha, beta, data_alpha, data_beta, zetaestim

            data = kwargs['data']
            zeta = kwargs['zeta']
            tau = kwargs['tau']
            zetaestim = kwargs['zetaestim']

            F = np.zeros((zeta.N,), dtype=float)
            Df = np.zeros((x.size,), dtype=float)

            for j from 0 <= j < self.J:
                J = 2**j
                left = J-1
                right = 2*J-1
                func = np.zeros((data.N,J), dtype=float)
                df = np.zeros((data.N,J), dtype=float)
                alpha = x[left:right] * tau.estim[j]
                beta = (1-x[left:right]) * tau.estim[j]
                
                for r from 0 <= r < data.R:
                    data_alpha = data.value[j][r] + alpha
                    data_beta = data.total[j][r] - data.value[j][r] + beta
                    func += gammaln(data_alpha) + gammaln(data_beta)
                    df += digamma(data_alpha) - digamma(data_beta)
                
                F += np.sum(func,1) - np.sum(gammaln(alpha) + gammaln(beta)) * data.R * J
                Df[left:right] = -1. * tau.estim[j] * (np.sum(zeta.estim[:,1:] * df,0) \
                    - zetaestim * (digamma(alpha) - digamma(beta)))
            
            f = -1. * np.sum(zeta.estim[:,1] * F)
            
            return f, Df

        def function_gradient_hessian(np.ndarray[np.float64_t, ndim=1] x, dict kwargs):

            """Computes part of the likelihood function that has
            terms containing `pi`, along with its gradient and hessian
            """

            cdef Data data
            cdef Zeta zeta
            cdef Tau tau
            cdef long j, J, r, left, right
            cdef double f
            cdef np.ndarray func, F, Df, df, hf, hess, Hf 

            data = kwargs['data']
            zeta = kwargs['zeta']
            tau = kwargs['tau']
            zetaestim = kwargs['zetaestim']

            F = np.zeros((zeta.N,), dtype=float)
            Df = np.zeros((x.size,), dtype=float)
            hess = np.zeros((x.size,), dtype=float)

            for j from 0 <= j < self.J:
                J = 2**j
                left = J-1
                right = 2*J-1
                func = np.zeros((data.N,J), dtype=float)
                df = np.zeros((data.N,J), dtype=float)
                hf = np.zeros((data.N,J), dtype=float)
                alpha = x[left:right] * tau.estim[j]
                beta = (1-x[left:right]) * tau.estim[j]

                for r from 0 <= r < data.R:
                    data_alpha = data.value[j][r] + alpha
                    data_beta = data.total[j][r] - data.value[j][r] + beta
                    func += gammaln(data_alpha) + gammaln(data_beta)
                    df += digamma(data_alpha) - digamma(data_beta)
                    hf += polygamma(1, data_alpha) + polygamma(1, data_beta)

                F += np.sum(func,1) - np.sum(gammaln(alpha) + gammaln(beta)) * data.R * J
                Df[left:right] = -1. * tau.estim[j] * (np.sum(zeta.estim[:,1:] * df,0) \
                    - zetaestim * (digamma(alpha) - digamma(beta)))
                hess[left:right] = -1. * tau.estim[j]**2 * (np.sum(zeta.estim[:,1:] * hf,0) \
                    - zetaestim * (polygamma(1, alpha) + polygamma(1, beta)))

            f = -1. * np.sum(zeta.estim[:,1] * F)
            Hf = np.diag(hess)
            
            return f, Df, Hf

        # initialize optimization variable
        xo = np.array([v for j in xrange(self.J) for v in self.value[j]])
        zetaestim = zeta.estim[:,1].sum()
        X = xo.size

        # set constraints for optimization variable
        G = np.vstack((np.diag(-1*np.ones((X,), dtype=float)), \
                np.diag(np.ones((X,), dtype=float))))
        h = np.vstack((np.zeros((X,1), dtype=float), \
                np.ones((X,1), dtype=float)))

        # call optimizer
        x_final = optimizer(xo, function_gradient, function_gradient_hessian, \
            G=G, h=h, data=data, zeta=zeta, tau=tau, zetaestim=zetaestim)

        if np.isnan(x_final).any():
            print "Nan in Pi"
            raise ValueError

        if np.isinf(x_final).any():
            print "Inf in Pi"
            raise ValueError

        # store optimum in data structure
        self.value = dict([(j,x_final[2**j-1:2**(j+1)-1]) for j in xrange(self.J)])


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
        self.estim = 10*np.random.rand(self.J)

    def update(self, Data data, Zeta zeta, Pi pi):
        """Update the estimates of parameter `tau` (and `tau_o`) in the model.
        """

        def function_gradient(np.ndarray[np.float64_t, ndim=1] x, dict kwargs):
            """Computes part of the likelihood function that has
            terms containing `tau`, and its gradient.
            """

            cdef Data data
            cdef Zeta zeta
            cdef Pi pi
            cdef long j, r, left, right
            cdef double F
            cdef np.ndarray func, f, Df, df, alpha, beta, data_alpha, data_beta, data_x

            data = kwargs['data']
            zeta = kwargs['zeta']
            pi = kwargs['pi']
            zetaestim = kwargs['zetaestim']

            func = np.zeros((zeta.N,), dtype=float)
            Df = np.zeros((x.size,), dtype=float)
            # loop over each scale
            for j from 0 <= j < self.J:

                alpha = pi.value[j] * x[j]
                beta = (1 - pi.value[j]) * x[j]
                df = np.zeros((zeta.N,), dtype=float)
                # loop over replicates
                for r from 0 <= r < data.R:

                    data_alpha = data.value[j][r] + alpha
                    data_beta = data.total[j][r] - data.value[j][r] + beta
                    data_x = data.total[j][r] + x[j]
                    f = gammaln(data_alpha) + gammaln(data_beta) \
                        - gammaln(data_x) + gammaln(x[j]) \
                        - gammaln(pi.value[j] * x[j]) - gammaln((1 - pi.value[j]) * x[j])
                    func += np.sum(f, 1)

                    f = pi.value[j] * digamma(data_alpha) \
                        + (1 - pi.value[j]) * gammaln(data_beta) \
                        - digamma(data_x) + digamma(x[j]) \
                        - pi.value[j] * digamma(pi.value[j] * x[j]) \
                        - (1 - pi.value[j]) * digamma((1 - pi.value[j]) * x[j])
                    df += np.sum(f, 1)

                Df[j] = -1 * np.sum(zeta.estim[:,1] * df)

            F = -1. * np.sum(zeta.estim[:,1] * func)
            
            return F, Df

        def function_gradient_hessian(np.ndarray[np.float64_t, ndim=1] x, dict kwargs):
            """Computes part of the likelihood function that has
            terms containing `tau`, and its gradient and hessian.
            """

            cdef Data data
            cdef Zeta zeta
            cdef Pi pi
            cdef long j, r, left, right
            cdef double F
            cdef np.ndarray func, f, Df, df, hf, hess, Hf, alpha, beta, data_alpha, data_beta, data_x

            data = kwargs['data']
            zeta = kwargs['zeta']
            pi = kwargs['pi']
            zetaestim = kwargs['zetaestim']

            func = np.zeros((zeta.N,), dtype=float)
            Df = np.zeros((x.size,), dtype=float)
            hess = np.zeros((x.size,), dtype=float)
            # loop over each scale
            for j from 0 <= j < self.J:

                alpha = pi.value[j] * x[j]
                beta = (1 - pi.value[j]) * x[j]
                df = np.zeros((zeta.N,), dtype=float)
                hf = np.zeros((zeta.N,), dtype=float)
                # loop over replicates
                for r from 0 <= r < data.R:

                    data_alpha = data.value[j][r] + alpha
                    data_beta = data.total[j][r] - data.value[j][r] + beta
                    data_x = data.total[j][r] + x[j]

                    f = gammaln(data_alpha) + gammaln(data_beta) \
                        - gammaln(data_x) + gammaln(x[j]) \
                        - gammaln(pi.value[j] * x[j]) - gammaln((1 - pi.value[j]) * x[j])
                    func += np.sum(f, 1)

                    f = pi.value[j] * digamma(data_alpha) \
                        + (1 - pi.value[j]) * gammaln(data_beta) \
                        - digamma(data_x) + digamma(x[j]) \
                        - pi.value[j] * digamma(pi.value[j] * x[j]) \
                        - (1 - pi.value[j]) * digamma((1 - pi.value[j]) * x[j])
                    df += np.sum(f, 1)

                    f = pi.value[j]**2 * polygamma(1, data_alpha) \
                        + (1 - pi.value[j])**2 * polygamma(1, data_beta) \
                        - polygamma(1, data_x) + polygamma(1, x[j]) \
                        - pi.value[j]**2 * polygamma(1, pi.value[j] * x[j]) \
                        - (1 - pi.value[j])**2 * polygamma(1, (1 - pi.value[j]) * x[j])
                    hf += np.sum(f, 1)

                Df[j] = -1 * np.sum(zeta.estim[:,1] * df)
                hess[j] = -1 * np.sum(zeta.estim[:,1] * hf)

            F = -1. * np.sum(zeta.estim[:,1] * func)
            Hf = np.diag(hess)

            return F, Df, Hf

        # initialize optimization variables
        xo = self.estim.copy()
        zetaestim = np.sum(zeta.estim[:,1])

        # set constraints for optimization variables
        G = np.diag(-1 * np.ones((self.J,), dtype=float))
        h = np.zeros((self.J,1), dtype=float)

        # call optimizer
        x_final = optimizer(xo, function_gradient, function_gradient_hessian, \
            G=G, h=h, data=data, zeta=zeta, pi=pi, zetaestim=zetaestim)
        self.estim = x_final

        if np.isnan(self.estim).any():
            print "Nan in Tau"
            raise ValueError

        if np.isinf(self.estim).any():
            print "Inf in Tau"
            raise ValueError


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
        self.estim = np.random.rand(self.R,2)*10

    def update(self, Zeta zeta, Omega omega):
        """Update the estimates of parameter `alpha` in the model.
        """

        def function_gradient(np.ndarray[np.float64_t, ndim=1] x, dict kwargs):
            """Computes part of the likelihood function that has
            terms containing `alpha`, and its gradient
            """

            cdef long r
            cdef double f
            cdef Zeta zeta
            cdef Omega omega
            cdef np.ndarray df, Df, constant, zetaestim, func, xzeta

            zeta = kwargs['zeta']
            omega = kwargs['omega']
            constant = kwargs['constant']
            zetaestim = kwargs['zetaestim']

            func = 0
            df = np.zeros((2*self.R,), dtype='float')

            for r from 0 <= r < self.R:
                xzeta = zeta.total[:,r:r+1] + x[2*r:2*r+2]
                func = func + np.sum(np.sum(gammaln(xzeta) * zeta.estim, 0) \
                    - gammaln(x[2*r:2*r+2]) * zetaestim + constant[r] * x[2*r:2*r+2])
                df[2*r:2*r+2] = np.sum(digamma(xzeta) * zeta.estim, 0) \
                    - digamma(x[2*r:2*r+2]) * zetaestim + constant[r]

            f = -1.*func
            Df = -1. * df

            return f, Df

        def function_gradient_hessian(np.ndarray[np.float64_t, ndim=1] x, dict kwargs):
            """Computes part of the likelihood function that has
            terms containing `alpha`, and its gradient and hessian
            """

            cdef long r
            cdef double f
            cdef Zeta zeta
            cdef Omega omega
            cdef np.ndarray df, Df, hf, Hf, constant, zetaestim, func, xzeta

            zeta = kwargs['zeta']
            omega = kwargs['omega']
            zetaestim = kwargs['zetaestim']
            constant = kwargs['constant']
            
            func = 0
            df = np.zeros((2*self.R,), dtype='float')
            hess = np.zeros((2*self.R,), dtype='float')

            for r from 0 <= r < self.R:
                xzeta = zeta.total[:,r:r+1] + x[2*r:2*r+2]
                func = func + np.sum(np.sum(gammaln(xzeta) * zeta.estim, 0) \
                    - gammaln(x[2*r:2*r+2]) * zetaestim + constant[r] * x[2*r:2*r+2])
                df[2*r:2*r+2] = np.sum(digamma(xzeta) * zeta.estim, 0) \
                    - digamma(x[2*r:2*r+2]) * zetaestim + constant[r]
                hess[2*r:2*r+2] = np.sum(polygamma(1, xzeta) * zeta.estim, 0) \
                    - polygamma(1, x[2*r:2*r+2]) * zetaestim

            f = -1.*func
            Df = -1. * df       
            Hf = -1. * np.diag(hess)

            return f, Df, Hf

        cdef np.ndarray zetaestim, constant, xo, G, h, x_final

        zetaestim = np.sum(zeta.estim,0)
        constant = zetaestim*nplog(omega.estim)

        # initialize optimization variables
        xo = self.estim.ravel()

        # set constraints for optimization variables
        G = np.diag(-1 * np.ones((2*self.R,), dtype=float))
        h = np.zeros((2*self.R,1), dtype=float)

        # call optimizer
        x_final = optimizer(xo, function_gradient, function_gradient_hessian, \
            G=G, h=h, omega=omega, zeta=zeta, constant=constant, zetaestim=zetaestim)
        self.estim = x_final.reshape(self.R,2)

        if np.isnan(self.estim).any():
            print "Nan in Alpha"
            raise ValueError

        if np.isinf(self.estim).any():
            print "Inf in Alpha"
            raise ValueError


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
        self.estim = np.random.rand(self.R,2)
        self.estim[:,1] = self.estim[:,1]/100

    cdef update(self, Zeta zeta, Alpha alpha):
        """Update the estimates of parameter `omega` in the model.
        """

        cdef np.ndarray numerator, denominator

        numerator = np.sum(zeta.estim,0) * alpha.estim
        denominator = np.array([np.sum(zeta.estim * (estim + zeta.total[:,r:r+1]), 0) \
            for r,estim in enumerate(alpha.estim)])
        self.estim = numerator / denominator

        if np.isnan(self.estim).any():
            print "Nan in Omega"
            raise ValueError

        if np.isinf(self.estim).any():
            print "Inf in Omega"
            raise ValueError


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

    def __cinit__(self, np.ndarray[np.float64_t, ndim=2] scores):
    
        self.S = scores.shape[1]
        self.estim = np.random.rand(self.S)

    def update(self, np.ndarray[np.float64_t, ndim=2] scores, Zeta zeta):
        """Update the estimates of parameter `beta` in the model.
        """

        def function_gradient(np.ndarray[np.float64_t, ndim=1] x, dict kwargs):
            """Computes part of the likelihood function that has
            terms containing `beta`, and its gradient.
            """

            scores = kwargs['scores']
            zeta = kwargs['zeta']

            arg = insum(x * scores,[1])
            
            func = arg * zeta.estim[:,1:] - nplog(1 + np.exp(arg))
            f = -1. * func.sum()
            
            Df = -1 * np.sum(scores * (zeta.estim[:,1:] - logistic(-arg)),0)
            
            return f, Df

        def function_gradient_hessian(np.ndarray[np.float64_t, ndim=1] x, dict kwargs):
            """Computes part of the likelihood function that has
            terms containing `beta`, and its gradient and hessian.
            """

            scores = kwargs['scores']
            zeta = kwargs['zeta']

            arg = insum(x * scores,[1])
            
            func = arg * zeta.estim[:,1:] - nplog(1 + np.exp(arg))
            f = -1. * func.sum()
            
            Df = -1 * np.sum(scores * (zeta.estim[:,1:] - logistic(-arg)),0)
            
            larg = scores * logistic(arg) * logistic(-arg)
            Hf = np.dot(scores.T, larg)
            
            return f, Df, Hf

        xo = self.estim.copy()
        self.estim = optimizer(xo, function_gradient, \
            function_gradient_hessian, scores=scores, zeta=zeta)

        if np.isnan(self.estim).any():
            print "Nan in Beta"
            raise ValueError

        if np.isinf(self.estim).any():
            print "Inf in Beta"
            raise ValueError


def optimizer(np.ndarray[np.float64_t, ndim=1] xo, function_gradient, function_gradient_hessian, **kwargs):
    """Calls the appropriate nonlinear convex optimization solver 
    in the package `cvxopt` to find optimal values for the relevant
    parameters, given subroutines that evaluate a function, 
    its gradient, and hessian, this subroutine 

    Arguments
        function : function object
        evaluates the function at the specified parameter values

        gradient : function object
        evaluates the gradient of the function

        hessian : function object
        evaluates the hessian of the function

    """

    def F(x=None, z=None):
        """A subroutine that the cvxopt package can call to get 
        values of the function, gradient and hessian during
        optimization.
        """

        if x is None:
            return 0, cvx.matrix(x_init)

        xx = np.array(x).ravel().astype(np.float64)

        if z is None:

            # compute likelihood function and gradient
            f, Df = function_gradient(xx, kwargs)

            # check for infs and nans in function and gradient
            if np.isnan(f) or np.isinf(f):
                f = np.array([np.finfo('float32').max]).astype('float')
            else:
                f = np.array([f]).astype('float')
            if np.isnan(Df).any() or np.isinf(Df).any():
                Df = -1 * np.finfo('float32').max * np.ones((1,xx.size), dtype=float)
            else:
                Df = Df.reshape(1,xx.size)

            return cvx.matrix(f), cvx.matrix(Df)

        else:

            # compute likelihood function, gradient, and hessian
            f, Df, hess = function_gradient_hessian(xx, kwargs)

            # check for infs and nans in function and gradient
            if np.isnan(f) or np.isinf(f):
                f = np.array([np.finfo('float32').max]).astype('float')
            else:
                f = np.array([f]).astype('float')
            if np.isnan(Df).any() or np.isinf(Df).any():
                Df = -1 * np.finfo('float32').max * np.ones((1,xx.size), dtype=float)
            else:
                Df = Df.reshape(1,xx.size)

            Hf = z[0] * hess
            return cvx.matrix(f), cvx.matrix(Df), cvx.matrix(Hf)

    # warm start for the optimization
    optimized = False
    V = xo.size
    x_init = xo.reshape(V,1)

    while not optimized:

        try:

            # call the optimization subroutine in cvxopt
            if kwargs.has_key('G'):
                # call a constrained nonlinear solver
                solution = solvers.cp(F, G=cvx.matrix(kwargs['G']), h=cvx.matrix(kwargs['h']))
            else:
                # call an unconstrained nonlinear solver
                solution = solvers.cp(F)

            # check if optimal value has been reached; 
            # if not, re-optimize with a cold start
            if solution['status']=='optimal':
                optimized = True
                x_final = np.array(solution['x']).ravel()
            else:
                # cold start
                x_init = np.random.rand(V,1)

        except ValueError:

            # if any parameter becomes Inf or Nan during optimization,
            # re-optimize with a cold start
            x_init = np.random.rand(V,1)

    return x_final


cdef tuple compute_footprint_likelihood(Data data, Pi pi, Tau tau, Pi pi_null, Tau tau_null, str model):
    """Evaluates the likelihood function for the 
    footprint part of the bound model and background model.

    Arguments
        data : Data
        transformed read count data 

        pi : Pi
        estimate of mean footprint parameters at bound sites

        tau : Tau
        estimate of footprint heterogeneity at bound sites

        pi_null : Pi
        estimate of mean cleavage pattern at unbound sites

        tau_null : Tau or None
        estimate of cleavage heterogeneity at unbound sites

        model : string 
        {msCentipede, msCentipede-flexbgmean, msCentipede-flexbg}

    """

    cdef long j, r
    cdef Data lhood_bound, lhood_unbound

    lhood_bound = Data()
    lhood_unbound = Data()

    for j from 0 <= j < data.J:
        value = np.sum(data.value[j],0)
        total = np.sum(data.total[j],0)
        
        lhood_bound.value[j] = np.sum([gammaln(data.value[j][r] + pi.value[j] * tau.estim[j]) \
            + gammaln(data.total[j][r] - data.value[j][r] + (1 - pi.value[j]) * tau.estim[j]) \
            - gammaln(data.total[j][r] + tau.estim[j]) + gammaln(tau.estim[j]) \
            - gammaln(pi.value[j] * tau.estim[j]) - gammaln((1 - pi.value[j]) * tau.estim[j]) \
            for r in xrange(data.R)],0)

        if model in ['msCentipede','msCentipede_flexbgmean']:
            
            lhood_unbound.value[j] = value * nplog(pi_null.value[j]) \
                + (total - value) * nplog(1 - pi_null.value[j])
    
        elif model=='msCentipede_flexbg':
            
            lhood_unbound.value[j] = np.sum([gammaln(data.value[j][r] + pi_null.value[j] * tau_null.estim[j]) \
                + gammaln(data.total[j][r] - data.value[j][r] + (1 - pi_null.value[j]) * tau_null.estim[j]) \
                - gammaln(data.total[j][r] + tau_null.estim[j]) + gammaln(tau_null.estim[j]) \
                - gammaln(pi_null.value[j] * tau_null.estim[j]) - gammaln((1 - pi_null.value[j]) * tau_null.estim[j]) \
                for r in xrange(data.R)],0)

    return lhood_bound, lhood_unbound


cdef double likelihood(Data data, np.ndarray[np.float64_t, ndim=2] scores, \
    Zeta zeta, Pi pi, Tau tau, Alpha alpha, Beta beta, \
    Omega omega, Pi pi_null, Tau tau_null, str model):
    """Evaluates the likelihood function of the full
    model, given estimates of model parameters.

    Arguments
        data : Data
        transformed read count data

        scores : array
        an array of scores for each motif instance. these could include
        PWM score, conservation score, a measure of various histone
        modifications, outputs from other algorithms, etc.

        zeta : zeta
        expected value of factor binding state for each site.

        pi : Pi
        estimate of mean footprint parameters at bound sites

        tau : Tau
        estimate of footprint heterogeneity at bound sites

        alpha : Alpha
        estimate of negative binomial parameters for each replicate

        beta : Beta
        weights for various scores in the logistic function 

        omega : Omega
        estimate of negative binomial parameters for each replicate

        pi_null : Pi
        estimate of mean cleavage pattern at unbound sites

        tau_null : Tau or None
        estimate of cleavage heterogeneity at unbound sites

        model : string
        {msCentipede, msCentipede-flexbgmean, msCentipede-flexbg}

    """

    cdef long j
    cdef double L
    cdef np.ndarray apriori, footprint, null, P_1, P_0, LL
    cdef Data lhoodA, lhoodB

    apriori = insum(beta.estim * scores,[1])

    lhoodA, lhoodB = compute_footprint_likelihood(data, pi, tau, pi_null, tau_null, model)

    footprint = np.zeros((data.N,1),dtype=float)
    for j from 0 <= j < data.J:
        footprint += insum(lhoodA.value[j],[1])

    P_1 = footprint + insum(gammaln(zeta.total + alpha.estim[:,1]) - gammaln(alpha.estim[:,1]) \
        + alpha.estim[:,1] * nplog(omega.estim[:,1]) + zeta.total * nplog(1 - omega.estim[:,1]), [1])
    P_1[P_1==np.inf] = MAX
    P_1[P_1==-np.inf] = -MAX

    null = np.zeros((data.N,1), dtype=float)
    for j from 0 <= j < data.J:
        null += insum(lhoodB.value[j],[1])

    P_0 = null + insum(gammaln(zeta.total + alpha.estim[:,0]) - gammaln(alpha.estim[:,0]) \
        + alpha.estim[:,0] * nplog(omega.estim[:,0]) + zeta.total * nplog(1 - omega.estim[:,0]), [1])
    P_0[P_0==np.inf] = MAX
    P_0[P_0==-np.inf] = -MAX

    LL = P_0 * zeta.estim[:,:1] + insum(P_1 * zeta.estim[:,1:],[1]) + apriori * (1 - zeta.estim[:,:1]) \
        - nplog(1 + np.exp(apriori)) - insum(zeta.estim * nplog(zeta.estim),[1])
    
    L = LL.sum() / data.N

    if np.isnan(L):
        print "Nan in LogLike"
        return -np.inf

    if np.isinf(L):
        print "Inf in LogLike"
        return -np.inf

    return L


cdef EM(Data data, np.ndarray[np.float64_t, ndim=2] scores, \
    Zeta zeta, Pi pi, Tau tau, Alpha alpha, Beta beta, \
    Omega omega, Pi pi_null, Tau tau_null, str model):
    """This subroutine updates all model parameters once and computes an
    estimate of the posterior probability of binding.

    Arguments
        data : Data
        transformed read count data

        scores : array
        an array of scores for each motif instance. these could include
        PWM score, conservation score, a measure of various histone
        modifications, outputs from other algorithms, etc.

        zeta : zeta
        expected value of factor binding state for each site.

        pi : Pi
        estimate of mean footprint parameters at bound sites

        tau : Tau
        estimate of footprint heterogeneity at bound sites

        alpha : Alpha
        estimate of negative binomial parameters for each replicate

        beta : Beta
        weights for various scores in the logistic function 

        omega : Omega
        estimate of negative binomial parameters for each replicate

        pi_null : Pi
        estimate of mean cleavage pattern at unbound sites

        tau_null : Tau or None
        estimate of cleavage heterogeneity at unbound sites

        model : string
        {msCentipede, msCentipede-flexbgmean, msCentipede-flexbg}

    """

    cdef double starttime

    # update binding posteriors
    zeta.update(data, scores, pi, tau, \
            alpha, beta, omega, pi_null, tau_null, model)

    # update multi-scale parameters
    starttime = time.time()
    pi.update(data, zeta, tau)
    print "p_jk update in %.3f secs"%(time.time()-starttime)

    starttime = time.time()
    tau.update(data, zeta, pi)
    print "tau update in %.3f secs"%(time.time()-starttime)
    
    # update negative binomial parameters
    starttime = time.time()
    omega.update(zeta, alpha)
    print "omega update in %.3f secs"%(time.time()-starttime)

    starttime = time.time()
    alpha.update(zeta, omega)
    print "alpha update in %.3f secs"%(time.time()-starttime)

    # update prior parameters
    starttime = time.time()
    beta.update(scores, zeta)
    print "beta update in %.3f secs"%(time.time()-starttime)


cdef square_EM(Data data, np.ndarray[np.float64_t, ndim=2] scores, \
    Zeta zeta, Pi pi, Tau tau, Alpha alpha, Beta beta, \
    Omega omega, Pi pi_null, Tau tau_null, str model):
    """Accelerated update of model parameters and posterior probability of binding.

    Arguments
        data : Data
        transformed read count data

        scores : array
        an array of scores for each motif instance. these could include
        PWM score, conservation score, a measure of various histone
        modifications, outputs from other algorithms, etc.

        zeta : zeta
        expected value of factor binding state for each site.

        pi : Pi
        estimate of mean footprint parameters at bound sites

        tau : Tau
        estimate of footprint heterogeneity at bound sites

        alpha : Alpha
        estimate of negative binomial parameters for each replicate

        beta : Beta
        weights for various scores in the logistic function 

        omega : Omega
        estimate of negative binomial parameters for each replicate

        pi_null : Pi
        estimate of mean cleavage pattern at unbound sites

        tau_null : Tau or None
        estimate of cleavage heterogeneity at unbound sites

        model : string
        {msCentipede, msCentipede-flexbgmean, msCentipede-flexbg}

    """

    cdef long j, step
    cdef double a
    cdef np.ndarray r, v, varA, varB, varC, invalid, newparam
    cdef list parameters, oldvar, oldvars, R, V

    parameters = [pi, tau, alpha, omega]
    oldvar = []
    for parameter in parameters:
        try:
            oldvar.append(parameter.estim.copy())
        except AttributeError:
            oldvar.append(np.hstack([parameter.value[j].copy() for j in xrange(parameter.J)]))
    oldvars = [oldvar]

    # take two update steps
    for step in [0,1]:
        EM(data, scores, zeta, pi, tau, alpha, beta, omega, pi_null, tau_null, model)
        oldvar = []
        for parameter in parameters:
            try:
                oldvar.append(parameter.estim.copy())
            except AttributeError:
                oldvar.append(np.hstack([parameter.value[j].copy() for j in xrange(parameter.J)]))
        oldvars.append(oldvar)

    R = [oldvars[1][j]-oldvars[0][j] for j in xrange(len(parameters))]
    V = [oldvars[2][j]-oldvars[1][j]-R[j] for j in xrange(len(parameters))]
    a = -1.*np.sqrt(np.sum([(r*r).sum() for r in R]) / np.sum([(v*v).sum() for v in V]))

    if a>-1:
        a = -1.

    # given two update steps, compute an optimal step that achieves
    # a better likelihood than the two steps.
    a_ok = False
    while not a_ok:
        invalid = np.zeros((0,), dtype='bool')
        for parameter,varA,varB,varC in zip(parameters,oldvars[0],oldvars[1],oldvars[2]):
            try:
                parameter.estim = (1+a)**2*varA - 2*a*(1+a)*varB + a**2*varC
                # ensure constraints on variables are satisfied
                invalid = np.hstack((invalid,(parameter.estim<=0).ravel()))
            except AttributeError:
                newparam = (1+a)**2*varA - 2*a*(1+a)*varB + a**2*varC
                # ensure constraints on variables are satisfied
                invalid = np.hstack((invalid, np.logical_or(newparam<0, newparam>1)))
                parameter.value = dict([(j,newparam[2**j-1:2**(j+1)-1]) \
                    for j in xrange(parameter.J)])
        if np.any(invalid):
            a = (a-1)/2.
            if np.abs(a+1)<1e-4:
                a = -1.
        else:
            a_ok = True

    EM(data, scores, zeta, pi, tau, alpha, beta, omega, pi_null, tau_null, model)


def estimate_optimal_model(np.ndarray[np.float64_t, ndim=2] reads, \
    np.ndarray[np.float64_t, ndim=2] totalreads, \
    np.ndarray[np.float64_t, ndim=2] scores, \
    np.ndarray[np.float64_t, ndim=2] background, \
    str model, long restarts, double mintol):
    """Learn the model parameters by running an EM algorithm till convergence.
    Return the optimal parameter estimates from a number of EM results starting 
    from random restarts.

    Arguments
        reads : array
        array of read counts at each base in a genomic window,
        across motif instances and several measurement replicates.

        totalreads : array
        array of total read counts in a genomic window,
        across motif instances and several measurement replicates.
        the size of the genomic window can be different for 
        `reads` and `totalreads`.

        scores : array
        an array of scores for each motif instance. these could include
        PWM score, conservation score, a measure of various histone
        modifications, outputs from other algorithms, etc.

        background : array
        a uniform, normalized array for a uniform background model.
        when sequencing reads from genomic DNA are available, this
        is an array of read counts at each base in a genomic window,
        across motif instances.

        model : string
        {msCentipede, msCentipede-flexbgmean, msCentipede-flexbg}

        restarts : int
        number of independent runs of model learning

        mintol : float
        convergence criterion

    """

    cdef long restart, iteration, err
    cdef double change, maxLoglike, Loglike, tol, itertime, totaltime
    cdef np.ndarray oldtau, negbinmeans
    cdef Data data, data_null
    cdef Beta beta
    cdef Alpha alpha
    cdef Omega omega
    cdef Pi pi, pi_null
    cdef Tau tau, tau_null
    cdef Zeta zeta, zeta_null

    # transform data into multiscale representation
    data = Data(reads)
    data_null = Data(background)
    scores = np.hstack((np.ones((data.N,1), dtype=float), scores))
    del reads

    # set background model
    pi_null = Pi(data_null.J)
    for j in xrange(pi_null.J):
        pi_null.value[j] = np.sum(np.sum(data_null.value[j],0),0) / np.sum(np.sum(data_null.total[j],0),0).astype('float')
    
    tau_null = Tau(data_null.J)
    if model=='msCentipede_flexbg':

        zeta_null = Zeta(data_null, background.sum(1))
        zeta_null.estim[:,1] = 1
        zeta_null.estim[:,0] = 0

        # iterative update of background model; 
        # evaluate convergence based on change in estimated
        # background overdispersion
        change = np.inf
        while change>1e-2:
            oldtau = tau_null.estim.copy()
            
            tau_null.update(data_null, zeta_null, pi_null)
            pi_null.update(data_null, zeta_null, tau_null)

            change = np.abs(oldtau-tau_null.estim).sum() / tau_null.J

    maxLoglike = -np.inf
    restart = 0
    err = 1
    runlog = ['Number of sites = %d'%data.N]
    while restart<restarts:

        try:
            totaltime = time.time()
            print "Restart %d ..."%(restart+1)

            # initialize multi-scale model parameters
            pi = Pi(data.J)
            tau = Tau(data.J)

            # initialize negative binomial parameters
            alpha = Alpha(data.R)
            omega = Omega(data.R)

            # initialize prior parameters
            beta = Beta(scores)

            # initialize posterior over latent variables
            zeta = Zeta(data, totalreads)
            for j in xrange(pi.J):
                pi.value[j] = np.sum(data.value[j][0] * zeta.estim[:,1:],0) \
                    / np.sum(data.total[j][0] * zeta.estim[:,1:],0).astype('float')

            # initial log likelihood of the model
            Loglike = likelihood(data, scores, zeta, pi, tau, \
                    alpha, beta, omega, pi_null, tau_null, model)

            tol = np.inf
            iteration = 0

            while np.abs(tol)>mintol:

                itertime = time.time()
                square_EM(data, scores, zeta, pi, tau, \
                        alpha, beta, omega, pi_null, tau_null, model)

                newLoglike = likelihood(data, scores, zeta, pi, tau, \
                        alpha, beta, omega, pi_null, tau_null, model)

                tol = newLoglike - Loglike
                Loglike = newLoglike
                print "Iteration %d: log likelihood = %.7f, change in log likelihood = %.7f, iteration time = %.3f secs"%(iteration+1, Loglike, tol, time.time()-itertime)
                iteration += 1
            totaltime = (time.time()-totaltime)/60.

            # test if mean cleavage rate at bound sites is greater than at 
            # unbound sites, for each replicate; avoids local optima issues.
            negbinmeans = alpha.estim * (1-omega.estim)/omega.estim
            if np.any(negbinmeans[:,0]<negbinmeans[:,1]):
                restart += 1
                log = "%d. Log likelihood (per site) = %.3f (Completed in %.3f minutes)"%(restart,Loglike,totaltime)
                runlog.append(log)
                # choose these parameter estimates, if the likelihood is greater.
                if Loglike>maxLoglike:
                    maxLoglikeres = Loglike
                    if model in ['msCentipede','msCentipede_flexbgmean']:
                        footprint_model = (pi, tau, pi_null)
                    elif model=='msCentipede_flexbg':
                        footprint_model = (pi, tau, pi_null, tau_null)
                    count_model = (alpha, omega)
                    prior = beta

        except ValueError:

            print "encountered an invalid value"
            if err<5:
                print "re-initializing learning for Restart %d ... %d"%(restart,err)
                err += 1
            else:
                print "Error in learning model parameters. Please ensure the inputs are all valid"
                sys.exit(1)

    return footprint_model, count_model, prior, runlog


def infer_binding_posterior(reads, totalreads, scores, background, footprint, negbinparams, prior, model):
    """Infer posterior probability of factor binding, given optimal model parameters.

    Arguments
        reads : array
        array of read counts at each base in a genomic window,
        across motif instances and several measurement replicates.

        totalreads : array
        array of total read counts in a genomic window,
        across motif instances and several measurement replicates.
        the size of the genomic window can be different for 
        `reads` and `totalreads`.

        scores : array
        an array of scores for each motif instance. these could include
        PWM score, conservation score, a measure of various histone
        modifications, outputs from other algorithms, etc.

        background : array
        a uniform, normalized array for a uniform background model.
        when sequencing reads from genomic DNA are available, this
        is an array of read counts at each base in a genomic window,
        across motif instances.

        footprint : tuple
        (Pi, Tau) instances
        estimate of footprint model parameters

        negbinparams : tuple
        (Alpha, Omega) instances
        estimate of negative binomial model parameters

        prior : Beta
        estimate of weights in logistic function in the prior

        model : string
        {msCentipede, msCentipede-flexbgmean, msCentipede-flexbg}

    """

    data = Data(reads)
    data_null = Data(background)
    scores = np.hstack((np.ones((data.N,1), dtype=float), scores))
    del reads

    # negative binomial parameters
    alpha = negbinparams[0]
    omega = negbinparams[1]
    
    # weights in logistic function in the prior
    beta = prior

    # multiscale parameters
    pi = footprint[0]
    tau = footprint[1]
    
    # setting background model
    pi_null = footprint[2]
    for j in xrange(pi_null.J):
        pi_null.value[j] = np.sum(np.sum(data_null.value[j],0),0) \
            / np.sum(np.sum(data_null.total[j],0),0).astype('float')
    tau_null = None

    if model=='msCentipede_flexbg':

        tau_null = footprint[3]

        if data_null.N>1000:

            zeta_null = Zeta(data_null, background.sum(1))
            zeta_null.estim[:,1] = 1
            zeta_null.estim[:,0] = 0

            # iterative update of background model, when
            # accounting for overdispersion
            change = np.inf
            while change>1e-1:
                change = tau_null.estim.copy()
                
                pi_null.update(data_null, zeta_null, tau_null)

                tau_null.update(data_null, zeta_null, pi_null)

                change = np.abs(change-tau_null.estim).sum()

    zeta = Zeta(data, totalreads, infer=True)

    zeta.infer(data, scores, pi, tau, alpha, beta, omega, \
        pi_null, tau_null, model)
    
    return zeta.posterior_log_odds, \
        zeta.prior_log_odds, zeta.footprint_log_likelihood_ratio, \
        zeta.total_log_likelihood_ratio

