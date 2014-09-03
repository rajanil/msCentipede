import numpy as np
import cvxopt as cvx
from cvxopt import solvers
from scipy.special import digamma, gammaln, polygamma
import time, math, pdb

# defining some constants
EPS = np.finfo(np.double).tiny
MAX = np.finfo(np.double).max

# defining some simple functions
logistic = lambda x: 1./(1+np.exp(x))
insum = lambda x,axes: np.apply_over_axes(np.sum,x,axes)

def outsum(arr):
    """Summation over the first axis, without changing length of shape.

    Arguments
        arr : array

    Returns
        thesum : array

    .. note::
        This implementation is much faster than `numpy.sum`.

    """

    thesum = sum([a for a in arr])
    shape = [1]
    shape.extend(list(thesum.shape))
    thesum = thesum.reshape(tuple(shape))
    return thesum

def nplog(x):
    """Compute the natural logarithm, handling very
    small floats appropriately.

    """
    try:
        x[x<EPS] = EPS
    except TypeError:
        x = max([x,EPS])
    return np.log(x)


class Cascade:

    def __init__(self, reads=None):

        if reads:
            self.N, self.L, self.R = read.shape
            self.J = math.frexp(self.L)[1]-1
            self.value = dict()
            self.total = dict()
            self.transform(reads)
        else:
            self.N = 0
            self.L = 0
            self.R = 0
            self.J = 0
            self.value = dict()
            self.total = dict()

    def transform(self, profile):

        for j in xrange(self.J):
            size = self.L/(2**(j+1))
            self.total[j] = np.array([profile[:,k*size:(k+2)*size,:].sum(1) for k in xrange(0,2**(j+1),2)]).T
            self.value[j] = np.array([profile[:,k*size:(k+1)*size,:].sum(1) for k in xrange(0,2**(j+1),2)]).T

    def inverse_transform(self):

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

        newcopy = Cascade()
        newcopy.J = self.J
        newcopy.N = self.N
        newcopy.L = self.L
        newcopy.R = self.R
        for j in xrange(self.J):
            newcopy.value[j] = self.value[j]
            newcopy.total[j] = self.total[j]

        return newcopy


class Eta():

    def __init__(self, cascade, totalreads, scores, \
        B_null=None, gamma_null=None, omega_null=None, \
        B=None, gamma=None, omega=None, pi=None, \
        beta=None, alpha=None, tau=None, background='multinomial'):

        self.N = cascade.N
        self.total = totalreads
        self.update_count = 0
        self.background = background

        self.estim = np.zeros((self.N, 2),dtype=float)
        if alpha is None:
            indices = np.argsort(self.total.sum(1))[:self.N/2]
            self.estim[indices,1:] = -MAX
            indices = np.argsort(self.total.sum(1))[self.N/2:]
            self.estim[indices,1:] = MAX
        else:
            footprint_logodds = np.zeros((self.N,1), dtype=float)
            if self.background=='multinomial':
                lhoodA, lhoodB, lhoodC = likelihoodAB(cascade, B, omega, B_null, background=self.background)
            else:
                lhoodA, lhoodB, lhoodC = likelihoodAB(cascade, B, omega, B_null, \
                    gamma_null=gamma_null, omega_null=omega_null, background=self.background)

            for j in xrange(pi.J):
                footprint_logodds += insum(gamma.value[j] * lhoodA.value[j] - lhoodC.value[j] \
                    + (1 - gamma.value[j]) * lhoodB.value[j],[1])
                footprint_logodds += np.sum(gamma.value[j] * (nplog(pi.estim[j]) - nplog(gamma.value[j])) \
                    + (1 - gamma.value[j]) * (nplog(1 - pi.estim[j]) - nplog(1 - gamma.value[j]))) * cascade.R

            self.estim = np.zeros((self.N, 4),dtype=float)
            self.estim[:,1:2] = beta.estim[0] + insum(beta.estim[1:] * scores,[1])
            self.estim[:,2:3] = footprint_logodds
            self.estim[:,3:4] = insum(gammaln(self.total + alpha.estim.T[1]) \
                - gammaln(self.total + alpha.estim.T[0]) \
                + gammaln(alpha.estim.T[0]) - gammaln(alpha.estim.T[1]) \
                + alpha.estim.T[1] * nplog(tau.estim.T[1]) - alpha.estim.T[0] * nplog(tau.estim.T[0]) \
                + self.total * (nplog(1 - tau.estim.T[1]) - nplog(1 - tau.estim.T[0])),[1])

            self.estim[:,0] = self.estim[:,1:].sum(1)

        if alpha is None:
            self.estim[self.estim==np.inf] = MAX
            self.estim = np.exp(self.estim - np.max(self.estim,1).reshape(self.N,1))
            self.estim = self.estim / insum(self.estim,[1])
        else:
            self.estim = self.estim / np.log(10)

    def update_Estep(self, cascade, scores, B, pi, gamma, omega, \
        alpha, beta, tau, B_null, gamma_null=None, omega_null=None): 

        footprint_logodds = np.zeros((self.N,1),dtype=float)
        if self.background=='multinomial':
            lhoodA, lhoodB, lhoodC = likelihoodAB(cascade, B, omega, B_null, background=self.background)
        else:
            lhoodA, lhoodB, lhoodC = likelihoodAB(cascade, B, omega, B_null, \
                gamma_null=gamma_null, omega_null=omega_null, background=self.background)

        for j in xrange(pi.J):
                footprint_logodds += insum(gamma.value[j] * lhoodA.value[j] - lhoodC.value[j] \
                    + (1 - gamma.value[j]) * lhoodB.value[j],[1])
                footprint_logodds += np.sum(gamma.value[j] * (nplog(pi.estim[j]) - nplog(gamma.value[j])) \
                    + (1 - gamma.value[j]) * (nplog(1 - pi.estim[j]) - nplog(1 - gamma.value[j]))) * cascade.R

        prior_logodds = beta.estim[0] + beta.estim[1]*scores
        negbin_logodds = insum(gammaln(self.total + alpha.estim.T[1]) \
                - gammaln(self.total + alpha.estim.T[0]) \
                + gammaln(alpha.estim.T[0]) - gammaln(alpha.estim.T[1]) \
                + alpha.estim.T[1] * nplog(tau.estim.T[1]) - alpha.estim.T[0] * nplog(tau.estim.T[0]) \
                + self.total * (nplog(1 - tau.estim.T[1]) - nplog(1 - tau.estim.T[0])),[1])

        self.estim[:,1:] = prior_logodds + footprint_logodds + negbin_logodds
        self.estim[:,0] = 0.
        self.estim[self.estim==np.inf] = MAX
        self.estim = np.exp(self.estim-np.max(self.estim,1).reshape(self.N,1))
        self.estim = self.estim/insum(self.estim,[1])

        if np.isnan(self.estim).any():
            print "Nan in Eta"
            raise ValueError

        if np.isinf(self.estim).any():
            print "Inf in Eta"
            raise ValueError

        self.update_count += 1


class Gamma(Cascade):

    def __init__(self, L, background='multinomial'):

        Cascade.__init__(self, np.random.rand(1,L))
        self.background = background
        self.update_count = 0

    def update_Estep(self, cascade, eta, B, pi, omega, \
        B_null, gamma_null=None, omega_null=None):

        lhoodA, lhoodB, lhoodC = likelihoodAB(cascade, B, omega, \
            B_null, gamma_null=gamma_null, omega_null=omega_null, background=self.background)

        for j in xrange(self.J):
            log_posterior_odds = nplog(pi.estim[j]) - nplog(1 - pi.estim[j]) \
                + outsum(eta.estim[:,1:] * (lhoodA.value[j] - lhoodB.value[j])) \
                / outsum(eta.estim[:,1:]) / cascade.R
            self.value[j] = logistic(-log_posterior_odds)

        self.update_count += 1


class Pi:

    def __init__(self, gamma):

        self.J = gamma.J
        self.estim = np.array([gamma.value[j].sum() / gamma.value[j].size \
            for j in xrange(self.J)])
        self.update_count = 0

    def update_Mstep(self, gamma):

        self.estim = np.array([gamma.value[j].sum() / gamma.value[j].size \
            for j in xrange(self.J)])
        self.update_count += 1


class Bin(Cascade):

    def __init__(self, L):

        Cascade.__init__(self, np.random.rand(1,L))
        self.value = dict([(j,np.random.rand(2**j)) for j in xrange(self.J)])
        self.update_count = 0

    def update(self, cascade, eta, gamma, omega):

        def function(x, kwargs):
            cascade = kwargs['cascade']
            eta = kwargs['eta']
            gamma = kwargs['gamma']
            omega = kwargs['omega']

            F = 0
            for j in xrange(self.J):
                left = 2**j-1
                right = 2**(j+1)-1
                func = np.zeros(cascade.value[j][0].shape, dtype=float)
                for r in xrange(cascade.R):
                    func += gammaln(cascade.value[j][r] + omega.estim[j] * x[left:right]) \
                        + gammaln(cascade.total[j][r] - cascade.value[j][r] + omega.estim[j] * (1 - x[left:right])) \
                        - gammaln(omega.estim[j] * x[left:right]) - gammaln(omega.estim[j] * (1 - x[left:right]))
                F += -1. * np.sum((1 - gamma.value[j]) * np.sum(eta.estim[:,1:] * func,0))
            return F

        def gradient(x, kwargs):
            cascade = kwargs['cascade']
            eta = kwargs['eta']
            gamma = kwargs['gamma']
            omega = kwargs['omega']

            Df = np.zeros(x.shape, dtype=float)
            for j in xrange(self.J):
                left = 2**j-1
                right = 2**(j+1)-1
                df = np.zeros(cascade.value[j][0].shape, dtype=float)
                for r in xrange(cascade.R):
                    df = digamma(cascade.value[j][r] + omega.estim[j] * x[left:right]) \
                    - digamma(cascade.total[j][r] - cascade.value[j][r] + omega.estim[j] * (1 - x[left:right])) \
                    - digamma(omega.estim[j] * x[left:right]) + digamma(omega.estim[j] * (1 - x[left:right]))
                Df[left:right] = -1. * (1 - gamma.value[j]) * omega.estim[j] * np.sum(eta.estim[:,1:] * func,0)
            return Df

        def hessian(x, kwargs):
            cascade = kwargs['cascade']
            eta = kwargs['eta']
            gamma = kwargs['gamma']
            omega = kwargs['omega']

            hess = np.zeros(x.shape, dtype=float)
            for j in xrange(self.J):
                left = 2**j-1
                right = 2**(j+1)-1
                hf = np.zeros(cascade.value[j][0].shape, dtype=float)
                for r in xrange(cascade.R):
                    hf = polygamma(1, cascade.value[j][r] + omega.estim[j] * x[left:right]) \
                    + polygamma(1, cascade.total[j][r] - cascade.value[j][r] + omega.estim[j] * (1 - x[left:right])) \
                    - polygamma(1, omega.estim[j] * x[left:right]) - polygamma(1, omega.estim[j] * (1 - x[left:right]))
                Hf[left:right] = -1. * (1 - gamma.value[j]) * omega.estim[j]**2 * np.sum(eta.estim[:,1:] * func,0)
            
            Hf = np.diag(hess)
            return Hf

        # initialize, and set constraints
        xo = np.array([v for j in xrange(self.J) for v in self.value[j]])
        G = np.vstack((np.diag(-1*np.ones(xo.shape, dtype=float)), \
                np.diag(np.ones(xo.shape, dtype=float))))
        h = np.vstack((np.zeros((xo.size,1), dtype=float), \
                np.ones((xo.size,1), dtype=float)))

        # call optimizer
        x_final = optimizer(xo, function, gradient, hessian, \
            G=G, h=h, cascade=cascade, eta=eta, gamma=gamma, omega=omega)

        if np.isnan(x_final).any():
            print "Nan in Bin"
            raise ValueError

        if np.isinf(x_final).any():
            print "Inf in Bin"
            raise ValueError

        self.value = dict([(j,x_final[2**j-1:2**(j+1)-1]) for j in xrange(self.J)])
        self.update_count += 1


class Omega():

    def __init__(self, J):

        self.J = J
        self.estim = 10*np.random.rand(self.J)
        self.update_count = 0

    def update(self, cascade, eta, gamma, B):

        def function(x, kwargs):
            cascade = kwargs['cascade']
            eta = kwargs['eta']
            gamma = kwargs['gamma']
            B = kwargs['B']

            func = 0.
            # loop over each scale
            for j in xrange(gamma.J):

                f = np.zeros(cascade.value[j].shape, dtype=float)
                # contribution of the ``smooth`` state; loop over replicates
                for r in xrange(cascade.R):
                    f += gammaln(cascade.value[j][r] + 0.5 * x[j]) \
                        + gammaln(cascade.total[j][r] - cascade.value[j][r] + 0.5 * x[j]) \
                        - gammaln(cascade.total[j][r] + x[j]) + gammaln(x[j]) \
                        - 2 * gammaln(0.5 * x[j])
                func += np.sum(gamma.value[j] * np.sum(eta.estim[:,1:] * f,0))

                f = np.zeros(cascade.value[j].shape, dtype=float)
                # contribution of the ``non-smooth`` state; loop over replicates
                for r in xrange(cascade.R):
                    f += gammaln(cascade.value[j][r] + B.value[j] * x[j]) \
                        + gammaln(cascade.total[j][r] - cascade.value[j][r] + (1 - B.value[j]) * x[j]) \
                        - gammaln(cascade.total[j][r] + x[j]) + gammaln(x[j]) \
                        - gammaln(B.value[j] * x[j]) - gammaln((1 - B.value[j]) * x[j])
                func += np.sum((1 - gamma.value[j]) * np.sum(eta.estim[:,1:] * f,0))

            F = -1.*func.sum()
            return F

        def gradient(x, kwargs):
            cascade = kwargs['cascade']
            eta = kwargs['eta']
            gamma = kwargs['gamma']
            B = kwargs['B']

            Df = np.zeros(x.shape, dtype=float)
            # loop over each scale
            for j in xrange(gamma.J):

                f = np.zeros(cascade.value[j].shape, dtype=float)
                # contribution of the ``smooth`` state; loop over replicates
                for r in xrange(cascade.R):
                    f += 0.5 * digamma(cascade.value[j][r] + 0.5 * x[j]) \
                        + 0.5 * digamma(cascade.total[j][r] - cascade.value[j][r] + 0.5 * x[j]) \
                        - digamma(cascade.total[j][r] + x[j]) + digamma(x[j]) \
                        - digamma(0.5 * x[j])
                Df[j] = -1 * np.sum(gamma.value[j] * np.sum(eta.estim[:,1:] * f,0))

                f = np.zeros(cascade.value[j].shape, dtype=float)
                # contribution of the ``non-smooth`` state; loop over replicates
                for r in xrange(cascade.R):
                    f += B.value[j] * digamma(cascade.value[j][r] + B.value[j] * x[j]) \
                        + (1 - B.value[j]) * digamma(cascade.total[j][r] - cascade.value[j][r] + (1 - B.value[j]) * x[j]) \
                        - digamma(cascade.total[j][r] + x[j]) + digamma(x[j]) \
                        - B.value[j] * digamma(B.value[j] * x[j]) - (1 - B.value[j]) * digamma((1 - B.value[j]) * x[j])
                Df[j] += -1 * np.sum((1 - gamma.value[j]) * np.sum(eta.estim[:,1:] * f,0))

            return Df

        def hessian(x, kwargs):
            cascade = kwargs['cascade']
            eta = kwargs['eta']
            gamma = kwargs['gamma']
            B = kwargs['B']

            hess = np.zeros(x.shape, dtype=float)
            for j in xrange(gamma.J):

                f = np.zeros(cascade.value[j].shape, dtype=float)
                # contribution of the ``smooth`` state; loop over replicates
                for r in xrange(cascade.R):
                    f += 0.25 * polygamma(1, cascade.value[j][r] + 0.5 * x[j]) \
                        + 0.25 * polygamma(1, cascade.total[j][r] - cascade.value[j][r] + 0.5 * x[j]) \
                        - polygamma(1, cascade.total[j][r] + x[j]) + polygamma(1, x[j]) \
                        - 0.5 * polygamma(1, 0.5 * x[j])
                hess[j] = -1 * np.sum(gamma.value[j] * np.sum(eta.estim[:,1:] * f,0))

                f = np.zeros(cascade.value[j].shape, dtype=float)
                # contribution of the ``non-smooth`` state; loop over replicates
                for r in xrange(cascade.R):
                    f += B.value[j]**2 * polygamma(1, cascade.value[j][r] + B.value[j] * x[j]) \
                        + (1 - B.value[j])**2 * polygamma(1, cascade.total[j][r] - cascade.value[j][r] + (1 - B.value[j]) * x[j]) \
                        - polygamma(1, cascade.total[j][r] + x[j]) + polygamma(1, x[j]) \
                        - B.value[j]**2 * polygamma(1, B.value[j] * x[j]) \
                        - (1 - B.value[j])**2 * polygamma(1, (1 - B.value[j]) * x[j])
                hess[j] += -1 * np.sum((1 - gamma.value[j]) * np.sum(eta.estim[:,1:] * f,0))

            Hf = np.diag(hess)
            return Hf

        # set constraints
        G = np.diag(-1 * np.ones(xo.shape, dtype=float))
        h = np.zeros((xo.size,1), dtype=float)

        # call optimizer
        xo = self.estim.copy()
        x_final = optimizer(xo, function, gradient, hessian, \
            G=G, h=h, cascade=cascade, eta=eta, gamma=gamma, B=B)
        self.estim = x_final.reshape(self.estim.shape)

        if np.isnan(self.estim).any():
            print "Nan in Omega"
            raise ValueError

        if np.isinf(self.estim).any():
            print "Inf in Omega"
            raise ValueError

        self.update_count += 1


class Alpha():

    def __init__(self, R=1):

        self.R = R
        self.estim = np.random.rand(self.R,2)*10
        self.update_count = 0

    def update(self, eta, tau):

        def function(x, kwargs):
            eta = kwargs['eta']
            tau = kwargs['tau']
            constant = kwargs['constant']

            etaestim = outsum(eta.estim)
            func = np.array([outsum(gammaln(eta.total[:,r:r+1] + x[2*r:2*r+2]) * eta.estim) \
                    - gammaln(x[2*r:2*r+2]) * etaestim[0] + constant[r] * x[2*r:2*r+2] \
                    for r in xrange(tau.R)])
            f = -1.*func.sum()
            return f

        def gradient(x, kwargs):
            eta = kwargs['eta']
            tau = kwargs['tau']
            etaestim = kwargs['etaestim']
            constant = kwargs['constant']

            df = []
            for r in xrange(tau.R):
                df.append(outsum(digamma(eta.total[:,r:r+1] + x[2*r:2*r+2]) * eta.estim)[0] \
                    - digamma(x[2*r:2*r+2]) * etaestim[0] + constant[r])
            Df = -1. * np.hstack(df)
            return Df

        def hessian(x, kwargs):
            eta = kwargs['eta']
            tau = kwargs['tau']
            etaestim = kwargs['etaestim']
            constant = kwargs['constant']
            
            hess = []
            for r in xrange(tau.R):
                hess.append(outsum(polygamma(1, eta.total[:,r:r+1] + x[2*r:2*r+2]) * eta.estim)[0] \
                    - polygamma(1, x[2*r:2*r+2]) * etaestim[0])
            Hf = -1. * np.diag(np.hstack(hess))
            return Hf

        const = [nplog(tau.estim[r]) * outsum(eta.estim)[0] for r in xrange(self.R)]
        etaestim = outsum(eta.estim)
        xo = self.estim.ravel()

        # set constraints
        G = np.diag(-1 * np.ones(xo.shape, dtype=float))
        h = np.zeros((xo.size,1), dtype=float)

        # call optimizer
        x_final = optimizer(xo, function, gradient, hessian, \
            G=G, h=h, tau=tau, eta=eta, constant=constant, etaestim=etaestim)
        self.estim = x_final.reshape(self.estim.shape)

        if np.isnan(self.estim).any():
            print "Nan in Alpha"
            raise ValueError

        if np.isinf(self.estim).any():
            print "Inf in Alpha"
            raise ValueError

        self.update_count += 1


class Tau():

    def __init__(self, R=1):

        self.R = R
        self.estim = np.random.rand(self.R,2)
        self.estim[:,1] = self.estim[:,1]/100
        self.update_count = 0

    def update(self, eta, alpha):

        numerator = outsum(eta.estim)[0] * alpha.estim
        denominator = np.array([outsum(eta.estim * (estim + eta.total[:,r:r+1]))[0] \
            for r,estim in enumerate(alpha.estim)])
        self.estim = numerator / denominator

        if np.isnan(self.estim).any():
            print "Nan in Tau"
            raise ValueError

        if np.isinf(self.estim).any():
            print "Inf in Tau"
            raise ValueError

        self.update_count += 1


class Beta():

    def __init__(self, S):
    
        self.S = S
        self.estim = np.random.rand(self.S)
        self.update_count = 0

    def update(self, scores, eta):

        def function(x, kwargs):
            scores = kwargs['scores']
            eta = kwargs['eta']

            arg = insum(x * scores,[1])
            func = arg * eta.estim[:,1:] - nplog(1 + np.exp(arg))
            f = -1. * func.sum()
            return f

        def gradient(x, kwargs):
            scores = kwargs['scores']
            eta = kwargs['eta']

            arg = insum(x * scores,[1])
            Df = -1 * np.sum(scores * (eta.estim[:,1:] - logistic(-arg)),0)
            return Df

        def hessian(x, kwargs):
            scores = kwargs['scores']
            eta = kwargs['eta']

            arg = insum(x * scores,[1])
            larg = scores * logistic(arg) * logistic(-arg)
            Hf = np.dot(scores.T, larg)
            return Hf


        xo = self.estim.copy()
        self.estim = optimizer(xo, function, gradient, hessian, scores=scores, eta=eta)

        if np.isnan(self.estim).any():
            print "Nan in Beta"
            raise ValueError

        if np.isinf(self.estim).any():
            print "Inf in Beta"
            raise ValueError

        self.update_count += 1        


def optimizer(xo, function, gradient, hessian, **kwargs):

    def F(x=None, z=None):

        if x is None:
            return 0, cvx.matrix(x_init)

        xx = np.array(x).ravel().astype(np.float64)

        # compute likelihood function
        f = function(xx, kwargs)
        if np.isnan(f) or np.isinf(f):
            f = np.array([np.finfo('float32').max]).astype('float')
        else:
            f = np.array([f]).astype('float')
        
        # compute gradient
        Df = gradient(xx, kwargs)
        if np.isnan(Df).any() or np.isinf(Df).any():
            Df = -1 * np.finfo('float32').max * np.ones((1,xx.size), dtype=float)
        if z is None:
            return cvx.matrix(f), cvx.matrix(Df)

        # compute hessian
        hess = hessian(xx, kwargs)
        Hf = z[0] * hess
        return cvx.matrix(f), cvx.matrix(Df), cvx.matrix(Hf)

    # run the optimizer
    optimized = False
    V = xo.size
    # warm start
    x_init = xo.reshape(V,1)
    while not optimized:
        try:
            if kwargs.has_key('G'):
                solution = solvers.cp(F, G=kwargs['G'], h=kwargs['h'])
            else:
                solution = solvers.cp(F)
            if solution['status']=='optimal':
                optimized = True
                x_final = np.array(solution['x']).ravel()
            else:
                # cold start
                x_init = np.random.rand(V,1)
        except ValueError:
            # cold start
            x_init = np.random.rand(V,1)

    return x_final


def likelihoodAB(cascade, B, omega, B_null, gamma_null=None, omega_null=None, background='multinomial'):

    lhoodA = Cascade()
    lhoodB = Cascade()
    lhoodC = Cascade()
    lhoodCa = Cascade()
    lhoodCb = Cascade()

    for j in xrange(cascade.J):
        value = outsum(cascade.value[j])[0]
        total = outsum(cascade.total[j])[0]
            
        if background=='multinomial':

            lhoodA.value[j] = outsum([gammaln(cascade.value[j][r] + 0.5 * omega.estim[j]) \
                + gammaln(cascade.total[j][r] - cascade.value[j][r] + 0.5 * omega.estim[j]) \
                - gammaln(cascade.total[j][r] + omega.estim[j]) + gammaln(omega.estim[j]) \
                - 2 * gammaln(0.5 * omega.estim[j]) for r in xrange(cascade.R)])[0]
            
            lhoodB.value[j] = outsum([gammaln(cascade.value[j][r] + B.value[j] * omega.estim[j]) \
                + gammaln(cascade.total[j][r] - cascade.value[j][r] + (1 - B.value[j]) * omega.estim[j]) \
                - gammaln(cascade.total[j][r] + omega.estim[j]) + gammaln(omega.estim[j]) \
                - gammaln(B.value[j] * omega.estim[j]) - gammaln((1 - B.value[j]) * omega.estim[j]) \
                for r in xrange(cascade.R)])[0]
            
            lhoodC.value[j] = value * nplog(B_null.value[j]) \
                + (total - value) * nplog(1 - B_null.value[j])
    
        else:

            lhoodA.value[j] = outsum([gammaln(cascade.value[j][r] + 0.5 * omega.estim[j]) \
                + gammaln(cascade.total[j][r] - cascade.value[j][r] + 0.5 * omega.estim[j]) \
                - gammaln(cascade.total[j][r] + omega.estim[j]) + gammaln(omega.estim[j]) \
                - 2 * gammaln(0.5 * omega.estim[j]) for r in xrange(cascade.R)])[0]
            
            lhoodB.value[j] = outsum([gammaln(cascade.value[j][r] + B.value[j] * omega.estim[j]) \
                + gammaln(cascade.total[j][r] - cascade.value[j][r] + (1 - B.value[j]) * omega.estim[j]) \
                - gammaln(cascade.total[j][r] + omega.estim[j]) + gammaln(omega.estim[j]) \
                - gammaln(B.value[j] * omega.estim[j]) - gammaln((1 - B.value[j]) * omega.estim[j]) \
                for r in xrange(cascade.R)])[0]

            lhoodCa.value[j] = outsum([gammaln(cascade.value[j][r] + 0.5 * omega_null.estim[j]) \
                + gammaln(cascade.total[j][r] - cascade.value[j][r] + 0.5 * omega_null.estim[j]) \
                - gammaln(cascade.total[j][r] + omega_null.estim[j]) + gammaln(omega_null.estim[j]) \
                - 2 * gammaln(0.5 * omega_null.estim[j]) for r in xrange(cascade.R)])[0]
            
            lhoodCb.value[j] = outsum([gammaln(cascade.value[j][r] + B_null.value[j] * omega_null.estim[j]) \
                + gammaln(cascade.total[j][r] - cascade.value[j][r] + (1 - B_null.value[j]) * omega_null.estim[j]) \
                - gammaln(cascade.total[j][r] + omega_null.estim[j]) + gammaln(omega_null.estim[j]) \
                - gammaln(B_null.value[j] * omega_null.estim[j]) - gammaln((1 - B_null.value[j]) * omega_null.estim[j]) \
                for r in xrange(cascade.R)])[0]
            
            lhoodC.value[j] = gamma_null.value[j] * lhoodCa.value[j] + (1 - gamma_null.value[j]) * lhoodCb.value[j]

    return lhoodA, lhoodB, lhoodC


def likelihood(cascade, scores, eta, B, pi, gamma, omega, \
    alpha, beta, tau, B_null, gamma_null=None, omega_null=None, background='multinomial'):

    apriori = insum(beta.estim * scores,[1])

    if background=='multinomial':
        lhoodA, lhoodB, lhoodC = likelihoodAB(cascade, B, omega, B_null, background=background)
    else:
        lhoodA, lhoodB, lhoodC = likelihoodAB(cascade, B, omega, B_null, \
            gamma_null=gamma_null, omega_null=omega_null, background=background)

    footprint = np.zeros((cascade.N,1),dtype=float)
    for j in xrange(pi.J):
        footprint += insum(gamma.value[j] * lhoodA.value[j] + (1 - gamma.value[j]) * lhoodB.value[j] \
            + cascade.R * (gamma.value[j] * (nplog(pi.estim[j]) - nplog(gamma.value[j])) \
            + (1 - gamma.value[j]) * (nplog(1 - pi.estim[j]) - nplog(1 - gamma.value[j]))),[1])

    P_1 = footprint + insum(gammaln(eta.total + alpha.estim[:,1]) - gammaln(alpha.estim[:,1]) \
        + alpha.estim[:,1] * nplog(tau.estim[:,1]) + eta.total * nplog(1 - tau.estim[:,1]), [1])
    P_1[P_1==np.inf] = MAX
    P_1[P_1==-np.inf] = -MAX

    null = np.zeros((cascade.N,1), dtype=float)
    for j in xrange(cascade.J):
        null = null + insum(lhoodC.value[j],[1])

    P_0 = null + insum(gammaln(eta.total + alpha.estim[:,0]) - gammaln(alpha.estim[:,0]) \
        + alpha.estim[:,0] * nplog(tau.estim[:,0]) + eta.total * nplog(1 - tau.estim[:,0]), [1])
    P_0[P_0==np.inf] = MAX
    P_0[P_0==-np.inf] = -MAX

    L = P_0 * eta.estim[:,:1] + insum(P_1 * eta.estim[:,1:],[1]) + apriori * (1 - eta.estim[:,:1]) \
        - nplog(1 + np.exp(apriori)) - insum(eta.estim * nplog(eta.estim),[1])
    
    L = L.sum()

    if np.isnan(L):
        print "Nan in LogLike"
        return -np.inf

    if np.isinf(L):
        print "Inf in LogLike"
        return -np.inf

    return L

# need to re-write
def bayes_optimal_estimator(cascade, eta, pi, B=None, mu=None, omega=None, gamma_null=None, B_null=None, omega_null=None, model='modelA'):
    """
    computes the posterior mean conditional on the most likely
    set of states for gamma.
    """

    M1 = Cascade(cascade.L)
    M2 = Cascade(cascade.L)
    if isinstance(eta, Eta):
        states = eta.estim[:,1:]>0.5
    else:
        states = eta[:,1:]

    if not isinstance(pi, Pi):
        gamma = Gamma(cascade.L)
        pitmp = Pi(gamma)
        pitmp.estim = pi
        pi = pitmp

    if model=='modelA':
        for j in range(pi.J):
            ratio = nplog(1-pi.estim[j]) - nplog(pi.estim[j]) + (cascade.value[j]*states).sum(0)*nplog(B.value[j]) \
                + ((cascade.total[j]-cascade.value[j])*states).sum(0)*nplog(1-B.value[j]) \
                - (cascade.total[j]*states).sum(0)*nplog(B_null.value[j])
            M1.value[j] = 0.5*logistic(ratio) + B.value[j]*logistic(-ratio)
            M2.value[j] = 0.25*logistic(ratio) + B.value[j]**2*logistic(-ratio)

    elif model=='modelB':
        for j in range(pi.J):
            partratio = outsum([gammaln(cascade.value[j][r]+mu.estim[j]) + gammaln(cascade.total[j][r]-cascade.value[j][r]+mu.estim[j]) \
                - gammaln(cascade.total[j][r]+2*mu.estim[j]) + gammaln(2*mu.estim[j]) - 2*gammaln(mu.estim[j]) \
                - cascade.total[j][r]*nplog(B_null.value[j]) for r in xrange(cascade.R)])[0]
            ratio = nplog(1-pi.estim[j]) - nplog(pi.estim[j]) + np.sum(states*partratio,0)
            skew = (mu.estim[j]+np.sum([cascade.value[j][r] for r in xrange(cascade.R)],0).sum(0))/(2*mu.estim[j]+np.sum([cascade.total[j][r] for r in xrange(cascade.R)],0).sum(0))
            M1.value[j] = 0.5*logistic(ratio) + skew*logistic(-ratio)
            M2.value[j] = 0.25*logistic(ratio) + skew**2*logistic(-ratio)

    elif model in ['modelC','modelD','modelE','modelF']:
        for j in range(pi.J):
            partratio = outsum([gammaln(cascade.value[j][r]+B.value[j]*omega.estim[j]) + gammaln(cascade.total[j][r]-cascade.value[j][r]+(1-B.value[j])*omega.estim[j]) \
                - gammaln(cascade.value[j][r]+0.5*omega.estim[j]) - gammaln(cascade.total[j][r]-cascade.value[j][r]+0.5*omega.estim[j]) \
                + 2*gammaln(0.5*omega.estim[j]) \
                - gammaln(B.value[j]*omega.estim[j]) - gammaln((1-B.value[j])*omega.estim[j]) for r in xrange(cascade.R)])[0]
            ratio = nplog(1-pi.estim[j]) - nplog(pi.estim[j]) + np.sum(states*partratio,0)
            M1.value[j] = 0.5*logistic(ratio) + B.value[j]*logistic(-ratio)
            M2.value[j] = 0.25*logistic(ratio) + B.value[j]**2*logistic(-ratio)

    return M1, M2


def EM(cascade, scores, eta, B, pi, gamma, omega, B_null, \
    gamma_null=None, omega_null=None, background='multinomial'):

    # update binding posteriors
    if background=='multinomial':
        eta.update(cascade, scores, B, pi, gamma, omega, \
            alpha, beta, tau, B_null)
    else:
        eta.update(cascade, scores, B, pi, gamma, omega, \
            alpha, beta, tau, B_null, gamma_null=gamma_null, omega_null=omega_null)

    # update multi-scale latent variables
    if background=='multinomial':
        gamma.update(cascade, eta, B, pi, omega, B_null)
    else:
        gamma.update(cascade, eta, B, pi, omega, B_null, \
            gamma_null=gamma_null, omega_null=omega_null)

    # update multi-scale parameters
    pi.update(gamma)
    B.update(cascade, eta, gamma, omega)

    # update negative binomial parameters
    tau.update(eta, alpha)
    alpha.update(eta, tau)

    # update prior parameters
    beta.update(scores, eta)


def infer(reads, totalreads, scores, background, background_model='multinomial', restarts=3, mintol=1e-2):

    (N,L,R) = reads.shape
    cascade = Cascade(reads)
    cascade_null = Cascade(background)
    del reads

    # set background model
    B_null = Bin(L, background=background_model)
    if background=='multinomial':
        
        for j in xrange(B_null.J):
            B_null.value[j] = np.sum(cascade_null.value[j],1) / np.sum(cascade_null.total[j],1).astype('float')

    else:
        
        Bg = Bin(L, background=background_model)
        Bg.value = dict([(j,0.5*np.ones((2**j,))) for j in xrange(Bg.J)])
        gamma_null = Gamma(L, background=background_model)
        pi_null = Pi(gamma_null)
        omega_null = Omega(cascade_null.J, background=background_model)

        eta_null = Eta(cascade_null, totalreads, scores)
        eta_null.estim[:,1] = 1
        eta_null.estim[:,0] = 0

        # iterative update of background model; 
        # evaluate convergence based on change in estimated
        # background overdispersion
        change = np.inf
        while change>1e-1:
            change = omega_null.estim.copy()

            gamma_null.update_Estep(cascade_null, eta_null, B_null, pi_null, omega_null, Bg)

            pi_null.update_Mstep(gamma_null)
            
            B_null.update_Mstep(cascade_null, eta_null, gamma_null, omega_null)

            omega_null.update_Mstep(cascade_null, eta_null, gamma_null, B_null)

            change = np.abs(change-omega_null.estim).sum()

    maxLoglike = -np.inf
    restart = 0
    while restart<restarts:

        try:
            # initialize multi-scale model parameters
            gamma = Gamma(L, background=background_model)
            pi = Pi(gamma)
            B = Bin(L, background=background_model)
            omega = Omega(cascade.J, background=background_model)

            # initialize negative binomial parameters
            alpha = Alpha(R=R)
            tau = Tau(R=R)

            # initialize prior parameters
            beta = Beta()

            # initialize posterior over latent variables
            eta = Eta(cascade, totalreads, scores)

            if background=='multinomial':
                Loglike = likelihood(cascade, scores, eta, B, pi, gamma, omega, \
                    alpha, beta, tau, B_null)
            else:
                Loglike = likelihood(cascade, scores, eta, B, pi, gamma, omega, \
                    alpha, beta, tau, B_null, gamma_null=gamma_null, omega_null=omega_null)

            tol = np.inf
            iter = 0
            itertime = time.time()

            while np.abs(tol)>mintol:

                if background=='multinomial':
                    EM(cascade, scores, eta, B, pi, gamma, omega, \
                        B_null, background='multinomial')
                else:
                    EM(cascade, scores, eta, B, pi, gamma, omega, \
                        B_null, gamma_null=gamma_null, omega_null=None, background='multinomial')

                # compute likelihood every 10 iterations
                if (iter+1)%10==0:

                    if background=='multinomial':
                        newLoglike = likelihood(cascade, scores, eta, B, pi, gamma, omega, \
                            alpha, beta, tau, B_null)
                    else:
                        newLoglike = likelihood(cascade, scores, eta, B, pi, gamma, omega, \
                            alpha, beta, tau, B_null, gamma_null=gamma_null, omega_null=omega_null)

                    tol = newLoglike - Loglike
                    Loglike = newLoglike
                    print iter+1, newLoglike, tol, time.time()-itertime
                    itertime = time.time()

                iter += 1

            # test if bound sites have more total reads than unbound sites;
            # avoids local optima issues.
            negbinmeans = alpha.estim*(1-tau.estim)/tau.estim
            if np.any(negbinmeans[:,0]<negbinmeans[:,1]):
                restart += 1
                if Loglike>maxLoglike:
                    maxLoglikeres = Loglike
                    if background=='multinomial':
                        footprint = (B, pi, gamma, omega, B_null)
                    elif model=='modelE':
                        footprint = (B, pi, gamma, omega, B_null, pi_null, gamma_null, omega_null)
                    count_model = (alpha, tau)
                    prior = beta

        except ValueError as err:

            print "restarting inference"

    return footprint_model, count_model, prior


def decode(reads, totalreads, scores, background, footprint, negbinparams, prior, background_model='multinomial'):

    (N,L,R) = reads.shape
    cascade = Cascade(reads)
    cascade_null = Cascade(background)
    del reads

    # negative binomial parameters
    alpha = negbinparams[0]
    tau = negbinparams[1]
    beta = prior

    # multiscale parameters
    B = footprint[0]
    pi = footprint[1]
    gamma = footprint[2]
    omega = footprint[3]
    
    # setting background model
    B_null = footprint[4]

    if background_model=='multinomial':

        for j in xrange(B_null.J):
            B_null.value[j] = np.sum(cascade_null.value[j],1) / np.sum(cascade_null.total[j],1).astype('float')
    
    else:

        gamma_null = footprint[5]
        pi_null = footprint[6]
        omega_null = footprint[7]

        Bg = Bin(L, background=background_model)
        Bg.value = dict([(j,0.5*np.ones((2**j,))) for j in xrange(Bg.J)])

        eta_null = Eta(cascade_null, totalreads, scores)
        eta_null.estim[:,1] = 1
        eta_null.estim[:,0] = 0

        # iterative update of background model, when
        # accounting for overdispersion
        change = np.inf
        while change>1e-1:
            change = omega_null.estim.copy()

            gamma_null.update_Estep(cascade_null, eta_null, B_null, pi_null, omega_null, Bg)

            pi_null.update_Mstep(gamma_null)
            
            B_null.update_Mstep(cascade_null, eta_null, gamma_null, omega_null)

            omega_null.update_Mstep(cascade_null, eta_null, gamma_null, B_null)

            change = np.abs(change-omega_null.estim).sum()

    if background=='multinomial':

        eta = Eta(cascade, totalreads, scores, \
            B=B, pi=pi, gamma=gamma, omega=omega, B_null=B_null, \
            beta=beta, alpha=alpha, tau=tau)
    
    else:
    
        eta = Eta(cascade, totalreads, scores, \
            B=B, pi=pi, gamma=gamma, omega=omega, \
            B_null=B_null, gamma_null=gamma_null, omega_null=omega_null, \
            beta=beta, alpha=alpha, tau=tau)

    return eta.estim
