import numpy as np
import cvxopt as cvx
from cvxopt import solvers
from scipy.special import digamma, gammaln, polygamma
import time, math, pdb

# suppress optimizer output
solvers.options['show_progress'] = False

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

        if reads is None:
            self.N = 0
            self.L = 0
            self.R = 0
            self.J = 0
            self.value = dict()
            self.total = dict()
        else:
            self.N, self.L, self.R = reads.shape
            self.J = math.frexp(self.L)[1]-1
            self.value = dict()
            self.total = dict()
            self.transform(reads)

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

    def __init__(self, cascade, totalreads, scores=None, \
        B_null=None, omega_null=None, B=None, omega=None, \
        beta=None, alpha=None, tau=None, model='msCentipede'):

        self.N = cascade.N
        self.total = totalreads
        self.update_count = 0

        self.estim = np.zeros((self.N, 2),dtype=float)
        if alpha is None:
            indices = np.argsort(self.total.sum(1))[:self.N/2]
            self.estim[indices,1:] = -MAX
            indices = np.argsort(self.total.sum(1))[self.N/2:]
            self.estim[indices,1:] = MAX
        else:
            footprint_logodds = np.zeros((self.N,1), dtype=float)
            lhoodA, lhoodB = likelihoodAB(cascade, B, omega, B_null, omega_null, model)

            for j in xrange(cascade.J):
                footprint_logodds += insum(lhoodA.value[j] - lhoodB.value[j],[1])

            self.estim = np.zeros((self.N, 4),dtype=float)
            self.estim[:,1:2] = insum(beta.estim * scores, [1])
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

    def update(self, cascade, scores, \
        B, omega, alpha, beta, tau, \
        B_null, omega_null, model):

        footprint_logodds = np.zeros((self.N,1),dtype=float)
        lhoodA, lhoodB = likelihoodAB(cascade, B, omega, B_null, omega_null, model)

        for j in xrange(cascade.J):
            footprint_logodds += insum(lhoodA.value[j] - lhoodB.value[j],[1])

        prior_logodds = insum(beta.estim * scores, [1])
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


class Bin(Cascade):

    def __init__(self, J):

        Cascade.__init__(self)
        self.J = J
        for j in xrange(self.J):
            self.value[j] = np.random.rand(2**j)
        self.update_count = 0

    def update(self, cascade, eta, omega):

        def function(x, kwargs):
            cascade = kwargs['cascade']
            eta = kwargs['eta']
            omega = kwargs['omega']

            F = np.zeros(eta.estim[:,1].shape, dtype=float)
            for j in xrange(self.J):
                left = 2**j-1
                right = 2**(j+1)-1
                func = np.zeros(cascade.value[j][0].shape, dtype=float)
                for r in xrange(cascade.R):
                    func += gammaln(cascade.value[j][r] + omega.estim[j] * x[left:right]) \
                        + gammaln(cascade.total[j][r] - cascade.value[j][r] + omega.estim[j] * (1 - x[left:right])) \
                        - gammaln(omega.estim[j] * x[left:right]) - gammaln(omega.estim[j] * (1 - x[left:right]))
                F += np.sum(func,1)
            f = -1. * np.sum(eta.estim[:,1] * F)
            return f

        def gradient(x, kwargs):
            cascade = kwargs['cascade']
            eta = kwargs['eta']
            omega = kwargs['omega']

            Df = np.zeros(x.shape, dtype=float)
            for j in xrange(self.J):
                left = 2**j-1
                right = 2**(j+1)-1
                df = np.zeros(cascade.value[j][0].shape, dtype=float)
                for r in xrange(cascade.R):
                    df += digamma(cascade.value[j][r] + omega.estim[j] * x[left:right]) \
                    - digamma(cascade.total[j][r] - cascade.value[j][r] + omega.estim[j] * (1 - x[left:right])) \
                    - digamma(omega.estim[j] * x[left:right]) + digamma(omega.estim[j] * (1 - x[left:right]))
                Df[left:right] = -1. * omega.estim[j] * np.sum(eta.estim[:,1:] * df,0)
            return Df

        def hessian(x, kwargs):
            cascade = kwargs['cascade']
            eta = kwargs['eta']
            omega = kwargs['omega']

            hess = np.zeros(x.shape, dtype=float)
            for j in xrange(self.J):
                left = 2**j-1
                right = 2**(j+1)-1
                hf = np.zeros(cascade.value[j][0].shape, dtype=float)
                for r in xrange(cascade.R):
                    hf += polygamma(1, cascade.value[j][r] + omega.estim[j] * x[left:right]) \
                    + polygamma(1, cascade.total[j][r] - cascade.value[j][r] + omega.estim[j] * (1 - x[left:right])) \
                    - polygamma(1, omega.estim[j] * x[left:right]) - polygamma(1, omega.estim[j] * (1 - x[left:right]))
                hess[left:right] = -1. * omega.estim[j]**2 * np.sum(eta.estim[:,1:] * hf,0)
            
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
            G=G, h=h, cascade=cascade, eta=eta, omega=omega)

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

    def update(self, cascade, eta, B):

        def function(x, kwargs):
            cascade = kwargs['cascade']
            eta = kwargs['eta']
            B = kwargs['B']

            func = np.zeros(eta.estim[:,1].shape, dtype=float)
            # loop over each scale
            for j in xrange(self.J):

                # loop over replicates
                for r in xrange(cascade.R):
                    F = gammaln(cascade.value[j][r] + B.value[j] * x[j]) \
                        + gammaln(cascade.total[j][r] - cascade.value[j][r] + (1 - B.value[j]) * x[j]) \
                        - gammaln(cascade.total[j][r] + x[j]) + gammaln(x[j]) \
                        - gammaln(B.value[j] * x[j]) - gammaln((1 - B.value[j]) * x[j])
                    func += np.sum(F, 1)

            F = -1. * np.sum(eta.estim[:,1] * func)
            return F

        def gradient(x, kwargs):
            cascade = kwargs['cascade']
            eta = kwargs['eta']
            B = kwargs['B']

            Df = np.zeros(x.shape, dtype=float)
            # loop over each scale
            for j in xrange(self.J):

                # loop over replicates
                df = np.zeros(eta.estim[:,1].shape, dtype=float)
                for r in xrange(cascade.R):
                    f = B.value[j] * digamma(cascade.value[j][r] + B.value[j] * x[j]) \
                        + (1 - B.value[j]) * digamma(cascade.total[j][r] - cascade.value[j][r] + (1 - B.value[j]) * x[j]) \
                        - digamma(cascade.total[j][r] + x[j]) + digamma(x[j]) \
                        - B.value[j] * digamma(B.value[j] * x[j]) - (1 - B.value[j]) * digamma((1 - B.value[j]) * x[j])
                    df += np.sum(f, 1)
                Df[j] = -1 * np.sum(eta.estim[:,1] * df)

            return Df

        def hessian(x, kwargs):
            cascade = kwargs['cascade']
            eta = kwargs['eta']
            B = kwargs['B']

            hess = np.zeros(x.shape, dtype=float)
            for j in xrange(self.J):

                # loop over replicates
                hf = np.zeros(eta.estim[:,1].shape, dtype=float)
                for r in xrange(cascade.R):
                    f = B.value[j]**2 * polygamma(1, cascade.value[j][r] + B.value[j] * x[j]) \
                        + (1 - B.value[j])**2 * polygamma(1, cascade.total[j][r] - cascade.value[j][r] + (1 - B.value[j]) * x[j]) \
                        - polygamma(1, cascade.total[j][r] + x[j]) + polygamma(1, x[j]) \
                        - B.value[j]**2 * polygamma(1, B.value[j] * x[j]) \
                        - (1 - B.value[j])**2 * polygamma(1, (1 - B.value[j]) * x[j])
                    hf += np.sum(f, 1)
                hess[j] = -1 * np.sum(eta.estim[:,1] * hf)

            Hf = np.diag(hess)
            return Hf

        # set constraints
        xo = self.estim.copy()
        G = np.diag(-1 * np.ones(xo.shape, dtype=float))
        h = np.zeros((xo.size,1), dtype=float)

        # call optimizer
        x_final = optimizer(xo, function, gradient, hessian, \
            G=G, h=h, cascade=cascade, eta=eta, B=B)
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
            etaestim = kwargs['etaestim']

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

        constant = [nplog(tau.estim[r]) * outsum(eta.estim)[0] for r in xrange(self.R)]
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

    def __init__(self, scores):
    
        self.S = scores.shape[1]
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
        else:
            Df = Df.reshape(1,xx.size)
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
                solution = solvers.cp(F, G=cvx.matrix(kwargs['G']), h=cvx.matrix(kwargs['h']))
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


def likelihoodAB(cascade, B, omega, B_null, omega_null, model):

    lhoodA = Cascade()
    lhoodB = Cascade()

    for j in xrange(cascade.J):
        value = outsum(cascade.value[j])[0]
        total = outsum(cascade.total[j])[0]
            
        if model in ['msCentipede','msCentipede_flexbgmean']:
            
            lhoodA.value[j] = outsum([gammaln(cascade.value[j][r] + B.value[j] * omega.estim[j]) \
                + gammaln(cascade.total[j][r] - cascade.value[j][r] + (1 - B.value[j]) * omega.estim[j]) \
                - gammaln(cascade.total[j][r] + omega.estim[j]) + gammaln(omega.estim[j]) \
                - gammaln(B.value[j] * omega.estim[j]) - gammaln((1 - B.value[j]) * omega.estim[j]) \
                for r in xrange(cascade.R)])[0]
            
            lhoodB.value[j] = value * nplog(B_null.value[j]) \
                + (total - value) * nplog(1 - B_null.value[j])
    
        elif model=='msCentipede_flexbg':
            
            lhoodA.value[j] = outsum([gammaln(cascade.value[j][r] + B.value[j] * omega.estim[j]) \
                + gammaln(cascade.total[j][r] - cascade.value[j][r] + (1 - B.value[j]) * omega.estim[j]) \
                - gammaln(cascade.total[j][r] + omega.estim[j]) + gammaln(omega.estim[j]) \
                - gammaln(B.value[j] * omega.estim[j]) - gammaln((1 - B.value[j]) * omega.estim[j]) \
                for r in xrange(cascade.R)])[0]
            
            lhoodB.value[j] = outsum([gammaln(cascade.value[j][r] + B_null.value[j] * omega_null.estim[j]) \
                + gammaln(cascade.total[j][r] - cascade.value[j][r] + (1 - B_null.value[j]) * omega_null.estim[j]) \
                - gammaln(cascade.total[j][r] + omega_null.estim[j]) + gammaln(omega_null.estim[j]) \
                - gammaln(B_null.value[j] * omega_null.estim[j]) - gammaln((1 - B_null.value[j]) * omega_null.estim[j]) \
                for r in xrange(cascade.R)])[0]

    return lhoodA, lhoodB


def likelihood(cascade, scores, eta, B, omega, \
    alpha, beta, tau, B_null, omega_null, model):

    apriori = insum(beta.estim * scores,[1])

    lhoodA, lhoodB = likelihoodAB(cascade, B, omega, B_null, omega_null, model)

    footprint = np.zeros((cascade.N,1),dtype=float)
    for j in xrange(cascade.J):
        footprint += insum(lhoodA.value[j],[1])

    P_1 = footprint + insum(gammaln(eta.total + alpha.estim[:,1]) - gammaln(alpha.estim[:,1]) \
        + alpha.estim[:,1] * nplog(tau.estim[:,1]) + eta.total * nplog(1 - tau.estim[:,1]), [1])
    P_1[P_1==np.inf] = MAX
    P_1[P_1==-np.inf] = -MAX

    null = np.zeros((cascade.N,1), dtype=float)
    for j in xrange(cascade.J):
        null += insum(lhoodB.value[j],[1])

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


def EM(cascade, scores, eta, B, omega, alpha, beta, tau, B_null, omega_null, model):

    # update binding posteriors
    eta.update(cascade, scores, B, omega, \
            alpha, beta, tau, B_null, omega_null, model)

    # update multi-scale parameters
    starttime = time.time()
    B.update(cascade, eta, omega)
    print "p_jk update in %.3f secs"%(time.time()-starttime)

    starttime = time.time()
    omega.update(cascade, eta, B)
    print "omega update in %.3f secs"%(time.time()-starttime)
    
    # update negative binomial parameters
    starttime = time.time()
    tau.update(eta, alpha)
    print "tau update in %.3f secs"%(time.time()-starttime)

    starttime = time.time()
    alpha.update(eta, tau)
    print "alpha update in %.3f secs"%(time.time()-starttime)

    # update prior parameters
    starttime = time.time()
    beta.update(scores, eta)
    print "beta update in %.3f secs"%(time.time()-starttime)


def square_EM(cascade, scores, eta, B, omega, alpha, beta, tau, B_null, omega_null, model):

    parameters = [B, omega, alpha, tau]
    oldvar = []
    for parameter in parameters:
        try:
            oldvar.append(parameter.estim.copy())
        except AttributeError:
            oldvar.append(np.hstack([parameter.value[j].copy() for j in xrange(parameter.J)]))
    oldvars = [oldvar]

    for step in [0,1]:
        EM(cascade, scores, eta, B, omega, \
                alpha, beta, tau, \
                B_null, omega_null, model)
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

    a_ok = False
    while not a_ok:
        invalid = np.zeros((0,), dtype='bool')
        for parameter,varA,varB,varC in zip(parameters,oldvars[0],oldvars[1],oldvars[2]):
            try:
                parameter.estim = (1+a)**2*varA - 2*a*(1+a)*varB + a**2*varC
                invalid = np.hstack((invalid,(parameter.estim<=0).ravel()))
            except AttributeError:
                newparam = (1+a)**2*varA - 2*a*(1+a)*varB + a**2*varC
                invalid = np.hstack((invalid, np.logical_or(newparam<0, newparam>1)))
                parameter.value = dict([(j,newparam[2**j-1:2**(j+1)-1]) \
                    for j in xrange(self.J)])
        if np.any(invalid):
            a = (a-1)/2.
            if np.abs(a+1)<1e-4:
                a = -1.
        else:
            a_ok = True

    EM(cascade, scores, eta, B, omega, \
                alpha, beta, tau, \
                B_null, omega_null, model)


def infer(reads, totalreads, scores, background, model, restarts, mintol):

    cascade = Cascade(reads)
    cascade_null = Cascade(background)
    scores = np.hstack((np.ones((cascade.N,1), dtype=float), scores))
    del reads

    # set background model
    B_null = Bin(cascade_null.J)
    for j in xrange(B_null.J):
        B_null.value[j] = np.sum(np.sum(cascade_null.value[j],0),0) / np.sum(np.sum(cascade_null.total[j],0),0).astype('float')
    omega_null = None
        
    if model=='msCentipede_flexbg':
        
        omega_null = Omega(cascade_null.J)

        eta_null = Eta(cascade_null, background.sum(1))
        eta_null.estim[:,1] = 1
        eta_null.estim[:,0] = 0

        # iterative update of background model; 
        # evaluate convergence based on change in estimated
        # background overdispersion
        change = np.inf
        while change>1e-1:
            change = omega_null.estim.copy()
            
            omega_null.update(cascade_null, eta_null, B_null)
            B_null.update(cascade_null, eta_null, omega_null)

            change = np.abs(change-omega_null.estim).sum()

    maxLoglike = -np.inf
    restart = 0
    while restart<restarts:

        try:
            # initialize multi-scale model parameters
            B = Bin(cascade.J)
            omega = Omega(cascade.J)

            # initialize negative binomial parameters
            alpha = Alpha(R=cascade.R)
            tau = Tau(R=cascade.R)

            # initialize prior parameters
            beta = Beta(scores)

            # initialize posterior over latent variables
            eta = Eta(cascade, totalreads, model=model)
            for j in xrange(B.J):
                B.value[j] = np.sum(cascade.value[j][0]*eta.estim[:,1:],0) / np.sum(cascade.total[j][0]*eta.estim[:,1:],0).astype('float')

            Loglike = likelihood(cascade, scores, eta, B, omega, \
                    alpha, beta, tau, B_null, omega_null, model)

            tol = np.inf
            iter = 0

            while np.abs(tol)>mintol:

                itertime = time.time()
                square_EM(cascade, scores, eta, B, omega, \
                        alpha, beta, tau, B_null, omega_null, model)

                # compute likelihood every 10 iterations
                if (iter+1)%1==0:

                    newLoglike = likelihood(cascade, scores, eta, B, omega, \
                            alpha, beta, tau, B_null, omega_null, model)

                    tol = newLoglike - Loglike
                    Loglike = newLoglike
                    print iter+1, newLoglike, tol, time.time()-itertime

                iter += 1

            # test if bound sites have more total reads than unbound sites;
            # avoids local optima issues.
            negbinmeans = alpha.estim*(1-tau.estim)/tau.estim
            if np.any(negbinmeans[:,0]<negbinmeans[:,1]):
                restart += 1
                if Loglike>maxLoglike:
                    maxLoglikeres = Loglike
                    if model in ['msCentipede','msCentipede_flexbgmean']:
                        footprint_model = (B, omega, B_null)
                    elif model=='msCentipede_flexbg':
                        footprint_model = (B, omega, B_null, omega_null)
                    count_model = (alpha, tau)
                    prior = beta

        except ValueError as err:

            print "restarting inference"

    return footprint_model, count_model, prior


def decode(reads, totalreads, scores, background, footprint, negbinparams, prior, model):

    (N,L,R) = reads.shape
    cascade = Cascade(reads)
    cascade_null = Cascade(background)
    scores = np.hstack((np.ones((cascade.N,1), dtype=float), scores))
    del reads

    # negative binomial parameters
    alpha = negbinparams[0]
    tau = negbinparams[1]
    beta = prior

    # multiscale parameters
    B = footprint[0]
    omega = footprint[1]
    
    # setting background model
    B_null = footprint[2]
    for j in xrange(B_null.J):
        B_null.value[j] = np.sum(np.sum(cascade_null.value[j],0),0) / np.sum(np.sum(cascade_null.total[j],0),0).astype('float')
    omega_null = None

    if model=='msCentipede_flexbg':

        omega_null = footprint[3]

        eta_null = Eta(cascade_null, background.sum(1))
        eta_null.estim[:,1] = 1
        eta_null.estim[:,0] = 0

        # iterative update of background model, when
        # accounting for overdispersion
        change = np.inf
        while change>1e-1:
            change = omega_null.estim.copy()
            
            B_null.update(cascade_null, eta_null, omega_null)

            omega_null.update(cascade_null, eta_null, B_null)

            change = np.abs(change-omega_null.estim).sum()

    eta = Eta(cascade, totalreads, scores, \
            B=B, omega=omega, B_null=B_null, omega_null=omega_null, \
            beta=beta, alpha=alpha, tau=tau, model=model)
    
    return eta.estim
