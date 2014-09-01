import numpy as np
import cvxopt as cvx
from cvxopt import solvers
from scipy.special import digamma, gammaln
from utils import insum, outsum, nplog, EPS, MAX
import time, math, pdb

oldlogistic = lambda x: 1./(1+insum(np.exp(x),[1]))
logistic = lambda x: 1./(1+np.exp(x))

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

# change optimizer
class Bin(Cascade):

    def __init__(self, L, background='multinomial'):

        Cascade.__init__(self, np.random.rand(1,L))
        self.value = dict([(j,np.random.rand(2**j)) for j in xrange(self.J)])
        self.background = background
        self.update_count = 0

    def update_Mstep(self, cascade, eta, gamma, omega):

        def F(x):
            func = 0
            for j in xrange(self.J):
                left = 2**j-1
                right = 2**(j+1)-1
                func = func + ((1-gamma.value[j])*np.sum(eta.estim[:,1:]*outsum([gammaln(cascade.value[j][r]+omega.estim[j]*x[left:right]) \
                    + gammaln(cascade.total[j][r]-cascade.value[j][r]+omega.estim[j]*(1-x[left:right])) \
                    - gammaln(omega.estim[j]*x[left:right]) - gammaln(omega.estim[j]*(1-x[left:right])) for r in xrange(cascade.R)])[0],0)).sum()
            f = -1.*func
            if np.isnan(f) or np.isinf(f):
                return np.inf
            else:
                return f

        def Fprime(x):
            df = np.zeros(x.shape, dtype=float)
            for j in xrange(self.J):
                left = 2**j-1
                right = 2**(j+1)-1
                df[left:right] = (1-gamma.value[j])*omega.estim[j]*np.sum(eta.estim[:,1:]*outsum([digamma(cascade.value[j][r]+omega.estim[j]*x[left:right]) \
                    - digamma(cascade.total[j][r]-cascade.value[j][r]+omega.estim[j]*(1-x[left:right])) \
                    - digamma(omega.estim[j]*x[left:right]) + digamma(omega.estim[j]*(1-x[left:right])) for r in xrange(cascade.R)])[0],0)
            Df = -1.*df.ravel()
            if np.isnan(Df).any() or np.isinf(Df).any():
                return np.inf*np.ones(x.shape,dtype=float)
            else:
                return Df

        xo = np.array([v for j in xrange(self.J) for v in self.value[j]])
        bounds = [(0, 1) for i in xrange(xo.size)]
        solution = opt.fmin_tnc(F, xo, fprime=Fprime, bounds=bounds, disp=0)
        if solution[2] not in [1,2]:
            print "Run SLSQP"
            solution = opt.fmin_slsqp(F, xo, fprime=Fprime, bounds=bounds, disp=0)
        else:
            solution = solution[0]
        self.value = dict([(j,solution[2**j-1:2**(j+1)-1]) for j in xrange(self.J)])

    self.update_count += 1
    print "Bin updated %d"%self.update_count


# change optimizer
class Omega():

    def __init__(self, J):

        self.J = J
        self.estim = 10*np.random.rand(self.J)
        self.update_count = 0

    def update_Mstep(self, cascade, eta, gamma, B):

        def F(x):
            func = 0
            for j in xrange(gamma.J):
                func = func + (gamma.value[j]*np.sum(eta.estim[:,1:]*outsum([gammaln(cascade.value[j][r]+0.5*x[j]) \
                    + gammaln(cascade.total[j][r]-cascade.value[j][r]+0.5*x[j]) \
                    - gammaln(cascade.total[j][r]+x[j]) + gammaln(x[j]) \
                    - 2*gammaln(0.5*x[j]) for r in xrange(cascade.R)])[0],0) \
                    + (1-gamma.value[j])*np.sum(eta.estim[:,1:]*outsum([gammaln(cascade.value[j][r]+B.value[j]*x[j]) \
                    + gammaln(cascade.total[j][r]-cascade.value[j][r]+(1-B.value[j])*x[j]) \
                    - gammaln(cascade.total[j][r]+x[j]) + gammaln(x[j]) \
                    - gammaln(B.value[j]*x[j]) - gammaln((1-B.value[j])*x[j]) for r in xrange(cascade.R)])[0],0)).sum()
            f = -1.*func.sum()
            if np.isnan(f) or np.isinf(f):
                return np.inf
            else:
                return f

        def Fprime(x):
            df = np.zeros(x.shape,dtype=float)
            for j in xrange(gamma.J):
                dfj = 0.5*gamma.value[j]*np.sum(eta.estim[:,1:]*outsum([digamma(cascade.value[j][r]+0.5*x[j]) \
                    + digamma(cascade.total[j][r]-cascade.value[j][r]+0.5*x[j]) \
                    - 2*digamma(cascade.total[j][r]+x[j]) + 2*digamma(x[j]) \
                    - 2*digamma(0.5*x[j]) for r in xrange(cascade.R)])[0],0) \
                    + (1-gamma.value[j])*np.sum(eta.estim[:,1:]*outsum([B.value[j]*digamma(cascade.value[j][r]+B.value[j]*x[j]) \
                    + (1-B.value[j])*digamma(cascade.total[j][r]-cascade.value[j][r]+(1-B.value[j])*x[j]) \
                    - digamma(cascade.total[j][r]+x[j]) + digamma(x[j]) \
                    - B.value[j]*digamma(B.value[j]*x[j]) - (1-B.value[j])*digamma((1-B.value[j])*x[j]) for r in xrange(cascade.R)])[0],0)
                df[j] = dfj.sum()
            Df = -1*df
            if np.isnan(Df).any() or np.isinf(Df).any():
                return np.inf*np.ones(Df.shape,dtype=float)
            else:
                return Df

        xo = self.estim
        bounds = [(0, np.inf) for i in xrange(xo.size)]
        solution = opt.fmin_tnc(F, xo, fprime=Fprime, bounds=bounds, disp=0)
        if solution[2] not in [1,2]:
            solution = opt.fmin_tnc(F, np.random.rand(xo.size), fprime=Fprime, bounds=bounds, disp=0)
        print (solution[2], opt.tnc.RCSTRINGS[solution[2]])
        self.estim = solution[0]
        self.update_count += 1

    print "Omega updated %d"%self.update_count


# change optimizer
class Alpha():

    def __init__(self, R=1):

        self.R = R
        self.estim = np.random.rand(self.R,2)*10
        self.update_count = 0

    def update_Mstep(self, eta, tau):

        C = [nplog(tau.estim[r]) * outsum(eta.estim)[0] \
            for r in xrange(self.R)]

        def F(x):
            func = np.array([outsum(gammaln(eta.total[:,r:r+1]+x[2*r:2*r+2])*eta.estim) \
                - gammaln(x[2*r:2*r+2])*outsum(eta.estim)[0] + C[r]*x[2*r:2*r+2] for r in xrange(self.R)])
            f = -1.*func.sum()
            if np.isnan(f) or np.isinf(f):
                return np.inf
            else:
                return f

        def Fprime(x):
            df = np.array([outsum(digamma(eta.total[:,r:r+1]+x[2*r:2*r+2])*eta.estim) \
                - digamma(x[2*r:2*r+2])*outsum(eta.estim)[0] + C[r] for r in xrange(self.R)])
            Df = -1.*df.ravel()
            if np.isnan(Df).any() or np.isinf(Df).any():
                return np.inf*np.ones(x.shape,dtype=float)
            else:
                return Df

        xo = self.estim.ravel()
        bounds = [(0, None) for i in xrange(xo.size)]
        solution = opt.fmin_l_bfgs_b(F, xo, fprime=Fprime, bounds=bounds, disp=0)
        self.estim = solution[0].reshape(self.R,2)
        print (solution[2]['warnflag'],solution[2]['task'])

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

    def update_Mstep(self, eta, alpha):

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

# change optimizer
class Beta():

    def __init__(self, S):
    
        self.S = S
        self.estim = np.random.rand(self.S)
        self.update_count = 0

    def update_Mstep(self, scores, eta):

        def F(x):
            arg = x[0]+x[1]*scores
            func = arg*insum(eta.estim[:,1:],1) - nplog(1+np.exp(arg))
            f = -1.*func.sum()
            if np.isnan(f) or np.isinf(f):
                return np.inf
            else:
                return f

        def Fprime(x):
            arg = x[0]+x[1]*scores
            df1 = insum(eta.estim[:,1:],1) - oldlogistic(-arg)
            df2 = df1*scores
            Df = -1.*np.array([df1.sum(), df2.sum()])
            if np.isnan(Df).any() or np.isinf(Df).any():
                return np.inf
            else:
                return Df

        xo = self.estim.copy()
        solution = opt.fmin_bfgs(F, xo, fprime=Fprime, disp=0)
        self.estim = solution

        if np.isnan(self.estim).any():
            print "Nan in Beta"
            raise ValueError

        if np.isinf(self.estim).any():
            print "Inf in Beta"
            raise ValueError

        self.update_count += 1        


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
        eta.update_Estep(cascade, scores, B, pi, gamma, omega, \
            alpha, beta, tau, B_null)
    else:
        eta.update_Estep(cascade, scores, B, pi, gamma, omega, \
            alpha, beta, tau, B_null, gamma_null=gamma_null, omega_null=omega_null)

    # update multi-scale latent variables
    if background=='multinomial':
        gamma.update_Estep(cascade, eta, B, pi, omega, B_null)
    else:
        gamma.update_Estep(cascade, eta, B, pi, omega, B_null, \
            gamma_null=gamma_null, omega_null=omega_null)

    # update multi-scale parameters
    pi.update_Mstep(gamma)
    B.update_Mstep(cascade, eta, gamma, omega)

    # update negative binomial parameters
    tau.update_Mstep(eta, alpha)
    alpha.update_Mstep(eta, tau)

    # update prior parameters
    beta.update_Mstep(scores, eta)


def infer(reads, totalreads, scores, background, background_model='multinomial', restarts=3, mintol=1.):

    """
    modelC -- Poisson-Binomial footprint model; flat background model
    modelD -- Poisson-Binomial footprint model; multinomial background model
    modelE -- Poisson-Binomial footprint model; Poisson-Binomial background model
    modelF -- Poisson-Binomial footprint model; site-specific multinomial model
    """

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
                    Loglike = Loglikenew
                    print iter+1, Loglikenew, tol, time.time()-itertime
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
                    negbinparams = (alpha, tau)
                    prior = beta

        except ValueError as err:

            print "restarting inference"

    return footprint, negbinparams, prior


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
