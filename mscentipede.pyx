
# base libraries
import sys
import time
import math
import pdb

# numerical libraries
import numpy as np
cimport numpy as np
from scipy.special import gammaln

# custom libraries
from model.footprint import Pi, Tau
from model.abundance import Alpha, Omega
from model.prior import Beta
import model.optimizer as optimizer
import model.utils as utils


cdef class Data:
    """
    A data structure to store a multiscale representation of
    chromatin accessibility read counts across `N` genomic windows of
    length `L` in `R` replicates.

    Arguments
        reads : array

    """

    def __cinit__(self, np.ndarray[np.float64_t, ndim=3] reads):

        cdef long L, k, j, size

        self.N = reads.shape[0]
        L = reads.shape[1]
        self.R = reads.shape[2]
        self.J = math.frexp(L)[1]-1
        self.left = np.zeros((self.N,L-1,self.R), dtype=np.int64)
        self.total = np.zeros((self.N,L-1,self.R), dtype=np.int64)
        for j from 0 <= j < self.J:
            size = L/(2**(j+1))
            self.total[:,2**j-1:2**(j+1)-1,:] = np.array([reads[:,k*size:(k+2)*size,:].sum(1) for k in xrange(0,2**(j+1),2)]).T
            self.left[:,2**j-1:2**(j+1)-1,:] = np.array([reads[:,k*size:(k+1)*size,:].sum(1) for k in xrange(0,2**(j+1),2)]).T


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
            self.value = np.zeros((self.N, 1),dtype=float)
            order = np.argsort(self.total.sum(1))
            indices = order[:self.N/2]
            self.value[indices,0] = -utils.MAX
            indices = order[self.N/2:]
            self.value[indices,0] = utils.MAX
            self.value = utils.logistic(-1*self.value)

    cdef update(self, Data data, np.ndarray[np.float64_t, ndim=2] scores, \
        Pi pi, Tau tau, Alpha alpha, Beta beta, Omega omega, \
        Pi pi_null, Tau tau_null, str model):

        cdef long j
        cdef np.ndarray[np.float64_t, ndim=2] footprint_logodds, prior_logodds, negbin_logodds
        cdef Data lhoodA, lhoodB

        footprint_logodds = np.zeros((self.N,1), dtype=float)
        lhoodA, lhoodB = compute_footprint_likelihood(data, pi, tau, pi_null, tau_null, model)

        for j from 0 <= j < data.J:
            footprint_logodds += utils.insum(lhoodA.value[j] - lhoodB.value[j],[1])

        prior_logodds = utils.insum(beta.value * scores, [1])
        negbin_logodds = utils.insum(gammaln(self.total + alpha.value.T[1]) \
                - gammaln(self.total + alpha.value.T[0]) \
                + gammaln(alpha.value.T[0]) - gammaln(alpha.value.T[1]) \
                + alpha.value.T[1] * utils.nplog(omega.value.T[1]) - alpha.value.T[0] * utils.nplog(omega.value.T[0]) \
                + self.total * (utils.nplog(1 - omega.value.T[1]) - utils.nplog(1 - omega.value.T[0])),[1])

        self.value = prior_logodds + footprint_logodds + negbin_logodds
        self.value[self.value==np.inf] = utils.MAX
        self.value = utils.logistic(-1*self.value)

    cdef infer(self, Data data, np.ndarray[np.float64_t, ndim=2] scores, \
        Pi pi, Tau tau, Alpha alpha, Beta beta, Omega omega, \
        Pi pi_null, Tau tau_null, str model):

        cdef long j
        cdef Data lhoodA, lhoodB

        lhoodA, lhoodB = compute_footprint_likelihood(data, pi, tau, pi_null, tau_null, model)

        for j from 0 <= j < data.J:
            self.footprint_log_likelihood_ratio += utils.insum(lhoodA.value[j] - lhoodB.value[j],[1])
        self.footprint_log_likelihood_ratio = self.footprint_log_likelihood_ratio / np.log(10)

        self.prior_log_odds = utils.insum(beta.value * scores, [1]) / np.log(10)

        self.total_log_likelihood_ratio = utils.insum(gammaln(self.total + alpha.value.T[1]) \
            - gammaln(self.total + alpha.value.T[0]) \
            + gammaln(alpha.value.T[0]) - gammaln(alpha.value.T[1]) \
            + alpha.value.T[1] * utils.nplog(omega.value.T[1]) - alpha.value.T[0] * utils.nplog(omega.value.T[0]) \
            + self.total * (utils.nplog(1 - omega.value.T[1]) - utils.nplog(1 - omega.value.T[0])),[1])
        self.total_log_likelihood_ratio = self.total_log_likelihood_ratio / np.log(10)

        self.posterior_log_odds = self.prior_log_odds \
            + self.footprint_log_likelihood_ratio \
            + self.total_log_likelihood_ratio


cdef tuple compute_footprint_likelihood(Data data, Pi pi, Tau tau, Pi pi_null, Tau tau_null, str model_type):
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

        model_type : string 
        {msCentipede, msCentipede-flexbgmean, msCentipede-flexbg}

    """

    cdef long j, r
    cdef Data lhood_bound, lhood_unbound

    lhood_bound = Data()
    lhood_unbound = Data()

    for j from 0 <= j < data.J:
        value = np.sum(data.value[j],0)
        total = np.sum(data.total[j],0)
        
        lhood_bound.value[j] = np.sum([gammaln(data.value[j][r] + pi.value[j] * tau.value[j]) \
            + gammaln(data.total[j][r] - data.value[j][r] + (1 - pi.value[j]) * tau.value[j]) \
            - gammaln(data.total[j][r] + tau.value[j]) + gammaln(tau.value[j]) \
            - gammaln(pi.value[j] * tau.value[j]) - gammaln((1 - pi.value[j]) * tau.value[j]) \
            for r in xrange(data.R)],0)

        if model_type in ['msCentipede','msCentipede_flexbgmean']:
            
            lhood_unbound.value[j] = value * utils.nplog(pi_null.value[j]) \
                + (total - value) * utils.nplog(1 - pi_null.value[j])
    
        elif model_type=='msCentipede_flexbg':
            
            lhood_unbound.value[j] = np.sum([gammaln(data.value[j][r] + pi_null.value[j] * tau_null.value[j]) \
                + gammaln(data.total[j][r] - data.value[j][r] + (1 - pi_null.value[j]) * tau_null.value[j]) \
                - gammaln(data.total[j][r] + tau_null.value[j]) + gammaln(tau_null.value[j]) \
                - gammaln(pi_null.value[j] * tau_null.value[j]) - gammaln((1 - pi_null.value[j]) * tau_null.value[j]) \
                for r in xrange(data.R)],0)

    return lhood_bound, lhood_unbound


cdef double likelihood(Data data, np.ndarray[np.float64_t, ndim=2] scores, \
    Zeta zeta, Pi pi, Tau tau, Alpha alpha, Beta beta, \
    Omega omega, Pi pi_null, Tau tau_null, str model_type):
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

        model_type : string
        {msCentipede, msCentipede_flexbgmean, msCentipede_flexbg}

    """

    cdef long j
    cdef double L
    cdef np.ndarray apriori, footprint, null, P_1, P_0, LL
    cdef Data lhoodA, lhoodB

    apriori = utils.insum(beta.value * scores,[1])

    lhoodA, lhoodB = compute_footprint_likelihood(data, pi, tau, pi_null, tau_null, model_type)

    footprint = np.zeros((data.N,1),dtype=float)
    for j from 0 <= j < data.J:
        footprint += utils.insum(lhoodA.value[j],[1])

    P_1 = footprint + utils.insum(gammaln(zeta.total + alpha.value[:,1]) - gammaln(alpha.value[:,1]) \
        + alpha.value[:,1] * utils.nplog(omega.value[:,1]) + zeta.total * utils.nplog(1 - omega.value[:,1]), [1])
    P_1[P_1==np.inf] = utils.MAX
    P_1[P_1==-np.inf] = -utils.MAX

    null = np.zeros((data.N,1), dtype=float)
    for j from 0 <= j < data.J:
        null += utils.insum(lhoodB.value[j],[1])

    P_0 = null + utils.insum(gammaln(zeta.total + alpha.value[:,0]) - gammaln(alpha.value[:,0]) \
        + alpha.value[:,0] * utils.nplog(omega.value[:,0]) + zeta.total * utils.nplog(1 - omega.value[:,0]), [1])
    P_0[P_0==np.inf] = utils.MAX
    P_0[P_0==-np.inf] = -utils.MAX

    LL = P_0 * zeta.value[:,:1] + utils.insum(P_1 * zeta.value[:,1:],[1]) + apriori * (1 - zeta.value[:,:1]) \
        - utils.nplog(1 + np.exp(apriori)) - utils.insum(zeta.value * utils.nplog(zeta.value),[1])
 
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
    Omega omega, Pi pi_null, Tau tau_null, str model_type):
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

        model_type : string
        {msCentipede, msCentipede_flexbgmean, msCentipede_flexbg}

    """

    cdef double starttime

    # update binding posteriors
    zeta.update(data, scores, pi, tau, \
            alpha, beta, omega, pi_null, tau_null, model_type)

    # update multi-scale parameters
    starttime = time.time()
    tau.update(data, zeta, pi)
    print "tau update in %.3f secs"%(time.time()-starttime)

    starttime = time.time()
    pi.update(data, zeta, tau)
    print "p_jk update in %.3f secs"%(time.time()-starttime)

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
    Omega omega, Pi pi_null, Tau tau_null, str model_type):
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

        model_type : string
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
            oldvar.append(parameter.value.copy())
        except AttributeError:
            oldvar.append(np.hstack([parameter.value[j].copy() for j in xrange(parameter.J)]))
    oldvars = [oldvar]

    # take two update steps
    for step in [0,1]:
        EM(data, scores, zeta, pi, tau, alpha, beta, omega, pi_null, tau_null, model_type)
        oldvar = []
        for parameter in parameters:
            try:
                oldvar.append(parameter.value.copy())
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
                parameter.value = (1+a)**2*varA - 2*a*(1+a)*varB + a**2*varC
                # ensure constraints on variables are satisfied
                invalid = np.hstack((invalid,(parameter.value<=0).ravel()))
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

    EM(data, scores, zeta, pi, tau, alpha, beta, omega, pi_null, tau_null, model_type)


def estimate_optimal_model(np.ndarray[np.float64_t, ndim=3] reads, \
    np.ndarray[np.float64_t, ndim=2] totalreads, \
    np.ndarray[np.float64_t, ndim=2] scores, \
    np.ndarray[np.float64_t, ndim=3] background, \
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

        zeta_null = Zeta(background.sum(1), data_null.N, False)
        zeta_null.value[:,1] = 1
        zeta_null.value[:,0] = 0

        # iterative update of background model; 
        # evaluate convergence based on change in estimated
        # background overdispersion
        change = np.inf
        while change>1e-2:
            oldtau = tau_null.value.copy()
            
            tau_null.update(data_null, zeta_null, pi_null)
            pi_null.update(data_null, zeta_null, tau_null)

            change = np.abs(oldtau-tau_null.value).sum() / tau_null.J

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
            zeta = Zeta(totalreads, data.N, False)
            for j in xrange(pi.J):
                pi.value[j] = np.sum(data.value[j][0] * zeta.value[:,1:],0) \
                    / np.sum(data.total[j][0] * zeta.value[:,1:],0).astype('float')
                pi.value[j][pi.value[j]<1e-10] = 1e-10
                pi.value[j][pi.value[j]>1-1e-10] = 1-1e-10

            # initial log likelihood of the model
            Loglike = likelihood(data, scores, zeta, pi, tau, \
                    alpha, beta, omega, pi_null, tau_null, model)
            print Loglike

            tol = np.inf
            iteration = 0

            while np.abs(tol)>mintol:

                itertime = time.time()
                EM(data, scores, zeta, pi, tau, \
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
            negbinmeans = alpha.value * (1-omega.value)/omega.value
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

    data = Data()
    data.transform_to_multiscale(reads)
    data_null = Data()
    data_null.transform_to_multiscale(background)
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

            zeta_null = Zeta(background.sum(1), data_null.N, False)
            zeta_null.value[:,1] = 1
            zeta_null.value[:,0] = 0

            # iterative update of background model, when
            # accounting for overdispersion
            change = np.inf
            while change>1e-1:
                change = tau_null.value.copy()
                
                pi_null.update(data_null, zeta_null, tau_null)

                tau_null.update(data_null, zeta_null, pi_null)

                change = np.abs(change-tau_null.value).sum()

    zeta = Zeta(totalreads, data.N, True)

    zeta.infer(data, scores, pi, tau, alpha, beta, omega, \
        pi_null, tau_null, model)
    
    return zeta.posterior_log_odds, \
        zeta.prior_log_odds, zeta.footprint_log_likelihood_ratio, \
        zeta.total_log_likelihood_ratio

