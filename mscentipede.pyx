
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
        self.left = np.zeros((self.N,L-1,self.R), dtype=np.float64)
        self.total = np.zeros((self.N,L-1,self.R), dtype=np.float64)
        for j from 0 <= j < self.J:
            size = L/(2**(j+1))
            self.total[:,2**j-1:2**(j+1)-1,:] = np.transpose(np.array([reads[:,k*size:(k+2)*size,:].sum(1) \
                                                for k in xrange(0,2**(j+1),2)]), (1,0,2))
            self.left[:,2**j-1:2**(j+1)-1,:] = np.transpose(np.array([reads[:,k*size:(k+1)*size,:].sum(1) \
                                                for k in xrange(0,2**(j+1),2)]), (1,0,2))


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
            order = np.argsort(self.total.sum(1))[::-1]
            indices = order[:self.N/5]
            self.value[indices,0] = utils.MAX
            indices = order[self.N/5:]
            self.value[indices,0] = -utils.MAX
            self.value = utils.logistic(-1*self.value)

    cdef update(self, Data data, np.ndarray[np.float64_t, ndim=2] scores, \
        Pi pi, Tau tau, Alpha alpha, Beta beta, Omega omega, \
        Pi pi_null, Tau tau_null, str model_type):

        cdef long j
        cdef np.ndarray[np.float64_t, ndim=2] footprint_logodds, prior_logodds, negbin_logodds, lhood_bound, lhood_unbound

        lhood_bound, lhood_unbound = compute_footprint_likelihood(data, pi, tau, pi_null, tau_null, model_type)
        footprint_logodds = utils.insum(lhood_bound-lhood_unbound, [1])

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
        Pi pi_null, Tau tau_null, str model_type):

        cdef long j
        cdef np.ndarray lhood_bound, lhood_unbound

        lhood_bound, lhood_unbound = compute_footprint_likelihood(data, pi, tau, pi_null, tau_null, model_type)
        self.footprint_log_likelihood_ratio = utils.insum(lhood_bound-lhood_unbound, [1]) / np.log(10)

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
    cdef np.ndarray lhood_bound, lhood_unbound, alpha, beta, ttau

    alpha = np.zeros((pi.L,), dtype='float')
    beta = np.zeros((pi.L,), dtype='float')
    ttau = np.zeros((pi.L,), dtype='float')

    for j from 0 <= j < data.J:
        start = 2**j-1
        end = 2**(j+1)-1
        alpha[start:end] = pi.value[start:end] * tau.value[j]
        beta[start:end] = (1-pi.value[start:end]) * tau.value[j]
        ttau[start:end] = tau.value[j]

    lhood_bound = np.sum([gammaln(data.left[:,:,r] + alpha) + \
                  gammaln(data.total[:,:,r] - data.left[:,:,r] + beta) - \
                  gammaln(data.total[:,:,r] + ttau) + gammaln(ttau) - \
                  gammaln(alpha) - gammaln(beta) \
                  for r in xrange(data.R)],0)

    if model_type in ['msCentipede','msCentipede_flexbgmean']:
        
        lhood_unbound = np.sum(data.left,2) * utils.nplog(pi_null.value) + \
                        (np.sum(data.total,2) - np.sum(data.left,2)) * \
                        utils.nplog(1-pi_null.value)

    elif model_type=='msCentipede_flexbg':

        alpha = np.zeros((pi_null.L,), dtype='float')
        beta = np.zeros((pi_null.L,), dtype='float')
        ttau = np.zeros((pi_null.L,), dtype='float')
        for j from 0 <= j < data.J:
            start = 2**j-1
            end = 2**(j+1)-1
            alpha[start:end] = pi.value[start:end] * tau_null.value[j]
            beta[start:end] = (1-pi.value[start:end]) * tau_null.value[j]
            ttau[start:end] = tau_null.value[j]

        lhood_unbound = np.sum([gammaln(data.left[:,:,r] + alpha) + \
                        gammaln(data.total[:,:,r] - data.left[:,:,r] + beta) - \
                        gammaln(data.total[:,:,r] + ttau) + gammaln(ttau) - \
                        gammaln(alpha) - gammaln(beta) \
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
    cdef np.ndarray apriori, footprint, null, P_1, P_0, LL, pidx, nidx

    apriori = utils.insum(beta.value * scores,[1])
    pidx = apriori>=0
    nidx = apriori<0
    L_prior = np.sum(apriori[nidx] * zeta.value[nidx]) - \
              np.sum(apriori[pidx] * (1-zeta.value[pidx])) - \
              np.sum(utils.nplog(1 + np.exp(-1*np.abs(apriori))))

    lhoodA, lhoodB = compute_footprint_likelihood(data, pi, tau, pi_null, tau_null, model_type)

    footprint = utils.insum(lhoodA,[1])

    P_1 = footprint + utils.insum(gammaln(zeta.total + alpha.value[:,1]) - gammaln(alpha.value[:,1]) \
        + alpha.value[:,1] * utils.nplog(omega.value[:,1]) + zeta.total * utils.nplog(1 - omega.value[:,1]), [1])
    P_1[P_1==np.inf] = utils.MAX
    P_1[P_1==-np.inf] = -utils.MAX

    null = utils.insum(lhoodB,[1])

    P_0 = null + utils.insum(gammaln(zeta.total + alpha.value[:,0]) - gammaln(alpha.value[:,0]) \
        + alpha.value[:,0] * utils.nplog(omega.value[:,0]) + zeta.total * utils.nplog(1 - omega.value[:,0]), [1])
    P_0[P_0==np.inf] = utils.MAX
    P_0[P_0==-np.inf] = -utils.MAX

    LL = P_0 * (1-zeta.value) + \
         P_1 * zeta.value - \
         zeta.value * utils.nplog(zeta.value) - \
         (1-zeta.value) * utils.nplog(1-zeta.value)
 
    L = (LL.sum() + L_prior) / data.N

    return L


cdef EM(Data data, np.ndarray[np.float64_t, ndim=2] scores, \
    Zeta zeta, Pi pi, Tau tau, Alpha alpha, Beta beta, \
    Omega omega, Pi pi_null, Tau tau_null, long threads, str model_type):
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

        threads : int
        number of threads to use while updating `pi` and `tau`

        model_type : string
        {msCentipede, msCentipede_flexbgmean, msCentipede_flexbg}

    """

    cdef double starttime

    # update binding posteriors
    zeta.update(data, scores, pi, tau, \
            alpha, beta, omega, pi_null, tau_null, model_type)

    # update multi-scale parameters
    starttime = time.time()
    tau.update(data, zeta, pi, threads)
    print "tau update in %.3f secs"%(time.time()-starttime)

    starttime = time.time()
    pi.update(data, zeta, tau, threads)
    print "p_jk update in %.3f secs"%(time.time()-starttime)
    print pi.value

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
    Omega omega, Pi pi_null, Tau tau_null, long threads, str model_type):
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

        threads : int
        number of threads to run during execution

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
        EM(data, scores, zeta, pi, tau, alpha, beta, omega, pi_null, tau_null, threads, model_type)
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

    EM(data, scores, zeta, pi, tau, alpha, beta, omega, pi_null, tau_null, threads, model_type)


def estimate_optimal_model(np.ndarray[np.float64_t, ndim=3] reads, \
    np.ndarray[np.float64_t, ndim=2] totalreads, \
    np.ndarray[np.float64_t, ndim=2] scores, \
    np.ndarray[np.float64_t, ndim=3] background, \
    str model_type, long threads, long restarts, double mintol):
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

        threads : int
        number of threads to use during execution (default: 1)

        restarts : int
        number of independent runs of model learning

        mintol : float
        convergence criterion

    """

    cdef long restart, iteration, err
    cdef double change, maxLoglike, Loglike, tol, itertime, totaltime
    cdef np.ndarray oldtau, negbinmeans, xmin
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
    pi_null.value = np.sum(np.sum(data_null.left,2),0) / np.sum(np.sum(data_null.total,2),0).astype('float')
    
    tau_null = Tau(data_null.J)
    if model_type=='msCentipede_flexbg':

        zeta_null = Zeta(background.sum(1), data_null.N, False)
        zeta_null.value[:,0] = 1

        # iterative update of background model; 
        # evaluate convergence based on change in estimated
        # background overdispersion
        change = np.inf
        while change>1e-2:
            oldtau = tau_null.value.copy()
            
            tau_null.update(data_null, zeta_null, pi_null, threads)
            pi_null.update(data_null, zeta_null, tau_null, threads)

            change = np.abs(oldtau-tau_null.value).sum() / tau_null.J

    maxLoglike = -np.inf
    restart = 0
    err = 1
    runlog = ['Number of sites = %d'%data.N]
    while restart<restarts:

        try:
            totaltime = time.time()
            print "Restart %d ..."%(restart+1)

            # instantiate multi-scale model parameters
            pi = Pi(data.J)
            tau = Tau(data.J)

            # instantiate negative binomial parameters
            alpha = Alpha(data.R)
            omega = Omega(data.R)

            # instantiate prior parameters
            beta = Beta(scores)

            # instantiate posterior over latent variables
            zeta = Zeta(totalreads, data.N, False)

            # initialize footprint
            pi.value = np.sum(np.sum(data.left,2) * zeta.value, 0) \
                / np.sum(np.sum(data.total,2) * zeta.value, 0).astype('float')
            pi.value[pi.value==0] = np.min(pi.value[pi.value!=0])
            pi.value[pi.value==1] = np.max(pi.value[pi.value!=1])
            tau.value = np.array([1./min([np.min(pi.value[2**j-1:2**(j+1)-1]), \
                                     np.min(1-pi.value[2**j-1:2**(j+1)-1])]) \
                                  for j in xrange(tau.J)]) + 1000 * np.random.rand(tau.J)

            # initial log likelihood of the model
            Loglike = likelihood(data, scores, zeta, pi, tau, \
                    alpha, beta, omega, pi_null, tau_null, model_type)
            print Loglike

            tol = np.inf
            iteration = 0

            while np.abs(tol)>mintol:

                itertime = time.time()
                EM(data, scores, zeta, pi, tau, \
                        alpha, beta, omega, pi_null, tau_null, threads, model_type)

                newLoglike = likelihood(data, scores, zeta, pi, tau, \
                        alpha, beta, omega, pi_null, tau_null, model_type)

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
                    if model_type in ['msCentipede','msCentipede_flexbgmean']:
                        footprint_model = (pi, tau, pi_null)
                    elif model_type=='msCentipede_flexbg':
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


def infer_binding_posterior(reads, totalreads, scores, background, footprint, negbinparams, prior, model_type):
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

        model_type : string
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

    if model_type=='msCentipede_flexbg':

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
        pi_null, tau_null, model_type)
    
    return zeta.posterior_log_odds, \
        zeta.prior_log_odds, zeta.footprint_log_likelihood_ratio, \
        zeta.total_log_likelihood_ratio

