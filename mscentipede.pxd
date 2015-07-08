import numpy as np
cimport numpy as np
from cpython cimport bool

cdef class Data:

	cdef public long N, L, R, J
	cdef public dict value, total

	cdef transform_to_multiscale(self, np.ndarray[np.float64_t, ndim=3] reads)


cdef class Zeta:

	cdef public long N
	cdef public np.ndarray[np.float64_t, ndim=2] total, prior_log_odds, \
		footprint_log_likelihood_ratio, total_log_likelihood_ratio, \
		posterior_log_odds, estim

	cdef update(self)

	cdef infer(self)


cdef class Pi:

	cdef public long J
	cdef public dict value

cdef tuple pi_function_gradient(np.ndarray[np.float64_t, ndim=1] x, dict kwargs)

cdef tuple pi_function_gradient_hessian(np.ndarray[np.float64_t, ndim=1] x, dict kwargs)


cdef class Tau:

	cdef public long J
	cdef public np.ndarray estim

cdef tuple tau_function_gradient(np.ndarray[np.float64_t, ndim=1] x, dict kwargs)

cdef tuple tau_function_gradient_hessian(np.ndarray[np.float64_t, ndim=1] x, dict kwargs)


cdef class Alpha:

	cdef public long R
	cdef public np.ndarray estim

cdef tuple alpha_function_gradient(np.ndarray[np.float64_t, ndim=1] x, dict kwargs)

cdef tuple alpha_function_gradient_hessian(np.ndarray[np.float64_t, ndim=1] x, dict kwargs)


cdef class Omega:

	cdef public long R
	cdef public np.ndarray estim

	cdef update(self, Zeta zeta, Alpha alpha)


cdef class Beta:

	cdef public long S
	cdef public np.ndarray estim

cdef tuple beta_function_gradient(np.ndarray[np.float64_t, ndim=1] x, dict kwargs)

cdef tuple beta_function_gradient_hessian(np.ndarray[np.float64_t, ndim=1] x, dict kwargs)


cdef tuple compute_footprint_likelihood(Data data, Pi pi, Tau tau, Pi pi_null, Tau tau_null, str model)


cdef double likelihood(Data data, np.ndarray[np.float64_t, ndim=2] scores, \
	Zeta zeta, Pi pi, Tau tau, Alpha alpha, Beta beta, \
	Omega omega, Pi pi_null, Tau tau_null, str model)

cdef EM(Data data, np.ndarray[np.float64_t, ndim=2] scores, \
    Zeta zeta, Pi pi, Tau tau, Alpha alpha, Beta beta, \
    Omega omega, Pi pi_null, Tau tau_null, str model)

cdef square_EM(Data data, np.ndarray[np.float64_t, ndim=2] scores, \
    Zeta zeta, Pi pi, Tau tau, Alpha alpha, Beta beta, \
    Omega omega, Pi pi_null, Tau tau_null, str model)