
import numpy as np
cimport numpy as np

cdef class Pi:

	cdef public long J
	cdef public dict value

	cdef update(self, data, zeta, Tau tau)

	cdef tuple function_gradient(self, np.ndarray[np.float64_t, ndim=1] x, dict args)

	cdef tuple function_gradient_hessian(self, np.ndarray[np.float64_t, ndim=1] x, dict args)

cdef class Tau:

	cdef public long J
	cdef public np.ndarray value

	cdef update(self, data, zeta, Pi pi)

	cdef tuple function_gradient(self, np.ndarray[np.float64_t, ndim=1] x, dict args)

	cdef tuple function_gradient_hessian(self, np.ndarray[np.float64_t, ndim=1] x, dict args)