
import numpy as np
cimport numpy as np

cdef class Alpha:

	cdef public long R
	cdef public np.ndarray value

	cdef update(self, zeta, Omega omega)

	cdef tuple function_gradient(self, np.ndarray[np.float64_t, ndim=1] x, dict args)

	cdef tuple function_gradient_hessian(self, np.ndarray[np.float64_t, ndim=1] x, dict args)

cdef class Omega:

	cdef public long R
	cdef public np.ndarray estim

	cdef update(self, zeta, Alpha alpha)