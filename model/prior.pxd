
import numpy as np
cimport numpy as np

cdef class Beta:

	cdef public long S
	cdef public np.ndarray estim

	cdef tuple function_gradient(self, np.ndarray[np.float64_t, ndim=1] x, dict args)

	cdef tuple function_gradient_hessian(self, np.ndarray[np.float64_t, ndim=1] x, dict args)