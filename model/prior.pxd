
import numpy as np
cimport numpy as np
from numpy cimport ndarray

cdef class Beta:

    cdef public long S
    cdef public ndarray estim

    cdef update(self, ndarray[np.float64_t, ndim=2] scores, zeta)

    cdef tuple function_gradient(self, ndarray[np.float64_t, ndim=1] x, dict args)

    cdef tuple function_gradient_hessian(self, ndarray[np.float64_t, ndim=1] x, dict args)
