
import numpy as np
cimport numpy as np
from numpy cimport ndarray

cdef class Beta:

    cdef public long S
    cdef public ndarray value

    cdef update(self, ndarray[np.float64_t, ndim=2] scores, zeta)

    @staticmethod
    cdef public tuple function_gradient(ndarray[np.float64_t, ndim=1] x, dict args)

    @staticmethod
    cdef public tuple function_gradient_hessian(ndarray[np.float64_t, ndim=1] x, dict args)
