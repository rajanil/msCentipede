
import numpy as np
cimport numpy as np
from numpy cimport ndarray

cdef class Pi:

    cdef public long J, L
    cdef public ndarray value

    cdef update(self, data, zeta, Tau tau, long threads)

    @staticmethod
    cdef public tuple function_gradient(ndarray[np.float64_t, ndim=1] x, dict args)

    @staticmethod
    cdef public tuple function_gradient_hessian(ndarray[np.float64_t, ndim=1] x, dict args)

cdef class Tau:

    cdef public long J
    cdef public ndarray value

    cdef update(self, data, zeta, Pi pi, long threads)

    @staticmethod
    cdef public tuple function_gradient(ndarray[np.float64_t, ndim=1] x, dict args)

    @staticmethod
    cdef public tuple function_gradient_hessian(ndarray[np.float64_t, ndim=1] x, dict args)
