
import numpy as np
cimport numpy as np

cdef class Alpha:

    cdef public long R
    cdef public np.ndarray value

    cdef update(self, zeta, Omega omega)

    @staticmethod
    cdef public tuple function_gradient(np.ndarray[np.float64_t, ndim=1] x, dict args)

    @staticmethod
    cdef public tuple function_gradient_hessian(np.ndarray[np.float64_t, ndim=1] x, dict args)

cdef class Omega:

    cdef public long R
    cdef public np.ndarray value 

    cdef update(self, zeta, Alpha alpha)
