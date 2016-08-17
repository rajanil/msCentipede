
import numpy as np
cimport numpy as np

# defining some constants
EPS = np.finfo(np.double).tiny
MAX = np.finfo(np.double).max

# defining some simple functions

def logistic(np.ndarray x):

    y = x.copy()
    y[x>0] = np.exp(-x[x>0])/(1+np.exp(-x[x>0]))
    y[x<=0] = 1./(1+np.exp(x[x<=0]))
    return y

def insum(np.ndarray x, list axes):

    return np.apply_over_axes(np.sum,x,axes)

def nplog(x):

    return np.nan_to_num(np.log(x))

def outsum(np.ndarray arr):

    """
    Fast summation over the 0-th axis.
    """

    cdef list shape
    cdef np.ndarray thesum
    thesum = sum([a for a in arr])
    thesum = thesum.reshape(1,thesum.size)
    return thesum
