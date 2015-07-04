from scipy import special
import numpy as np
cimport numpy as np
cimport cython

from numpy import empty, dot
from libc.math cimport cos, acos, sqrt


Legendre = special.legendre
jn_zeros = special.jn_zeros
j0 = special.j0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double HCM_single(int n, np.ndarray[np.double_t, ndim=2] ifourvects, np.ndarray[np.double_t, ndim=2] jfourvects):
    cdef double max_norm = 10.
    cdef double dphi, bessel_zero
    cdef int i, j, idx
    cdef np.ndarray[np.double_t, ndim=1] weights = empty(ifourvects.shape[0] * jfourvects.shape[0])
    cdef np.ndarray[np.double_t, ndim=1] delta_r = empty(ifourvects.shape[0] * jfourvects.shape[0])
    for i in range(ifourvects.shape[0]):
        for j in range(jfourvects.shape[0]):
            idx = i * jfourvects.shape[0] + j
            # pTi * pTj
            weights[idx] = ifourvects[i, 4] * jfourvects[j, 4]
            dphi = acos(cos(ifourvects[i, 6] - jfourvects[j, 6]))
            delta_r[idx] = sqrt((ifourvects[i, 5] - jfourvects[j, 5])**2 + dphi**2)
    weights /= weights.sum()
    delta_r /= max_norm
    bessel_zero = jn_zeros(0, n)[n - 1]
    cdef np.ndarray[np.double_t, ndim=1] bessel_delta_r = j0(bessel_zero * delta_r)
    cdef np.ndarray[np.double_t, ndim=1] C_nm = bessel_delta_r * weights
    return C_nm.sum() # C_n


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double FWM_single(int n, np.ndarray[np.double_t, ndim=2] ifourvects, np.ndarray[np.double_t, ndim=2] jfourvects):
    cdef int i, j, idx
    cdef np.ndarray[np.double_t, ndim=1] weights = empty(ifourvects.shape[0] * jfourvects.shape[0])
    cdef np.ndarray[np.double_t, ndim=1] cos_theta = empty(ifourvects.shape[0] * jfourvects.shape[0])
    for i in range(ifourvects.shape[0]):
        for j in range(jfourvects.shape[0]):
            idx = i * jfourvects.shape[0] + j
            # |p|i * |p|j
            weights[idx] = ifourvects[i, 0] * jfourvects[j, 0]
            # cosine of 3d angle
            cos_theta[idx] = (ifourvects[i, 1] * jfourvects[j, 1] +
                              ifourvects[i, 2] * jfourvects[j, 2] +
                              ifourvects[i, 3] * jfourvects[j, 3]) / weights[idx]
    weights /= weights.sum()
    cdef np.ndarray[np.double_t, ndim=1] legendre_nm = Legendre(n)(cos_theta)
    cdef np.ndarray[np.double_t, ndim=1] C_nm = legendre_nm * weights
    return C_nm.sum() # C_n


@cython.boundscheck(False)
@cython.wraparound(False)
def HCM(int n, np.ndarray[np.double_t, ndim=3] ifourvects, np.ndarray[np.double_t, ndim=3] jfourvects=None):
    cdef int idx
    assert n >= 0, 'n must be >= 0'
    if jfourvects is None:
        jfourvects = ifourvects
    elif ifourvects.shape[0] != jfourvects.shape[0]:
        raise ValueError("array lengths do not match")
    cdef np.ndarray[np.double_t, ndim=1] result = empty(ifourvects.shape[0], dtype=np.double)
    for idx in range(ifourvects.shape[0]):
        result[idx] = HCM_single(n, ifourvects[idx], jfourvects[idx])
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def FWM(int n, np.ndarray[np.double_t, ndim=3] ifourvects, np.ndarray[np.double_t, ndim=3] jfourvects=None):
    cdef int idx
    assert n >= 0, 'n must be >= 0'
    if jfourvects is None:
        jfourvects = ifourvects
    elif ifourvects.shape[0] != jfourvects.shape[0]:
        raise ValueError("array lengths do not match")
    cdef np.ndarray[np.double_t, ndim=1] result = empty(ifourvects.shape[0], dtype=np.double)
    for idx in range(ifourvects.shape[0]):
        result[idx] = FWM_single(n, ifourvects[idx], jfourvects[idx])
    return result
