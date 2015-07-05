from scipy import special
import numpy as np
cimport numpy as np
cimport cython

from numpy import empty, dot
from libc.math cimport cos, acos, sqrt


legendre = special.legendre
jn_zeros = special.jn_zeros
j0 = special.j0


@cython.boundscheck(False)
@cython.wraparound(False)
def HCM(int n, np.ndarray[np.double_t, ndim=3] ifourvects, np.ndarray[np.double_t, ndim=3] jfourvects=None):
    cdef int idx, ievent, ivect, jvect
    cdef double max_norm = 10.
    cdef double dphi, bessel_zero
    assert n >= 0, 'n must be >= 0'
    if jfourvects is None:
        jfourvects = ifourvects
    elif ifourvects.shape[0] != jfourvects.shape[0]:
        raise ValueError("array lengths do not match")
    cdef np.ndarray[np.double_t, ndim=1] result = empty(ifourvects.shape[0], dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1] weights = empty(ifourvects.shape[1] * jfourvects.shape[1])
    cdef np.ndarray[np.double_t, ndim=1] delta_r = empty(ifourvects.shape[1] * jfourvects.shape[1])
    cdef np.ndarray[np.double_t, ndim=1] bessel_delta_r
    cdef np.ndarray[np.double_t, ndim=1] C_nm
    bessel_zero = jn_zeros(0, n)[n - 1]
    for ievent in range(ifourvects.shape[0]):
        for ivect in range(ifourvects.shape[1]):
            for jvect in range(jfourvects.shape[1]):
                idx = ivect * jfourvects.shape[1] + jvect
                # pTi * pTj
                weights[idx] = ifourvects[ievent, ivect, 4] * jfourvects[ievent, jvect, 4]
                dphi = acos(cos(ifourvects[ievent, ivect, 6] - jfourvects[ievent, jvect, 6]))
                delta_r[idx] = sqrt((ifourvects[ievent, ivect, 5] - jfourvects[ievent, jvect, 5])**2 + dphi**2)
        weights /= weights.sum()
        delta_r /= max_norm
        bessel_delta_r = j0(bessel_zero * delta_r)
        C_nm = bessel_delta_r * weights
        result[ievent] = C_nm.sum()  # C_n
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def FWM(int n, np.ndarray[np.double_t, ndim=3] ifourvects, np.ndarray[np.double_t, ndim=3] jfourvects=None):
    cdef int idx, ievent, ivect, jvect
    assert n >= 0, 'n must be >= 0'
    if jfourvects is None:
        jfourvects = ifourvects
    elif ifourvects.shape[0] != jfourvects.shape[0]:
        raise ValueError("array lengths do not match")
    cdef np.ndarray[np.double_t, ndim=1] result = empty(ifourvects.shape[0], dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1] weights = empty(ifourvects.shape[1] * jfourvects.shape[1])
    cdef np.ndarray[np.double_t, ndim=1] cos_theta = empty(ifourvects.shape[1] * jfourvects.shape[1])
    cdef np.ndarray[np.double_t, ndim=1] legendre_nm
    cdef np.ndarray[np.double_t, ndim=1] C_nm
    legendre_n = legendre(n)
    for ievent in range(ifourvects.shape[0]):
        for ivect in range(ifourvects.shape[1]):
            for jvect in range(jfourvects.shape[1]):
                idx = ivect * jfourvects.shape[1] + jvect
                # |p|i * |p|j
                weights[idx] = ifourvects[ievent, ivect, 0] * jfourvects[ievent, jvect, 0]
                # cosine of 3d angle
                cos_theta[idx] = (ifourvects[ievent, ivect, 1] * jfourvects[ievent, jvect, 1] +
                                  ifourvects[ievent, ivect, 2] * jfourvects[ievent, jvect, 2] +
                                  ifourvects[ievent, ivect, 3] * jfourvects[ievent, jvect, 3]) / weights[idx]
        weights /= weights.sum()
        legendre_nm = legendre_n(cos_theta)
        C_nm = legendre_nm * weights
        result[ievent] = C_nm.sum()  # C_n
    return result
