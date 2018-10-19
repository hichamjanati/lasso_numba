cimport cython
cimport numpy as np
import numpy as np
from cython cimport floating
from libc.math cimport fabs, sqrt
from scipy.linalg.cython_blas cimport ddot, dasum, daxpy, dnrm2, dcopy, dscal
from scipy.linalg.cython_blas cimport sdot, sasum, saxpy, snrm2, scopy, sscal


cdef floating fsign(floating f) nogil:
    if f == 0:
        return 0
    elif f > 0:
        return 1.0
    else:
        return -1.0

cdef floating fdot(int * n, floating * x, int * inc1, floating * y,
                        int * inc2) nogil:
    if floating is double:
        return ddot(n, x, inc1, y, inc2)
    else:
        return sdot(n, x, inc1, y, inc2)

cdef void faxpy(int * n, floating * alpha, floating * x, int * incx,
                 floating * y, int * incy) nogil:
    if floating is double:
        daxpy(n, alpha, x, incx, y, incy)
    else:
        saxpy(n, alpha, x, incx, y, incy)


cdef void fcopy(int * n, floating * x, int * incx, floating * y,
                     int * incy) nogil:
    if floating is double:
        dcopy(n, x, incx, y, incy)
    else:
        scopy(n, x, incx, y, incy)


cdef floating fnrm2(int * n, floating * x, int * inc) nogil:
    if floating is double:
        return dnrm2(n, x, inc)
    else:
        return snrm2(n, x, inc)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef floating[:] ct_solver(int n_samples, int n_features,
                           floating[::1, :] X, floating[:] y,
                           floating[:] norms_X_col,
                           floating[:] beta, floating[:] R,
                           floating alpha,
                           int maxiter=200) nogil:
    """CD solver for l2 reg kl interpolation."""

    cdef:
        int inc = 1
        floating tmp
        floating mbetaj

    fcopy(&n_samples, &y[0], &inc, &R[0], &inc)
    for i in range(maxiter):
        maxw = 0.
        for j in range(n_features):
            # tmp is the prox argument
            if beta[j] != 0.:
                faxpy(&n_samples, &beta[j], &X[0, j], &inc, &R[0], &inc)
                #R += X[:, j] * beta[j]

            #tmp = X[:, j].dot(R)
            tmp = fdot(&n_samples, &R[0], &inc, &X[0, j], &inc)
            # l1 thresholding
            beta[j] = fsign(tmp) * max(fabs(tmp) - alpha, 0) / norms_X_col[j]

            if beta[j] != 0.:
                mbetaj = - beta[j]
                faxpy(&n_samples, &mbetaj, &X[0, j], &inc, &R[0], &inc)
                # R += - beta[j] * X[:, j]

    return beta


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def lasso_ct(floating[::1, :] X, floating[:] y, floating alpha, int maxiter=200):

    cdef:
        int n_samples
        int n_features
        int inc = 1
    n_samples = X.shape[0]
    n_features = X.shape[1]

    if floating is double:
        dtype = np.float64
    else:
        dtype = np.float32
    cdef:
        floating[:] sol
        floating[:] R = np.empty(n_samples, dtype=dtype)
        floating[:] beta = np.zeros(n_features, dtype=dtype)
        floating[:] norms_X_col = np.empty(n_features, dtype=dtype)
    # compute norms_X_col
    for j in range(n_features):
        norms_X_col[j] = fdot(&n_samples, &X[0, j], &inc, &X[0, j], &inc)
    with nogil:
        sol = ct_solver(n_samples, n_features, X, y, norms_X_col, beta, R, alpha, maxiter)

    return np.asarray(sol)
