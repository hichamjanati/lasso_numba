import time
import numpy as np
from scipy import linalg
from numba import njit, jit, float64, int64
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()},
                  reload_support=True)
from lasso_ct import lasso_ct


def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('%r %2.2f sec' % (method.__name__, te - ts))
        return result

    return timed


@njit(float64(float64))
def fsign(f):
    if f == 0:
        return 0
    elif f > 0:
        return 1.0
    else:
        return -1.0


@timeit
def lasso_np(X, y, alpha, maxiter=200):
    n, p = X.shape
    beta = np.zeros(p)
    R = y.copy()
    norm_cols_X = (X ** 2).sum(axis=0)

    for n_iter in range(maxiter):
        for ii in range(p):
            beta_ii = beta[ii]
            # Get current residual
            if beta_ii != 0.:
                R += X[:, ii] * beta_ii
            tmp = np.dot(X[:, ii], R)
            # Soft thresholding
            beta[ii] = fsign(tmp) * max(abs(tmp) - alpha, 0) / norm_cols_X[ii]
            if beta[ii] != 0.:
                R -= X[:, ii] * beta[ii]
    return beta


@timeit
def lasso_ct_(X, y, alpha, maxiter=200):
    return lasso_ct(X, y, alpha, maxiter)


@timeit
@jit(float64[:](float64[::1, :], float64[::1], float64, int64),
     nopython=True, cache=True)
def lasso_nb(X, y, alpha, maxiter=200):
    n, p = X.shape
    beta = np.zeros(p)
    R = y.copy()
    norm_cols_X = (X ** 2).sum(axis=0)

    for n_iter in range(maxiter):
        for ii in range(p):
            beta_ii = beta[ii]
            # Get current residual
            if beta_ii != 0.:
                R += X[:, ii] * beta_ii
            tmp = np.dot(X[:, ii], R)
            # Soft thresholding
            beta[ii] = fsign(tmp) * max(abs(tmp) - alpha, 0) / norm_cols_X[ii]

            if beta[ii] != 0.:
                R -= X[:, ii] * beta[ii]
    return beta


if __name__ == '__main__':
    n_samples, n_features = 1000, 20000
    np.random.seed(0)
    X = np.random.randn(n_samples, n_features)
    Xf = np.asfortranarray(X)
    y = np.random.randn(n_samples)
    alpha, maxiter = 1., 200

    def pobj(beta):
        R = y - np.dot(X, x)
        print(0.5 * linalg.norm(R) ** 2 + alpha * np.sum(np.abs(beta)))

    x = lasso_np(X, y, alpha, maxiter)
    pobj(x)
    x = lasso_nb(Xf, y, alpha, maxiter)
    pobj(x)
    x = lasso_ct_(Xf, y, alpha, maxiter)
    pobj(x)
