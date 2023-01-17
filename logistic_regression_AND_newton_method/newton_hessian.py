import numpy as np
from numba import njit, prange
from numpy import ndarray, arange
from derivativeTest import derivativeTest


@njit()
def phi(t: ndarray) -> ndarray:
    l = (t <= 12) * t
    l = np.log(1 + np.exp(l))
    r = l * (t <= 12) + (t > 12) * t
    return r


@njit()
def d_phi(x: ndarray) -> ndarray:
    l = (x <= 12) * x
    l = np.exp(l) / (1 + np.exp(l))
    r = l * (x <= 12) + (x > 12)
    return r
    # if x >= 12:
    #     return 1
    # else:
    #     return np.exp(x) / (1 + np.exp(x))


@njit()
def dd_phi(x: ndarray) -> ndarray:
    cl = (x <= 12) | (x >= -12)
    r = cl * x
    r = (np.exp(r) / np.power((1 + np.exp(r)), 2)) * cl
    return r


def phi_(t: ndarray) -> ndarray:
    l = (t <= 12) * t
    l = np.log(1 + np.exp(l))
    r = l * (t <= 12) + (t > 12) * t
    return r


def d_phi_(x: ndarray) -> ndarray:
    l = (x <= 12) * x
    l = np.exp(l) / (1 + np.exp(l))
    r = l * (x <= 12) + (x > 12)
    return r


def dd_phi_(x: ndarray) -> ndarray:
    cl = (x <= 12) | (x >= -12)
    r = cl * x
    r = (np.exp(r) / np.power((1 + np.exp(r)), 2)) * cl
    return r


@njit()
def logisticFun_numba(x: ndarray, A: ndarray, b: ndarray):
    n, d = A.shape[0], A.shape[1]
    Ax = A @ x
    v = phi(Ax) - b * Ax
    f = np.sum(v)
    df = np.zeros(d)
    for j in range(d):
        for i in range(n):
            # df[j] += d_phi(A[i] @ x) * A[i, j] - b[i] * A[i, j]
            df[j] += (d_phi(A[i] @ x) * A[i, j] - b[i] * A[i, j])[0]
    hf = np.zeros((d, d))
    dd_phi_Ax = dd_phi(A @ x)
    for i in range(d):
        for j in range(d):
            for k in range(n):
                # hf[i, j] += dd_phi(A[k] @ x) * A[k, i] * A[k, j]
                hf[i, j] += (dd_phi_Ax[k] * A[k, i] * A[k, j])[0]
    return f, df, hf


def logisticFun_python(x: ndarray, A: ndarray, b: ndarray):
    n, d = A.shape[0], A.shape[1]
    Ax = A @ x
    v = phi_(Ax) - b * Ax
    f = np.sum(v)
    df = np.zeros(d)
    for j in range(d):
        for i in range(n):
            df[j] += d_phi_(A[i] @ x) * A[i, j] - b[i] * A[i, j]
    hf = np.zeros((d, d))
    dd_phi_Ax = dd_phi_(A @ x)
    for i in range(d):
        for j in range(d):
            for k in range(n):
                hf[i, j] += dd_phi_Ax[k] * A[k, i] * A[k, j]
    return f, df, hf


@njit()
def hessian_vector_product(x: ndarray, A: ndarray, b: ndarray, v: ndarray):
    n, d = A.shape[0], A.shape[1]
    eps = 1e-5
    dfp = np.zeros(d)
    dfm = np.zeros(d)
    tp = x + eps * v
    tm = x - eps * v
    # O(nd) time complexity
    for j in range(d):
        for i in range(n):
            dfp[j] += (d_phi(A[i] @ tp) * A[i, j] - b[i] * A[i, j])[0]
            dfm[j] += (d_phi(A[i] @ tm) * A[i, j] - b[i] * A[i, j])[0]
    hv = 1 / (2 * eps) * (dfp - dfm)
    return hv


def hvp_wrapper(x: ndarray, A: ndarray, b: ndarray):
    n, d = A.shape[0], A.shape[1]
    Ax = A @ x
    v = phi(Ax) - b * Ax
    f = np.sum(v)
    df = np.zeros(d)
    for j in range(d):
        for i in range(n):
            df[j] += (d_phi(A[i] @ x) * A[i, j] - b[i] * A[i, j])[0]
    return f, df, lambda v: hessian_vector_product(x, A, b, v)


@njit()
def logisticFunl2(x: ndarray, A: ndarray, b: ndarray, c: float = 1.0):
    x, b = x.reshape(-1, 1), b.reshape(-1, 1)
    n, d = A.shape[0], A.shape[1]
    Ax = A @ x
    v = phi(Ax) - b * Ax
    f = np.sum(v)
    f = f + c / 2 * np.linalg.norm(x, 2) ** 2
    df = np.zeros(d)
    df = df.reshape(-1, 1)
    d_phi_Ax = d_phi(A @ x)
    for j in range(d):
        for i in range(n):
            df[j] += (d_phi_Ax[i] * A[i, j] - b[i] * A[i, j])[0]
    df = df + c * x
    hf = np.zeros((d, d))
    dd_phi_Ax = dd_phi(A @ x)
    for i in range(d):
        for j in range(d):
            for k in range(n):
                hf[i, j] += (dd_phi_Ax[k] * A[k, i] * A[k, j])[0]
    hf = hf + np.eye(d) * c
    return f, df, hf


@njit
def evalf(x: ndarray, A: ndarray, b: ndarray, c: float = 1.0):
    x, b = x.reshape(-1, 1), b.reshape(-1, 1)
    n, d = A.shape[0], A.shape[1]
    Ax = A @ x
    v = phi(Ax) - b * Ax
    f = np.sum(v)
    f = f + c / 2 * np.linalg.norm(x, 2) ** 2
    return f


@njit()
def gradient_descent(A: ndarray, b: ndarray, c: float = 1.0):
    n, d = A.shape[0], A.shape[1]
    x = np.zeros(d)
    x, b = x.reshape(-1, 1), b.reshape(-1, 1)
    k, cond = 0, 1
    lr = 10 / (1 / 4 * np.linalg.norm(A) ** 2 + c)
    while cond:
        k += 1
        df = np.zeros(d)
        df = df.reshape(-1, 1)
        d_phi_Ax = d_phi(A @ x)
        for j in range(d):
            for i in range(n):
                df[j] += (d_phi_Ax[i] * A[i, j] - b[i] * A[i, j])[0]
        df = df + c * x
        lr = 0.9 * lr
        x = x - lr * df
        tc = np.linalg.norm(df)
        cond = (tc >= 1e-4) & (k < 1000)
    return x


def newton_CG(A: ndarray, b: ndarray, c: float = 1.0):
    n, d = A.shape[0], A.shape[1]
    x = np.zeros(d)
    x = x.reshape(-1, 1)
    b = b.reshape(-1, 1)
    k, cond = 0, 1
    while cond:
        _, df, hf = logisticFunl2(x, A, b, c)
        x = x - np.linalg.inv(hf) @ df
        cond = (np.linalg.norm(df) >= 1e-6) & (k < 1000)
        k += 1
    return x


def sigmoid(A, x):
    return 1 / (1 + np.exp(-(A @ x)))


@njit()
def agd(A: ndarray, b: ndarray, c: float = 1.0):
    k = 4
    a = (np.sqrt(k) - 1) / (np.sqrt(k) + 1)
    n, d = A.shape[0], A.shape[1]
    x = np.zeros(d)
    lg = 1 / 4 * np.linalg.norm(A) ** 2 + c
    x, b = x.reshape(-1, 1), b.reshape(-1, 1)
    x_last = x
    for m in range(1000):
        y = x + a * (x - x_last)
        x_last = x
        df = np.zeros(d)
        df = df.reshape(-1, 1)
        d_phi_Ax = d_phi(A @ x)
        for j in range(d):
            for i in range(n):
                df[j] += (d_phi_Ax[i] * A[i, j] - b[i] * A[i, j])[0]
        df = df + c * y
        x = y - df / lg
        tc = np.linalg.norm(df)
        if (tc <= 1e-4) & (m < 1000):
            break
    return x


n, d = 1000, 50
A = np.random.randn(n, d)
I = np.eye(2, 1)
ind = np.random.randint(2, size=n)
b = I[ind, :]
x = np.ones((d, 1)).reshape(-1, 1)
f, df, hf = logisticFun_numba(x, A, b)
# # fun = lambda x: logisticFun(x, A, b)
# # fun = lambda x: hvp_wrapper(x, A, b)
# fun = lambda x: logisticFunl2(x, A, b)
# derivativeTest(fun, np.ones((d, 1)))
# # x = gradient_descent(A, b)

# n1 = []
# fval = []
# A_train, b_train, A_test, b_test = loadData()
# # b_train = b_train * 1.00
# # x = gradient_descent(A_train, b_train)
# x = newton_CG(A_train, b_train)
# y_proba = sigmoid(A_test, x)

# from rule_threshold import plot_roc
# plot_roc(b_test[:,0], y_proba, 'na')
# y_l = y_p.predict(A_test)
