import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rand
from numpy.linalg import norm


def derivativeTest(fun, x0):
    """
    INPUTS:
        fun: a function handle that gives f, g, Hv
        x0: starting point
    OUTPUTS:
        derivative test plots
    """
    x0 = x0.reshape(len(x0), 1)
    fun0 = fun(x0)
    dx = rand.randn(len(x0), 1)
    M = 20
    dxs = np.zeros((M, 1))
    firsterror = np.zeros((M, 1))
    seconderror = np.zeros((M, 1))

    for i in range(M):
        x = x0 + dx
        fun1 = fun(x)
        H0 = Ax(fun0[2], dx)
        firsterror[i] = abs(fun1[0] - (fun0[0] + np.dot(
            dx.T, fun0[1]))) / abs(fun0[0])
        seconderror[i] = abs(fun1[0] - (fun0[0] + np.dot(
            dx.T, fun0[1]) + 0.5 * np.dot(dx.T, H0))) / abs(fun0[0])
        print('First Order Error is %8.2e;   Second Order Error is %8.2e' % (
            firsterror[i], seconderror[i]))
        dxs[i] = norm(dx)
        dx = dx / 2

    step = [2 ** (-i - 1) for i in range(M)]
    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    plt.loglog(step, abs(firsterror), 'r', label='1st Order Error')
    plt.loglog(step, dxs ** 2, 'b', label='Theoretical Order')
    plt.gca().invert_xaxis()
    plt.legend()

    plt.subplot(212)
    plt.loglog(step, abs(seconderror), 'r', label='2nd Order Error')
    plt.loglog(step, dxs ** 3, 'b', label='Theoretical Order')
    plt.gca().invert_xaxis()
    plt.legend()

    return plt.show()


def Ax(A, x):
    if callable(A):
        Ax = A(x)
    else:
        Ax = A.dot(x)
    return Ax
