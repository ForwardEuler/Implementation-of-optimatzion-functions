import numpy as np


def lineSearchWolfeStrong(objFun, xk, pk, alpha0=1, c1=1e-4, c2=0.9, linesearchMaxItrs=200):
    """    
    All vector are column vectors.
    INPUTS:
        objFun: a function handle of both objective function and its gradient
        xk: starting xk
        pk: direction pk
        alpha0: initial step-size
        c1: parameter of Armijo line-search
        c2: parameter of strong Wolfe condition
        linesearchMaxItrs: maximum iteration of line search with strong Wolfe
        
    OUTPUTS:
        alpha: proper step-size
        T: iterations
    """
    itrLS = 0
    itrs2 = 0
    a1 = 0
    #    a2 = infun(0, amax)
    a2 = alpha0
    f0, g0 = objFun(xk)
    fb = f0  # f preview
    while itrLS < linesearchMaxItrs:
        fa, ga = objFun(xk + a2 * pk)
        g0p = g0.T.dot(pk)
        if fa > f0 + a2 * c1 * g0p or (fa >= fb and itrLS > 0):
            alpha, itrs2 = zoomf(a1, a2, objFun, f0, fa, g0p, xk, pk, c1, c2, itrLS, linesearchMaxItrs)
            break
        gap = ga.T.dot(pk)
        if abs(gap) <= -c2 * g0p:
            alpha = a2
            itrs2 = 0
            break
        if gap >= 0:
            alpha, itrs2 = zoomf(a2, a1, objFun, f0, fa, g0p, xk, pk, c1, c2, itrLS, linesearchMaxItrs)
            # alpha, itrs2 = zoomf(a1, a2, objFun, f0, fa, g0p, xk, pk, c1, c2, itrLS, linesearchMaxItrs)
            break
        a2 = a2 * 2
        fb = fa
        itrLS += 1
    itrLS += itrs2
    #     if itrLS >= linesearchMaxItrs:
    #         alpha = 0
    return alpha, itrLS


def zoomf(a1, a2, objFun, f0, fa, g0p, xk, pk, c1, c2, itrLS, linesearchMaxItrs):
    itrs2 = 0
    while (itrs2 + itrLS) < linesearchMaxItrs:
        # quadratic
        itrs2 += 1
        # lower bound
        fa1, ga1 = objFun(xk + a1 * pk)
        ga1p = ga1.T.dot(pk)
        # upper bound
        fa2, ga2 = objFun(xk + a2 * pk)
        ga2p = ga2.T.dot(pk)
        aj = cubicInterp(a1, a2, fa1, fa2, ga1p, ga2p)
        a_mid = (a1 + a2) / 2
        if not inside(aj, a1, a_mid):
            aj = a_mid
        faj, gaj = objFun(xk + aj * pk)
        if faj > f0 + aj * c1 * g0p or faj >= fa1:
            a2 = aj
        else:
            gajp = gaj.T.dot(pk)
            if np.abs(gajp) <= -c2 * g0p:
                break
            if gajp * (a2 - a1) >= 0:
                a2 = a1
            a1 = aj
    return aj, itrs2


def cubicInterp(x1, x2, f1, f2, g1, g2):
    # find minimizer of the Hermite-cubic polynomial interpolating a
    # function of one variable, at the two points x1 and x2, using the
    # function (f1 and f2) and derivative (g1 and g2).
    # Nocedal and Wright Eqn (3.59)
    #    print(x1 - x2)
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    #    if d1**2 - g1*g2 <= 0:
    #        d2 = 0
    #    else:
    #        d2 = np.sign(x2 - x1)*np.sqrt(d1**2 - g1*g2)
    d2 = np.sign(x2 - x1) * np.sqrt(d1 ** 2 - g1 * g2)
    xmin = x2 - (x2 - x1) * (g2 + d2 - d1) / (g2 - g1 + 2 * d2);
    return xmin


def inside(x, a, b):
    # test x \in (a, b) or not
    l = 0
    if not np.isreal(x):
        return l
    if a <= b:
        if x >= a and x <= b:
            l = 1
    else:
        if x >= b and x <= a:
            l = 1
    return l


def Ax(A, x):
    if callable(A):
        Ax = A(x)
    else:
        Ax = A.dot(x)
    return Ax
