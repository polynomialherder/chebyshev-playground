from chebyshev import ChebyshevPolynomial

import numpy as np

if __name__ == '__main__':
    X = np.linspace(-5, 5, 10000000)
    C = ChebyshevPolynomial(X=X, polynomial=np.poly1d([2.31235651, -5.9913754, -8.0986466, 21.21137011]))
    X = np.linspace(-5, 5, 1000)
    Y = np.linspace(-5, 5, 1000)

    xv, yv = np.meshgrid(X, Y)
    zv = xv + 1j*yv

    holds = np.ones(zv.shape)
    for n in range(1, 4):
        print(f"Performing comparisons with degree {n} polynomials")
        for i in range(50000):
            p = ChebyshevPolynomial(X=X, n=n)
            p = C.normalize_polynomial(p.polynomial)
            holds = np.logical_and(holds, abs(p(zv)) <= abs(C(zv)))
