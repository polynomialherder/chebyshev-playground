from chebyshev import ChebyshevPolynomial

import numpy as np

def blo_polynomial(n, a):
    A = np.sqrt(a**2 + 1)
    T = ChebyshevPolynomial.classical(n-1).polynomial
    U = ChebyshevPolynomial.classical(n-2, kind=2).polynomial
    # Note that numpy poly1d objects silently convert to arrays if multiplied on the left by a scalar
    # or array so we are careful to multiply with those on the right
    return (-np.poly1d([a, 1j])*T + np.poly1d([-1, 0, 1])*U*A)*(1/A)


if __name__ == '__main__':
    a = 2
    # Degree 2 maximizer at z0 = 2j
    # Q(z) = -(1 + 2/sqrt(5))z^2 - (1j/sqrt(5))z + 1 = 
    Q = blo_polynomial(2, 2)