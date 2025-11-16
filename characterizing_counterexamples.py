"""
When does the Chebyshev polynomial T(z) lose to another polynomial P(z) on all points on a disk?

We know from refinement-to-gaps that this happens when |T(z)| \le |P(z)| on a subset of n+1 extremal points of P
Are there other situations?

"""

from chebyshev import ChebyshevPolynomial

import numpy as np
from scipy.interpolate import lagrange

def test1(m):
    """
    This example seems to show that this does not hold for consecutive unions of disks -- 
    normalization on the gap itself is not enough.

    """
    T = ChebyshevPolynomial.classical(2)

    P = lagrange([-m, -1, 1, m ], [1, -1, -1, 1])
    if np.isclose(P.coef[0], 0, atol=1e-10):
        P = np.poly1d(P.coef[1:])
    P = ChebyshevPolynomial(polynomial=P)

    T.comparison_plot(P)
    return T, P


def test2(d, m):
    """
    This example seems to show that this does not hold for consecutive unions of disks -- 
    normalization on the gap itself is not enough.

    """
    T = ChebyshevPolynomial.classical(d)

    P = lagrange([-1, 0, 1], [-1, m, -1])
    if np.isclose(P.coef[0], 0, atol=1e-10):
        P = np.poly1d(P.coef[1:])
    P = ChebyshevPolynomial(polynomial=P)

    T.comparison_plot(P)
    return T, P


if __name__ == '__main__':
    m = 2
    test1(m)
