import numpy as np

import matplotlib.pyplot as plt

from chebyshev import ChebyshevPolynomial
from lagrange import LagrangePolynomial

from scipy.linalg import norm
from scipy.interpolate import lagrange


if __name__ == '__main__':

    X = np.linspace(-1, 1, 1000000)

    # A set of n+1 nodes with 3 alternations
    nodes = [-1, -0.8, 0, 1]
    known_values = [-1, 1, -1, 1]
    C = ChebyshevPolynomial(
        3, X, nodes=nodes, known_values=known_values
    )
    l1 = lagrange(nodes, known_values)

    # A set of n+3 nodes with 3 alternations
    nodes2 = [-1, -0.8, -0.5, 0, 0.4, 1]
    known_values2 = [-1, 1, 1, -1, -1, 1]
    C2 = ChebyshevPolynomial(
        3, X, nodes=nodes2, known_values=known_values2
    )
    l2 = lagrange(nodes2, known_values2)

    # Generate a random Chebyshev polynomial
    C = ChebyshevPolynomial(3, X)

    # Fix a set of nodes
    v = C.critical_points #[-1.5, -0.8, -0.7, -0.2, 0, 0.5, 0.8, 1.5]
    Cv = C(v)

    l3 = lagrange(v, Cv)

    # Interpolating on v gives a 7th degree polynomial with several "large"
    # roots, and a few complex roots with large imaginary parts
    C_roots = C.polynomial.r
    l3_roots = l3.r
