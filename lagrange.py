from dataclasses import dataclass, field
from typing import List

import numpy as np

@dataclass
class LagrangePolynomial:
    _nodes: list = field(default_factory=list)
    _values: list = field(default_factory=list)

    @property
    def nodes(self):
        if not isinstance(self._nodes, np.ndarray):
            return np.array(self._nodes)
        return self._nodes
    

    @property
    def values(self):
        if not isinstance(self._values, np.ndarray):
            return np.array(self._values)
        return self._values
    

def ell(L: LagrangePolynomial):
    return np.poly1d(L.nodes, r=True)


def barycentric_weights(L: LagrangePolynomial):
    weights = []
    for k in range(L.nodes.size):
        prod = 1
        for j in range(L.nodes.size):
            if k == j:
                continue
            prod *= 1/(L.nodes[k] - L.nodes[j])
        weights.append(prod)
    return np.array(weights)


def displacement(L: LagrangePolynomial, z: complex):
    return barycentric_weights(L)/(z - L.nodes)


def evaluate(L: LagrangePolynomial, z: complex):
    d = displacement(L, z)
    return ell(L)(z)*(d@L.values)


def basis_polynomials(L: LagrangePolynomial):
    l = ell(L)
    polys = []
    weights = barycentric_weights(L)
    for x, wk in zip(L.nodes, weights):
        lk, _ = l/np.poly1d([x], r=True)
        polys.append(lk*wk)
    return polys



def kernel(L: LagrangePolynomial, z: complex):
    K = []
    B = basis_polynomials(L)
    Bz = np.array([b(z).conjugate() for b in B])
    for l in B:
        K.append(l(z)*Bz)
    return np.array(K)


def square_modulus(L: LagrangePolynomial, z: complex):
    l = ell(L)
    d = displacement(L, z)
    terms = d*L.values
    first_factor = (l(z)*l(z).conjugate())
    second_factor = np.array([term*terms.conjugate() for term in terms])
    return first_factor*second_factor


def sequence_sqmod(left, right):
    return np.array([t*right.conjugate() for t in left])


def square_modulus(L: LagrangePolynomial, z: complex):
    l = ell(L)
    d = displacement(L, z)
    terms = d*L.values
    first_factor = (l(z)*l(z).conjugate())
    second_factor = sequence_sqmod(terms, terms)
    return first_factor*second_factor


def square_modulus_1d(L: LagrangePolynomial, z: complex):
    return sum(square_modulus(L, z))

