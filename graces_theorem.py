import matplotlib.pyplot as plt

import math

from chebyshev import ChebyshevPolynomial

import numpy as np

from scipy.linalg import norm, null_space

def apolarity_coefficients(p):
    return np.array([(-1)**(p.order-k)*(1/math.comb(p.order, k))*x for k, x in enumerate(p.coef[::-1])]).reshape((1, -1))

def apolarity_basis(p, as_array=False, weak=False):
    a = apolarity_coefficients(p)
    a = np.array([[0] + list(a[0])]) if weak else a
    coefs = null_space(a).T
    if as_array:
        return coefs
    non_normalized = [np.poly1d(coef) for coef in coefs]
    return [p/p.coef[0] for p in non_normalized]


def random_complex_number(radius_max, midpoint):
    radius = np.random.uniform(0, radius_max)
    angle = np.random.uniform(0, 2*np.pi)
    return radius*np.e**(1j*angle) + midpoint


if __name__ == '__main__':
    X = np.linspace(-10, 10, 1000)
    q = np.poly1d([0.5, 0.5])
    T = ChebyshevPolynomial.classical(2)
    Tq = T(q)
    basis = apolarity_basis(Tq, as_array=True)
    a = apolarity_coefficients(Tq)

    p = ChebyshevPolynomial(X=X, polynomial=np.poly1d([-1.5, 4], r=True)).polynomial
    pp = T.normalize_polynomial(p)(q)
    basis_p = apolarity_basis(pp, as_array=True)
    ap = apolarity_coefficients(pp)

    i = 0
    for t in np.linspace(0, 2*np.pi+np.pi/2, 20):
        g = Tq + pp*np.e**(t*1j)
        basis_g = apolarity_basis(g, as_array=True, weak=True)
        ag = np.array([0] + apolarity_coefficients(g)[0])

        indisk = []
        exc = 0
        for _ in range(50000):
            re = np.random.uniform(-100, 100, 3)
            im = np.random.uniform(-100, 100, 3)
            zz = np.random.choice([0, 1, 1j])*re + np.random.choice([0, 1, 1j])*im*1j
            try:
                poly = np.poly1d(basis_g.T@zz)
                assert poly.order == T.n+1
            except Exception as e:
                exc += 1
                continue
            distances = np.abs(poly.r)
            if np.all(distances <= 1):
                indisk.append(poly.r)

        print(f"{exc} exceptions")

        fig, ax = plt.subplots()
        indisk = np.array(indisk)
        ax.plot(indisk.flatten().real, indisk.flatten().imag, "o", color="red")
        TT = ChebyshevPolynomial(X=X, polynomial=Tq)
        TT.plot_disks(ax=ax)
        ax.plot(g.r.real, g.r.imag, "o", color="cornflowerblue")
        fig.suptitle(f"{t=:.2}")
        fig.savefig(f"{i:03}.png")
        i += 1
        print(i)