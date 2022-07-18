import hashlib

from itertools import combinations, permutations, product
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import inv
from numpy.ma import make_mask

from chebyshev import ChebyshevPolynomial


def test_polynomials(C, P):
    prod_ = np.zeros(zv.shape, dtype=np.complex128)
    G = []
    for i, j in product(range(len(P)), range(len(P))):
        G.append(P[i](zv)*(P[j](zv)).conjugate())
    return G


def test_roots(C, roots, savefig=None):
    p = np.poly1d(roots, r=True)
    p3b = C.normalize_polynomial(p)
    pm = C.polynomial - p3b
    pp = C.polynomial + p3b
    xv, yv, zv = C.grid()
    fig, ax = plt.subplots()
    ax.contourf(xv, yv, abs(C(zv)) >= abs(p3b(zv)))
    C.plot_disks(ax=ax)
    C.plot_roots(ax=ax)
    for mr, pr in zip(sorted(pm.r), sorted(pp.r)):
        ax.plot(pr, 0, "o", color="blue")
        ax.plot(mr, 0, "o", color="red")
        if not np.isclose(pr, mr, atol=1e-4):
            print(pr, mr)
            disk_midpoint = (mr + pr)/2
            disk_radius = abs(mr - pr)/2
            disk =  plt.Circle((disk_midpoint, 0), disk_radius, fill=False, color="purple", linestyle="--")
            ax.add_patch(disk)


    for mp in C.Ek_midpoints:
        ax.plot(mp, 0, "x", color="pink")
    if savefig is None:
        fig.show()
    else:
        fig.suptitle("Locations where $|C_n| \ge |p_n|$")
        r = tuple(sorted(np.round(roots, 3)))
        rc = tuple(sorted(np.round(C.r, 3)))
        ax.set_title("$r_p = " + f"{r}" + "$\n" + "$r_C = " + f"{rc}" + "$")
        fig.tight_layout()
        fig.savefig(savefig)


def get_region_of_interest(C):
    _, _, zv = C.grid()
    outside_gap = make_mask(zv)
    for midpoint, radius in zip(C.gap_midpoints, C.gap_radii):
        distance = abs(midpoint - zv)
        outside_gap = outside_gap & (distance > radius)

    outside_Ek = make_mask(zv)
    for midpoint, radius in zip(C.Ek_midpoints, C.Ek_radii):
        distance = abs(midpoint - zv)
        outside_Ek = outside_Ek & (distance > radius)

    inside_E_disk = abs(C.E_midpoint - zv) < C.E_disk_radius
    return inside_E_disk & outside_Ek #& outside_gap


def vandermonde(Xp):
    return np.array([[vi**i for i in range(len(Xp))] for vi in Xp])


def least_squares_basis(C, Xp):
    V = vandermonde(Xp)
    return (inv(V)*C(Xp)).transpose()


def circle_points(r, q, N=100):
    theta = np.linspace(0, 2 * np.pi, N+1)
    theta = theta[0:-1]
    X = q + r*np.cos(theta)
    Y = r*np.sin(theta)
    return X + 1j*Y


if __name__ == '__main__':
    X = np.linspace(-5, 5, 1000000)
    classical = ChebyshevPolynomial(polynomial=np.poly1d([4, 0, -3, 0]), X=X)
    test = lambda z: eta*(z - classical.r[0])*(z - classical.r[1])*(z - classical.r[2])/((z - p.r[0])*(z - p.r[1]))
    p = np.poly1d([-.75, 0.75], r=True)
    pp = classical.normalize_polynomial(p)
    xv, yv, zv = classical.grid()
    part = classical.partial_fractions_comparison(p.r)
    part2 = classical.partial_fractions_comparison([-0.9, 0, 0.9])
