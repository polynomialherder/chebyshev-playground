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



if __name__ == '__main__':
    X = np.linspace(-5, 5, 1000000)

    base_path = Path(f"PolynomialInterpolation")
    for _ in range(1000):
        C = ChebyshevPolynomial(X=X, n=2)
        parent = base_path / f"{C.n}"
        parent.mkdir(exist_ok=True)


        v = C.critical_points
        #Xp = np.array([v[0], v[1], v[4], v[5], v[8], v[9]])
        xv, yv, zv = C.grid(n=1000)
        aczv = abs(C(zv))
        Xp = np.array([v[0], v[1], v[2], v[3]])

        Xp = np.sort(Xp)
        polys = C.lagrange_polynomials(Xp)
        #polys = [np.poly1d(poly.coef[1::]) for poly in polys]

        # pos neg neg pos
        p1 = polys[2] + polys[3]
        p2 = polys[0]
        p3 = polys[1]
        #p2 = np.poly1d(polys[0][::-1])
        p = [
            p1, p2, p3
        ]

        S = range(len(p))
        holds = np.ones(zv.shape)
        summed = 0
        for m, n in product(S, S):
            prod_ = (p[m](zv)*p[n](zv).conjugate()).real
            holds = np.logical_and(holds, prod_ >= 0)
            summed += prod_

        if not np.all(np.isclose(summed, aczv**2, atol=0.001)):
            print(f"Sum is not close to |C(z)|**2")
            continue

        # erdos_holds = np.ones(zv.shape)
        # for i in range(50000):
        #     p = ChebyshevPolynomial(X=X, n=C.n)
        #     p = C.normalize_polynomial(p.polynomial)
        #     erdos_holds = np.logical_and(erdos_holds, abs(p(zv)) <= aczv)

        fig, ax = plt.subplots(figsize=(30, 17.5))
        ax.contourf(xv, yv, holds)
        ax.set_title(f"Region where inequality 5.2 holds")
        C.plot_disks(ax=ax)
        ax.plot(Xp, np.zeros(len(Xp)), "o", color="red")
        ax.plot(C.E_midpoint, 0, "o", color="blue")
        ax.set_xlim(C.E_midpoint - C.E_disk_radius, C.E_midpoint + C.E_disk_radius)
        ax.set_ylim(-C.E_disk_radius, C.E_disk_radius)
        fig.savefig(parent / f"{C.id}.png")

        for idx, qr in enumerate(zip(C.Ek_midpoints, C.Ek_radii)):
            q, r = qr
            Xm = np.linspace(q - r, q + r, 1000)
            Ym = np.linspace(-r, r, 1000)
            xv, yv = np.meshgrid(Xm, Ym)
            zv = xv + 1j*yv

            holds = np.ones(zv.shape)
            summed = 0
            for m, n in product(S, S):
                prod_ = (p[m](zv)*p[n](zv).conjugate()).real
                holds = np.logical_and(holds, prod_ > 0)
                summed += prod_

            # if not np.all(np.isclose(summed, aczv**2, atol=0.001)):
            #     print(f"Sum is not close to |C(z)|**2")
            #     continue


            fig, ax = plt.subplots(figsize=(30, 17.5))
            ax.contourf(xv, yv, holds)
            ax.set_title(f"Region where inequality 5.2 holds")
            C.plot_disks(ax=ax)
            ax.plot(Xp, np.zeros(len(Xp)), "o", color="red")
            ax.plot(C.E_midpoint, 0, "o", color="blue")
            ax.set_xlim(q - r, q + r)
            ax.set_ylim(-r, r)
            fig.savefig(parent / f"{C.id}-E{idx}.png")
