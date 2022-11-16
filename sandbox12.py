import time

from chebyshev import ChebyshevPolynomial

import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import norm


def plot_holds(c, holds, savefig=None):
    fig, ax = plt.subplots()
    xv, yv, zv = c.grid()
    ax.contourf(xv, yv, holds)
    c.plot_disks(ax=ax)
    if savefig is None:
        fig.show()
    else:
        fig.savefig(savefig)


def test_roots(C, roots, critical_normalize=False, savefig=None, plot_g=False, t=1, ax=None, fig=None):
    """ TODO: Add another function that normalizes only at the extremal points
    """
    p = np.poly1d(roots, r=True)
    if critical_normalize:
        norm_p = norm(p(C.critical_points), np.inf)
        p3b = p/norm_p
    else:
        p3b = C.normalize_polynomial(p)
    pm = C.polynomial - p3b
    pp = C.polynomial + p3b
    xv, yv, zv = C.grid()
    if not ax:
        fig, ax = plt.subplots()
    ax.contourf(xv, yv, abs(C(zv)) >= abs(p3b(zv)))
    C.plot_disks(ax=ax)
    C.plot_roots(ax=ax)
    ax.plot(roots.real, roots.imag, "x", color="blue")

    if plot_g:
        # for mr, pr in zip(sorted(pm.r), sorted(pp.r)):
        #     ax.plot(pr, 0, "o", color="blue")
        #     ax.plot(mr, 0, "o", color="red")
        #     if not np.isclose(pr, mr, atol=1e-4):
        #         disk_midpoint = (mr + pr)/2
        #         disk_radius = abs(mr - pr)/2
        #         disk =  plt.Circle((disk_midpoint, 0), disk_radius, fill=False, color="purple", linestyle="--")
        #         ax.add_patch(disk)

        g = C.polynomial + C.normalize_polynomial(p)*np.e**(1j*t)

        for gr in sorted(g.r):
            ax.plot(gr.real, gr.imag, "o", color="blue")

    for mp in C.Ek_midpoints:
        ax.plot(mp, 0, "x", color="pink")
    ax.set_xlim(zv.real.min(), zv.real.max())
    ax.set_ylim(zv.imag.min(), zv.imag.max())
    if fig:
        if savefig is None:
            fig.show()
        else:
            fig.suptitle("Locations where $|C_n| \ge |p_n|$")
            r = tuple(sorted(np.round(roots, 3)))
            rc = tuple(sorted(np.round(C.r, 3)))
            ax.set_title("$r_p = " + f"{r}" + "$\n" + "$r_C = " + f"{rc}" + "$")
            fig.tight_layout()
            fig.savefig(savefig)


def circle_points(r, q, N=100):
    theta = np.linspace(0, 2 * np.pi, N+1)
    #theta = theta[0:-1]
    X = q + r*np.cos(theta)
    Y = r*np.sin(theta)
    return X + 1j*Y



def trials(c, critical_normalize=True, n=1000):
    maxima = c.critical_points
    xv, yv, zv = c.grid()
    holds = np.ones(zv.shape)
    czv = np.abs(c(zv))
    E = c.E
    Emin = E.min()
    Emax = E.max()
    polynomials = []
    for i in range(n):
        p = np.poly1d(np.random.uniform(-50, 50, 2), r=True)
        if critical_normalize:
            eta_p = norm(p(maxima), np.inf)
            pp = p/eta_p
        else:
            pp = c.normalize_polynomial(p)
        ppzv = np.abs(pp(zv))
        holds_pp = ppzv/czv <= 1
        holds = np.logical_and(holds, holds_pp)
        if not ((i+1) % 1000):
            print(f"Checked against {i+1} trial polynomials")
    return holds, polynomials


if __name__ == '__main__':
    X = np.linspace(-10, 10, 1000000)
    c3 = ChebyshevPolynomial(X=X, n=3)
    # Generate an array corresponding to the boundary points of all Ek disks
    d = c3.Ek_circle_points(n=1000)
    start_time = time.time()
    # Generate random comparison polynomials
    trials = 3000000
    print(f"Generating {trials:,} random comparison polynomials")
    comparisons = [np.poly1d(np.random.uniform(-10, 10, 3), r=True) for _ in range(trials)]
    print(f"Checking {c3.id} with roots at {c3.r} against {trials} comparison polynomials")
    for j, p in enumerate(comparisons):
        pp = c3.normalize_polynomial(p)
        ratio = np.abs(pp(d)/c3(d))
        gt = np.logical_and(ratio > 1, ~np.isclose(ratio, 1, atol=1e-5))
        # If any elements of the logical array are True, print out a message with the index of
        # the polynomial in the comparisons array along so that it can be checked later. Include the
        # number of failing points and a sample of points and values at those points for which
        # the inequality fails to hold
        if any(gt):
            points = d[gt]
            print(f"Found one: {j=} {pp.r} at {points.size} points")
            rand = np.random.choice(points, 2)
            print(f"e.g. {abs(c3(rand))} vs {abs(pp(rand))} at x={rand}")
        # Print a message every 1000 comparisons just for progress tracking
        if j and not (j % 1000):
            elapsed = time.time() - start_time
            print(f"Checked against {j} polynomials in {elapsed}s")
    print(f"Run finished. Took {time.time() - start_time}s")
