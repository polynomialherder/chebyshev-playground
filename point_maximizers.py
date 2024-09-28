from functools import cached_property

from chebyshev import ChebyshevPolynomial

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from numpy.polynomial.chebyshev import Chebyshev

from functools import partial

from scipy.interpolate import lagrange
from scipy.linalg import norm, solve

from wand.image import Image

from itertools import product

import os


def test_roots(C, roots, critical_normalize=False, savefig=None, plot_g=False, t=1, ax=None, fig=None, normalize=True):
    """ TODO: Add another function that normalizes only at the extremal points
    """
    p = np.poly1d(roots, r=True)
    if normalize:
        if critical_normalize:
            norm_p = norm(p(C.critical_points), np.inf)
            p3b = p/norm_p
        else:
            p3b = C.normalize_polynomial(p)
    else:
        p3b = p
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
            #fig.suptitle("Locations where $|C_n| \ge |p_n|$")
            r = tuple(sorted(np.round(roots, 3)))
            rc = tuple(sorted(np.round(C.r, 3)))
            #ax.set_title("$r_p = " + f"{r}" + "$\n" + "$r_C = " + f"{rc}" + "$")
            fig.tight_layout()
            fig.savefig(savefig)



class ParametrizedPolynomial:

    def __init__(self, alpha, degree):
        self.alpha = alpha
        self.degree = degree


    @cached_property
    def p(self):
        return self.parametrized(self.alpha, self.degree)


    def parametrized_deg2(self, alpha):
        # z^2/2a - z - 1/2a = (z^2 - 2az - 1)/2a
        if -1 < alpha < 1:
            p = 2*np.poly1d([alpha, alpha], r=True)/(1+abs(alpha))**2 - 1
        else:
            p = np.poly1d([alpha, alpha], r=True)/(2*abs(alpha)) + 1 - (1 + abs(alpha))**2/(2*abs(alpha))
        return p


    def parametrized_deg3positive(self, alpha):
        critical_point = -alpha/2
        if not critical_point:
            p = -np.poly1d([2, 0, -1])
        elif -1/3 < critical_point < 0:
            h1 = np.poly1d([critical_point, critical_point, 1], r=True)/(-2*(1+critical_point)**2)
            h2 = np.poly1d([critical_point, critical_point, -1], r=True)/(2*(1- critical_point)**2)
            p = 1 - 2*h1 - 2*h2
        elif -1/2 <= critical_point <= -1/3:
            C3 = np.poly1d([4, 0, -3, 0])
            q = np.poly1d([-1], r=True)/(2*(1+critical_point)) + -1
            p = C3(q)
        else:
            raise NotImplementedError(f"The critical point must be between -1/2 and 0, got instead {critical_point=}, with {alpha=}")
        return p


    def parametrized_deg3negative(self, alpha):
        p = self.parametrized_deg3positive(-alpha)
        return -p(np.poly1d([-1, 0]))


    def parametrized_deg3(self, alpha):
        if alpha >= 0:
            p = self.parametrized_deg3positive(alpha)
        else:
            p = self.parametrized_deg3negative(alpha)
        return p

    def parametrized(self, alpha, n=2):
        return self.parametrized_deg2(alpha) if n == 2 else self.parametrized_deg3(alpha)


    def __call__(self, x):
        return self.p(x)


def invert_index(polynomials, index):
    return polynomials[index].alpha


def supindex(z, polynomials):
    values = -np.ones(z.shape) #np.zeros(z.shape) #np.abs(polynomials[0](z))
    indices = -np.ones(z.shape)
    i = 0
    for p in polynomials:
        pzv = np.abs(p(z))
        comp = pzv > values
        values[comp] = pzv[comp]
        indices[comp] = i
        if not (i % 100):
            print(f"Performed comparisons with {i+1} polynomials")
        i += 1
    return indices.astype(int), values


def label_supgrid(z, polynomials):
    indices, values = supindex(z, polynomials)
    alphas = []
    for row in indices:
        row_alphas = []
        for index in row:
            row_alphas.append(invert_index(polynomials, index))
        alphas.append(row_alphas)
    return np.array(alphas), indices, values


def gen_alpha_plots():
    c3 = np.poly1d([2, 0, -1])
    X = np.linspace(-2, 2, 1000000)
    c2 = np.poly1d([2, 0, -1])
    C2 = ChebyshevPolynomial(X=X, polynomial=c2)

    n = 101
    A = np.linspace(-1, 1, n, endpoint=True)

    polynomials = [ParametrizedPolynomial(a, 2) for a in A]
    #polynomials = [ParametrizedPolynomial(a, 3) for a in [-1, -1/2, 0, 1/2, 1]]
    xv, yv, zv = C2.calculate_grid(1, 0, 1000)

    fig, ax = plt.subplots()
    pz, indices, values = label_supgrid(zv, polynomials)

    cb = ax.contourf(xv, yv, pz, levels=len(polynomials))
    #ax.contour(xv, yv, pz, levels=[0])
    fig.colorbar(cb)
    C2.plot_disks(ax=ax)
    fig.savefig("alpha-gradients.png")


def stretch_polynomial(p, c):
    return (p*c)(np.poly1d([1/c,0]))


def gengifs():
    acc = []
    p = ParametrizedPolynomial(2, 2)
    c2 = np.poly1d([2, 0, -1])
    X = np.linspace(-1, 1, 100)
    C2 = ChebyshevPolynomial(X=X, polynomial=c2)
    v = C2.critical_points
    v = [v[0], v[1], v[2]]
    eps = 0.1
    t = 0.0
    i = 0
    xv, yv, zv = C2.grid()
    with Image() as wand:
        while t < 2*np.pi + eps:
            if not t:
                t += eps
                continue
            g = C2.polynomial + p.p*(0.5+0.5*np.e**(1j*t))
            h = C2.polynomial + p.p*(0.5+0.5*np.e**(1j*t)).conjugate()
            gh = g*h
            #h = p.p + C2.polynomial*np.e**(1j*t)
            #pp = (g*h)/(g*h).coef[0]
            #holds = C2.check_terms([pp], zv)
            #chebcoefs = Chebyshev.fit(v, g(v), deg=2).convert().coef
            #f = lambda z: [chebcoefs[0], chebcoefs[1]*z, chebcoefs[2]*C2(z)]
            #f = lambda z: [g.coef[1]*z, g.coef[0]*z**2+ g.coef[2]]
            #pc = ChebyshevPolynomial(X=X, polynomial=p.p, allow_complex=True)
            #pv = pc.critical_points
            #vv = [pv[0], pv[1], C2.Ek_midpoints[0] + C2.Ek_radii[0]*1j]  #[0.5*(-1 + np.e**(1j*t)), 0.5*(-1 + np.e**(1j*t-1j*np.pi)), 55]
            #lp = C2.lagrange_polynomials_(g*1j, vv)
            #f = lambda z: [lp[1](zv), lp[0](zv), lp[2](zv)]
            #fzv = f(zv)
            #acc = []
            #for l, r in product(fzv, fzv):
            #    acc.append(l * r.conjugate())
            #sgns = sum([np.sign(it) for it in acc])
            #abses = np.abs(sgns)
            fig, ax = plt.subplots()
            #val = abses == len(acc)
            test_roots(C2, p.p.r, ax=ax)
            #cm = ax.contourf(xv, yv, holds)
            #fig.colorbar(cm)
            #C2.plot_disks(ax=ax)
            #for node in vv:
            #    ax.plot(node.real, node.imag, "o", color="red")
            for root in gh.r:
                ax.plot(root.real, root.imag, "o", color="orange")
            for root in C2.r:
                ax.plot(root.real, root.imag, "o", color="green")
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            fig.suptitle(f"{t=:.2}")
            fig.savefig(f"{i}.png")
            with Image(filename=f"{i}.png") as im:
                wand.sequence.append(im)
            os.remove(f"{i}.png")
            t += eps
            i += 1
        wand.type = 'optimize'
        wand.save(filename="cheb-sequence-cheb-compnodes9.gif")


if __name__ == '__main__':
    gen_alpha_plots()