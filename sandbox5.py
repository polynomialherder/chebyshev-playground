import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from chebyshev import ChebyshevPolynomial
from lagrange import LagrangePolynomial
from numpy.ma import make_mask
from sandbox2 import plot_disks, Experiment

from scipy.interpolate import lagrange

import cplot


def plot_polynomial(ax, C):
    plot_disks(ax, C)
    # Compare derivatives of Chebyshev with derivatives of arbitrary normalized polynomials
    ax.plot(C.gap_critical_values, np.zeros(len(C.gap_critical_values)), "*", color="pink", label="Maxima of $C_n$")
    ax.plot(C.polynomial.r, np.zeros(len(C.polynomial.r)), "*", label="Roots of $C_n$")

if __name__ == '__main__':

    X_ = np.linspace(-5, 5, 1000000)
    p = np.poly1d([-2.35, -0.95, 1], r=True)
    p = p/p.coef[0]
    #p = np.poly1d([ 5.16946816e+02, -3.95728706e+02,  3.63319434e+01,  1.91041047e-01])
    p = np.poly1d([-2.84692545, -1.04700419,  0.23443957], r=True)
    p = p/p.coef[0]
    n = 3
    #Cn = ChebyshevPolynomial(X=X_, polynomial=p)
    Cn = ChebyshevPolynomial(X=X_, n=n)

    # This choice of nodes does not work for certain polynomials with large gaps on
    # one side, e.g. poly1d([ 5.16946816e+02, -3.95728706e+02,  3.63319434e+01,  1.91041047e-01])
    # or  poly1d([ 75.30292962, -47.82979006,  -1.98655078,   0.98519695])
    v = [
        Cn.critical_points[1], Cn.critical_points[2],
        Cn.critical_points[3], Cn.critical_points[-1]
    ]
    def p(i):
        def ps(z):
            val = 0
            for node in nodes[i]:
                val += Cn(v[node])*l.l_piecemeal(node, z)
            return val
        return ps


    circle_radius = Cn.E_disk_radius
    circle_midpoint = Cn.E_midpoint
    circular_inversion = lambda r, q, z: (q*z.conjugate() + (r**2 - abs(q)**2))/(z.conjugate() - q.conjugate())
    f = lambda z: circular_inversion(circle_radius, circle_midpoint, abs(Cn(z))**2)

    r = 1.5*Cn.E_disk_radius
    x = Cn.E_midpoint
    xlim_left, xlim_right = (x - r, x + r)
    ylim_below, ylim_above = (-r, r)

    X = np.linspace(xlim_left, xlim_right, 1000)
    Y = np.linspace(ylim_below, ylim_above, 1000)
    xv, yv = np.meshgrid(X, Y)
    zv = xv + 1j*yv


    fig, ax = plt.subplots()
    g = lambda z: circular_inversion(circle_radius, circle_midpoint, abs(Cn(z))**2)
    inverted = f(zv)
    psm = ax.contourf(xv, yv, g(zv))
    fig.colorbar(psm, ax=ax)
    ax.plot()
    Cn.plot_disks(ax=ax)
    Cn.plot_critical_values(ax=ax)
    Cn.plot_roots(ax=ax)
    ax.set_xlim(xlim_left, xlim_right)
    ax.set_ylim(ylim_below, ylim_above)
    fig.savefig("inversion.png")

    fig, ax = plt.subplots()
    psm = ax.contourf(xv, yv, circular_inversion(circle_radius, circle_midpoint, g(zv)))
    fig.colorbar(psm, ax=ax)
    ax.plot()
    Cn.plot_disks(ax=ax)
    Cn.plot_critical_values(ax=ax)
    Cn.plot_roots(ax=ax)
    ax.set_xlim(xlim_left, xlim_right)
    ax.set_ylim(ylim_below, ylim_above)
    fig.savefig("inversion-twice.png")

    C2 = ChebyshevPolynomial(n=n, X=X_)
    p2 = Cn.normalize_polynomial(C2.polynomial)

    def invert_point(C, z, p=None):
        fig, ax = plt.subplots()
        inverted = circular_inversion(C.Ek_radii[0], C.Ek_midpoints[0], abs(C(z)))
        ax.plot(z.real, z.imag, "o", label="z")
        ax.plot(abs(C(z)).real, abs(C(z)).imag, "o", label="C(z)")
        ax.plot(inverted.real, inverted.imag, "o", label="I(C(z))")
        if p:
            ax.plot(abs(p(z)).real, abs(p(z)).imag, "o", label="p(z)")
            inverted2 = circular_inversion(C.E_disk_radius, C.E_midpoint, abs(p(z)))
            ax.plot(inverted2.real, inverted2.imag, "o", label="I(p(z))")
        Cn.plot_disks(ax=ax)
        Cn.plot_critical_values(ax=ax)
        Cn.plot_roots(ax=ax)
        #ax.set_xlim(xlim_left, xlim_right)
        #ax.set_ylim(ylim_below, ylim_above)
        fig.legend()
        fig.show()

    def plot_inversion(C, z, inversion, comparison=False, name=None):
        fig, ax = plt.subplots()
        ax.plot(z.real, z.imag, "o", label="z")
        if not comparison:
            label = "$\\eta = \\lvert C(z) \\rvert$"
        else:
            label = "$\\eta = \\lvert p(z) \\rvert$"

        if not comparison:
            title = "Inversion of $\\lvert C_n(z) \\rvert$"
            ax.plot((abs(C(z))**2).real, (abs(C(z))**2).imag, "o", label=label)
        else:
            title = "Inversion of $\\lvert p(z) \\rvert$"
            ax.plot(abs(p2(z)).real, abs(p2(z)).imag, "o", label=label)
        ax.set_title(title)
        ax.plot(inversion.real, inversion.imag, "o", label="$I(\\eta)$")
        Cn.plot_disks(ax=ax)
        Cn.plot_roots(ax=ax)
        Cn.plot_critical_values(ax=ax)
        ax.plot(Cn.E_midpoint, 0, "x", color="red")
        ax.set_xlim(xlim_left, xlim_right)
        ax.set_ylim(ylim_below, ylim_above)
        fig.legend()
        if not name:
            fig.show()
        else:
            fig.savefig(name)

    C = Cn
    outside_gap = make_mask(zv)
    for midpoint, radius in zip(C.gap_midpoints, C.gap_radii):
        distance = abs(midpoint - zv)
        outside_gap = outside_gap & (distance > radius)

    outside_Ek = make_mask(zv)
    for midpoint, radius in zip(C.Ek_midpoints, C.Ek_radii):
        distance = abs(midpoint - zv)
        outside_Ek = outside_Ek & (distance > radius)

    inside_E_disk = abs(C.E_midpoint - zv) < C.E_disk_radius
    region = inside_E_disk & outside_Ek & outside_gap

    zv_in_region = np.random.choice(zv[region])
    r = Cn.E_disk_radius
    q = Cn.E_midpoint
    f = lambda z: circular_inversion(r, q, abs(Cn(z))**2)
    g = lambda z: circular_inversion(r, q, abs(p2(z))**2)
    plot_inversion(C, zv_in_region, f(zv_in_region), comparison=False, name="inversion-of-cz.png")
    plot_inversion(p2, zv_in_region, g(zv_in_region), comparison=True, name="inversion-of-pz.png")

    def plot_inversion_regions(inverted, plot_title, filename):
        fig, ax = plt.subplots()
        cm = ax.contourf(xv, yv, np.abs(inverted - q) > r)
        fig.colorbar(cm)
        Cn.plot_disks(ax=ax)
        ax.plot(Cn.E_midpoint, 0, "x", color="red")
        ax.set_title(plot_title)
        ax.set_xlim(Cn.E.min(), Cn.E.max())
        ax.set_ylim(-Cn.E_disk_radius, Cn.E_disk_radius)
        fig.savefig(filename)

    plot_inversion_regions(abs(Cn(zv))**2, f"Region where $\\lvert C_n(z) \\lvert - q > r$", f"region-where-ICnz-mapped-to-disk.png")
    plot_inversion_regions(abs(p2(zv))**2, f"Region where $\\lvert p(z) \\lvert - q > r$", f"region-where-Ipnz-mapped-to-disk.png")

    plot_inversion(Cn, zv[region], f(zv[region]), name="inversion-region-of-interest.png")
    plot_inversion(Cn, zv[region], g(zv[region]), name="inversion-region-of-interest-g.png", comparison=True)
