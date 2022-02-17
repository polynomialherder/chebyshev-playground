import hashlib
import random
import time
import warnings

from itertools import permutations, pairwise
from functools import partial, cached_property
from random import uniform

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from numpy.polynomial.polynomial import Polynomial, polyfromroots
from scipy.linalg import norm
from scipy.interpolate import lagrange

warnings.filterwarnings('ignore')

matplotlib.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r'\usepackage{amsmath}'
})


def T(n, x):
    """ Classical Chebyshev polynomial implementation. Note that this implementation
        is problematic performancewise since thet last 2 arguments are both evaluated for all x
    """
    return np.where(
        abs(x) <= 1,
        np.cos(n*np.arccos(x)),
        np.cosh(n*np.arccosh(x))
    )


def polynomial_factory(n, E):
    """ Given n and a discretized domain E, generates a random polynomial determined by
        roots chosen between max(E) and min(E), inclusive. The returned polynomial is
        normalized to have uniform norm of unity
    """
    roots = [uniform(E.min(), E.max()) for i in range(n)]
    coefficients = polyfromroots(roots)
    Pp = Poly(coefficients)
    return Pp / norm(Pp(E), np.Inf)


class ChebyshevPolynomial:

    """ Generates a random degree n Chebyshev polynomial and domain E given a discretized space X
    """

    def __init__(self, n, X, seed=1, absolute=False):
        self.n = n
        self.X = X
        self.seed = seed
        self.domain_size = X.size
        self.absolute = absolute
        # Figure and axis properties for plotting
        self.fig = None
        self.ax = None


    @cached_property
    def id(self):
        sha1 = hashlib.sha1()
        sha1.update(self.p.coef.tobytes())
        return sha1.hexdigest()[:7]


    @cached_property
    def polynomial(self):
        return lagrange(self.nodes, self.known_values)


    @cached_property
    def p(self):
        return Poly(self.polynomial)


    @cached_property
    def nodes(self):
        nodes = np.array(
            random.choices(self.X, k=self.n+1)
        )
        return sorted(nodes)


    @cached_property
    def known_values(self):
        return np.array([(-1)**(self.n-k) for k in range(self.n+1)])


    @cached_property
    def maximum_points(self):
        return (self.polynomial - 1).r


    @cached_property
    def maxima(self):
        return np.ones(self.maximum_points.size)


    @cached_property
    def minimum_points(self):
        return (self.polynomial + 1).r


    @cached_property
    def minima(self):
        return -1*np.ones(self.minimum_points.size)


    @cached_property
    def extrema(self):
        return sorted(np.concatenate((self.maxima, self.minima)))


    @cached_property
    def critical_points(self):
        return sorted(np.concatenate((self.minimum_points, self.maximum_points)))


    @cached_property
    def left(self):
        lower_bound = min(self.critical_points)
        return self.X[self.X < lower_bound]


    @cached_property
    def right(self):
        upper_bound = max(self.critical_points)
        return self.X[upper_bound < self.X]


    @cached_property
    def calculate_intervals(self):
        critical_point_pairs = pairwise(self.critical_points)
        Ek, gaps = [], []
        for index, bounds in enumerate(critical_point_pairs):
            minimum, maximum = bounds
            interval = self.X[np.logical_and(minimum <= self.X, self.X <= maximum)]
            if not index % 2:
                Ek.append(interval)
            else:
                gaps.append(interval)
        return Ek, gaps


    @cached_property
    def E(self):
        return np.concatenate(self.Ek)


    @cached_property
    def Ek(self):
        Ek, _ = self.calculate_intervals
        return Ek


    @staticmethod
    def intervals(sets):
        intervals = []
        for set_ in sets:
            intervals.append(
                (min(set_), max(set_))
            )
        return intervals


    @cached_property
    def E_intervals(self):
        return self.intervals(self.Ek)


    @cached_property
    def E_midpoint(self):
        return (self.E.min() + self.E.max())/2


    @cached_property
    def E_disk_radius(self):
        return self.E.max() - self.E_midpoint


    @cached_property
    def gap_intervals(self):
        return self.intervals(self.gaps)


    @cached_property
    def gaps(self):
        _, gaps = self.calculate_intervals
        return gaps


    @cached_property
    def gap_radii(self):
        radii = []
        for left, right in self.gap_intervals:
            radii.append(
                (right - left)/2
            )
        return radii


    @cached_property
    def gap_midpoints(self):
        midpoints = []
        for left, right in self.gap_intervals:
            midpoints.append(
                (left + right)/2
            )
        return midpoints


    @cached_property
    def comparison_polynomials(self):
        return [polynomial_factory(self.n, self.E) for i in range(5)]


    def initialize_plot(self, size=(25, 15)):
        if self.fig:
            self.clear_plot()
        self.fig, self.ax = plt.subplots(figsize=size)


    def clear_plot(self):
        self.fig.clear()
        self.ax.clear()
        self.fig = self.ax = None


    @staticmethod
    def label(iteration, label):
        if not iteration:
            return label
        return None


    def handle_absolute(self, data):
        if self.absolute:
            return np.absolute(data)
        return data


    def plot_gaps(self):
        for idx, gap in enumerate(self.gaps):
            y = self(gap)
            self.ax.plot(gap, y, "--", color="indigo", label=None)


    def plot_Ek(self):
        for idx, E in enumerate(self.Ek):
            curve_label = self.label(idx, f"$C_n :$ {self.p._repr_latex_()}")
            domain_label = self.label(idx, "$E_k$")
            y =  self(E)
            self.ax.plot(E, y, "-r", label=curve_label, color="indigo")
            self.ax.hlines(0, E.min(), E.max(), label=domain_label, colors="indigo", linewidth=10)


    def plot_comparison_polynomials(self, comparison_polynomials=None):
        if not comparison_polynomials:
            comparison_polynomials = self.comparison_polynomials
        for idx, p in enumerate(comparison_polynomials):
            y = self.handle_absolute(p(self.X))
            if "_repr_latex_" in dir(p):
                self.ax.plot(self.X, y, "-", label=f"{p._repr_latex_()}")
            else:
                self.ax.plot(self.X, y, "-", label="$"+f"p_{idx}"+"$")


    def plot_left_and_right(self):
        left_y = self(self.left)
        right_y = self(self.right)
        self.ax.plot(self.left, left_y, "--", label=None, color="indigo")
        self.ax.plot(self.right, right_y, "--", label=None, color="indigo")


    def plot_extrema(self):
        self.ax.plot(self.maximum_points, self.maxima, "o", color="red")
        minima = self.handle_absolute(self.minima)
        self.ax.plot(self.minimum_points, minima, "o", color="red")


    def plot_guidelines(self):
        if not self.absolute:
            self.ax.hlines(-1, self.X.min(), self.X.max(), colors='grey')
        self.ax.hlines(1, self.X.min(), self.X.max(), colors='grey')


    def plot_classical(self):
        self.ax.plot(self.X, T(self.n, self.X), "-", label='$T_n$')


    def configure_legend(self):
        self.ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        self.fig.subplots_adjust(right=0.6)


    def configure_plot(self):
        self.ax.grid()
        self.ax.set_xlim(self.X.min(), self.X.max())
        if not self.absolute:
            self.ax.set_ylim(-2, 2)
        else:
            self.ax.set_ylim(0, 1.5)


    def save_plot(self, path):
        self.fig.savefig(path)


    def show_plot(self):
        self.fig.show()


    def generate_plots(self, comparison_polynomials=None, show=True, save_to=None):
        self.initialize_plot()
        self.plot_Ek()
        self.plot_gaps()
        self.plot_comparison_polynomials(
            comparison_polynomials
        )
        self.plot_left_and_right()
        self.plot_extrema()
        self.plot_guidelines()
        self.configure_legend()
        self.configure_plot()
        if show:
            self.show_plot()
        if save_to:
            self.save_plot(save_to)


    def __call__(self, x):
        Y = self.polynomial(x)
        return Y if not self.absolute else np.absolute(Y)


    def __repr__(self):
        return f"{self.polynomial}"


class Poly(Polynomial):
    """ A subclass used to override NumPy polynomial string representations for
        less cluttered plots
    """

    @staticmethod
    def _repr_latex_scalar(x):
        return r'\text{{{}}}'.format(round(x, 3))


    def _generate_string(self, term_method):
        """
        Generate the full string representation of the polynomial, using
        ``term_method`` to generate each polynomial term.
        """
        # Get configuration for line breaks
        linewidth = np.get_printoptions().get('linewidth', 75)
        if linewidth < 1:
            linewidth = 1
        out = f"{round(self.coef[0], 3)}"
        for i, coef in enumerate(self.coef[1:]):
            out += " "
            power = str(i + 1)
            # Polynomial coefficient
            # The coefficient array can be an object array with elements that
            # will raise a TypeError with >= 0 (e.g. strings or Python
            # complex). In this case, represent the coefficient as-is.
            try:
                if coef >= 0:
                    next_term = f"+ {round(coef, 3)}"
                else:
                    next_term = f"- {round(-coef, 3)}"
            except TypeError:
                next_term = f"+ {coef}"
            # Polynomial term
            next_term += term_method(power, "x")
            # Length of the current line with next term added
            line_len = len(out.split('\n')[-1]) + len(next_term)
            # If not the last term in the polynomial, it will be two
            # characters longer due to the +/- with the next term
            if i < len(self.coef[1:]) - 1:
                line_len += 2
            # Handle linebreaking
            if line_len >= linewidth:
                next_term = next_term.replace(" ", "\n", 1)
            out += next_term
        return out


# Sandbox code

def contour_plot(p):
    X = np.linspace(-1, 1, 1000)
    Y = np.linspace(-2, 2, 1000)
    xv, yv = np.meshgrid(X, Y)
    p.absolute = True
    Z = xv + 1j*yv
    zv = p(Z)
    D = np.array([zv - abs(f(Z)) for f in p.comparison_polynomials])
    mins = np.amin(D, 0)
    midpoint = (p.E.min() + p.E.max())/2
    radius = p.E.max() - midpoint
    plt.contourf(xv, yv, mins)
    plt.colorbar()
    #ax.clabel(CS, inline=True, fontsize=10)
    plt.savefig("contour.png")


def surface_plot(p):
    X = np.linspace(-1, 1, 10000)
    Y = np.linspace(-2, 2, 10000)
    xv, yv = np.meshgrid(X, Y)
    Z = xv + 1j*yv


def check_lemma(z, xi, xj):
    return ((z.conjugate() - xj.conjugate()) * (z - xi)).real


def c(x, i):
    c = 1
    xi = x[i]
    for k, xk in enumerate(x):
        if k == i:
            continue
        c *= xi - xk
    return c

def check_lemma2(n, z, nodes, i, j):
    rp = check_lemma(z, nodes[i], nodes[j]) if i != j else 1
    prod = 1
    str_ = f"({i}, {j})\n"
    for k, node in enumerate(nodes):
        if k in (i, j):
            continue
        prod *= abs(z - nodes[k])**2
        str_ += f"|z-x_{k}|^2"
    denom = (-1)**(n-i) * (-1)**(n-j) * c(nodes, i) * c(nodes, j)
    print(str_ + f"* {rp} / ((-1)^{n-1} * (-1)^{n-j} * {c(nodes, i)} * {c(nodes, j)})")
    return prod*rp/denom


def compare_p_with_f(p, f):
    X = np.linspace(p.Ek[0].min(), p.Ek[0].max(), 1000)
    Y = np.linspace(-1, 1, 1000)
    xv,  yv = np.meshgrid(X,  Y)
    Z = xv + 1j*yv

    f = p.comparison_polynomials[0]
    abs_p = np.abs(p(Z))
    D =  abs_p - np.abs(f(Z))


    fig, ax = plt.subplots()
    X = np.linspace(p.Ek[0].min(), p.Ek[1].max(), 1000)
    Y = np.linspace(-1, 1, 1000)
    xv,  yv = np.meshgrid(X,  Y)
    Z = xv + 1j*yv
    abs_p = np.absolute(p(Z))
    abs_f = np.absolute(f(Z))
    D = abs_p - abs_f

    cs = ax.contour(xv, yv, abs_p, levels=[0, 1])
    cb = ax.contourf(xv, yv, np.sign(D), levels=[-1, -0.5, 0, 0.5, 1])
    ax.clabel(cs, inline=True, fontsize=10)

    for E in p.Ek:
        ax.hlines(-0.5, E.min(), E.max(), colors="indigo", linewidth=10)

    gap_x1 = p.E_intervals[0][-1]
    gap_x2 = p.E_intervals[1][0]
    midpoint = (gap_x1 + gap_x2)/2
    radius = (gap_x2 - gap_x1)/2
    gap_disk = plt.Circle((midpoint, 0), radius, fill=False, linestyle="--")
    ax.add_patch(gap_disk)
    fig.colorbar(cb)
    ax.grid()
    fig.show()
    #fig.savefig("20220212-2.png")

    fig, ax = plt.subplots()

    ax.plot(p.left, p(p.left), "--", color="indigo")
    ax.plot(p.left, f(p.left), color="orange")

    ax.plot(p.right, p(p.right), "--", color="indigo")
    ax.plot(p.right, f(p.right), color="orange")

    for idx, E in enumerate(p.Ek):
        if not idx:
            ax.plot(E, p(E), "-", color="indigo", label="C_n")
            ax.plot(E, f(E), "-", color="orange", label="p_n")
            ax.hlines(0, E.min(), E.max(), color="indigo", linewidth=10, label="E_k")
        else:
            ax.plot(E, p(E), color="indigo")
            ax.plot(E, f(E), "-", color="orange")
            ax.hlines(0, E.min(), E.max(), color="indigo", linewidth=10)
    for idx, G in enumerate(p.gaps):
        if not idx:
            ax.plot(G, p(G), "--", color="indigo")
            ax.plot(G, f(G), color="orange")
        else:
            ax.plot(G, p(G), "--", color="indigo")
            ax.plot(G, f(G), "-", color="orange")
    ax.hlines(1, X.min(), X.max(), color="grey")
    ax.hlines(-1, X.min(), X.max(), color="grey")
    ax.set_xlim(X.min(), X.max())
    ax.grid()
    fig.legend()
    fig.show()
    #fig.savefig("20220212-3.png")


def plot_gap(p):
    X = np.linspace(-1, 1, 1000)
    Y = np.linspace(-1, 1, 1000)
    xv, yv = np.meshgrid(X, Y)
    z = xv + 1j*yv
    p = ChebyshevPolynomial(2, X)
    fig, ax = plt.subplots()
    gap_start, gap_end = (p.E_intervals[0][1], p.E_intervals[-1][0])
    midpoint = (gap_start + gap_end)/2
    radius = (gap_end - gap_start)/2
    disk = plt.Circle((midpoint, 0), radius, fill=False, linestyle="-")
    #c = ax.contourf(xv, yv, np.absolute(p(z)))
    ax.plot(X, p(X))
    ax.add_patch(disk)
    ax.grid()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    for idx, E in enumerate(p.Ek):
        ax.hlines(0, E.min(), E.max(), colors="indigo", linewidth=10)

    #fig.colorbar(c)
    fig.show()


def experiment_erdos():
    X = np.linspace(-1, 1, 1000)
    Y = np.linspace(-1, 1, 1000)
    xv, yv = np.meshgrid(X, Y)
    z = xv + 1j*yv

    p2 = ChebyshevPolynomial(3, X)

    # Nodes for the second order Chebyshev polynomial
    nodes = list(p.critical_points)
    nodes.sort()
    nodes = [nodes[0], nodes[1], nodes[3]]

    # Nodes for the third order Chebyshev polynomial
    nodes2 = list(p2.critical_points)
    nodes2.sort()
    nodes2 = [x for i, x in enumerate(nodes2) if np.isclose(p2(x), (-1)**(i+1))]

    # Build index sets
    from itertools import product
    indices = list(product([0, 1, 2], [0, 1, 2]))
    indices2 = list(product([0, 1, 2, 3], [0, 1, 2, 3]))

    z = min(p.E_intervals[0][0], p2.E_intervals[0][0]) - 0.15

    # This sanity checks our choice of nodes
    # The sum of the next two line should evaluate to the same as the following line within some tolerance
    # likewise for the two lines that follow
    li = [abs((l(z, nodes, i).conjugate() * l(z, nodes, j)).real) for i, j in indices]
    pp = abs(p(z)) ** 2
    li2 = [abs((l(z, nodes2, i).conjugate() * l(z, nodes2, j)).real) for i, j in indices2]
    pp2 = abs(p2(z))**2

    L = [(-1)**i * l(z, nodes, i) for i in range(3)]
    check = abs(p(z))

    L2 = [(-1)**i * l(z, nodes2, i) for i in range(4)]
    check2 = abs(p2(z))

    df = pd.DataFrame({
        "$n$": sum([[n for _ in range(n+1)] for n in (2, 3)], []),
        "$j$": sum([[i for i in range(n + 1)] for n in (2, 3)], [])
    })
    df["$x$"] = nodes + nodes2
    nodes_ = np.array(nodes + nodes2)
    df["$\\gamma$"] = sum([[np.prod(z - nodelist, where=np.arange(len(nodelist)) != i) for i, x in enumerate(nodelist)] for nodelist in (nodes, nodes2)], [])
    df["$\\eta$"] = sum([[1/np.prod(x - nodelist, where=np.arange(len(nodelist)) != i) for i, x in enumerate(nodelist)] for nodelist in (nodes, nodes2)], [])
    df["$(-1)^{j}$"] = (-1)**df["$j$"]
    df["$\\sgn$"] = np.sign(df["$(-1)^{j}$"]*df["$\\eta$"]*df["$\\gamma$"])
    return df



def l(x, nodes, j):
    prod = 1
    for i, xi in enumerate(nodes):
        if i == j:
            continue
        prod *= (x-xi)/(nodes[j] - xi)
    return prod


def S1(P, n, z):
    return np.cosh(np.log(np.abs(P(z) + np.sqrt(P(z) ** 2 - 1+0j))))


def S2(P, n, z):
    Pz = P(z)
    sqrt = np.sqrt(Pz**2 - 1+0j)
    p1 = np.abs(Pz + sqrt)**(n/P.n)
    p2 = np.abs(Pz + sqrt)**(-n/P.n)
    return 0.5*(p1 + p2)



if __name__ == '__main__':
    X = np.linspace(-1, 1, 10000)
    Y = np.linspace(-1, 1, 10000)
    xv, yv = np.meshgrid(X, Y)
    z = xv + 1j*yv

    # Construct our polynomials
    p = ChebyshevPolynomial(3, X)
    p.absolute = True

    s = partial(S2, p, p.n)

    p.generate_plots(show=False, comparison_polynomials=[s], save_to=f"{p.id}.png")

    error = np.abs(p(z)) - s(z)
    # Whenever the absolute error is within epsilon of 0,
    # set it to zero
    epsilon = 1e-8
    error[np.abs(error) < epsilon] = 0.0

    fig, ax = plt.subplots()

    c = ax.contourf(xv, yv, np.sign(error), levels=[-1, -0.5, 0, 0.5, 1])
    fig.colorbar(c)

    E_disk = plt.Circle((p.E_midpoint, 0), p.E_disk_radius, fill=False, linestyle="-")

    ax.add_patch(E_disk)

    for midpoint, radius in zip(p.gap_midpoints, p.gap_radii):
        gap_disk = plt.Circle((midpoint, 0), radius, fill=False, linestyle="--")
        ax.add_patch(gap_disk)

    ax.grid()
    fig.savefig(f"{p.id}-contour-vs-s.png")
