import hashlib
import math
import numbers
import pickle
import random
import time
import warnings

from dataclasses import dataclass
from iter_utils import pairwise
from itertools import permutations
from functools import partial, cached_property, lru_cache
from random import uniform

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lagrange import LagrangePolynomial
from numpy.linalg import inv
from numpy.ma import make_mask
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
        is problematic performancewise since the last 2 arguments are both evaluated for all inputs
    """
    return np.where(
        abs(x) <= 1,
        np.cos(n*np.arccos(x)),
        np.cosh(n*np.arccosh(x))
    )


def polynomial_factory(n, range_start=-10, range_end=10):
    """ Given n and an interval determined by range_start and rage_end, generates a random polynomial
        determined by roots chosen from the closed interval [range_start, range_end].
    """
    roots = [uniform(-100, 100) for i in range(n)]
    coefficients = polyfromroots(roots)
    Pp = Poly(coefficients[::-1])
    return Pp


@dataclass
class PartialFractionsDecomposition:

    eta: float
    coefficients: np.array
    q: np.poly1d
    rem: np.poly1d
    diff: str
    terms: str


class ChebyshevPolynomial:

    """ Generates a random degree n Chebyshev polynomial and domain E given a discretized space X
    """

    def __init__(self, n=None, X=None, polynomial=None, seed=1, nodes=None, known_values=None, absolute=False):
        if X is None:
            raise Exception("A discretized domain X must be provided")
        if n is None and polynomial is None:
            raise Exception("One of n or polynomial must be provided")
        self.n = n or polynomial.order
        self.X = X
        self.seed = seed
        self.domain_size = X.size
        self.absolute = absolute
        self._polynomial = polynomial
        # Figure and axis properties for plotting
        self.fig = None
        self.ax = None
        self._nodes = nodes
        self._known_values = known_values
        self.parent = None


    def normalize_polynomial(self, poly, node_normalize=False):
        if node_normalize:
            c = poly / norm(poly(self.critical_values, np.Inf))
        else:
            c = poly / norm(poly(self.E), np.Inf)
        return c


    def normalize(self, poly):
        return self.normalize_polynomial(poly)


    @cached_property
    def id(self):
        sha1 = hashlib.sha1()
        sha1.update(self.p.coef.tobytes())
        return sha1.hexdigest()[:7]


    @cached_property
    def polynomial(self):
        if self._polynomial:
            poly = self._polynomial
        else:
            poly = lagrange(self.nodes, self.known_values)
        return poly

    @cached_property
    def derivative(self):
        c = ChebyshevPolynomial(X=self.X, polynomial=self.polynomial.deriv())
        c.parent = self
        return c


    @cached_property
    def p(self):
        return Poly(self.polynomial.coef[::-1])


    @cached_property
    def nodes(self):

        min_ = self.X.min()
        max_ = self.X.max()
        midpoint = (min_ + max_)/2
        if self._nodes is None:
            nodes = np.random.triangular(min_, midpoint, max_, size=self.n+1)
        else:
            nodes = self._nodes
        return sorted(nodes)

    @staticmethod
    def calculate_grid(r, q, n):
        X = np.linspace(q - r, q + r, n)
        Y = np.linspace(-r, r, n)
        xv, yv = np.meshgrid(X, Y)
        return xv, yv, xv + 1j*yv


    @staticmethod
    def calculate_circle_points(r, q, n):
        theta = np.linspace(0, 2 * np.pi, n+1)
        X = q + r*np.cos(theta)
        Y = r*np.sin(theta)
        return X + 1j*Y



    def grid(self, n=1000):
        return self.calculate_grid(self.E_disk_radius, self.E_midpoint, n)


    def Ek_circles(self, n=1000):
        disks = []
        for r, q in zip(self.Ek_radii, self.Ek_midpoints):
            d = self.calculate_circle_points(r, q, n)
            disks.append(d)
        return disks


    def Ek_circle_points(self, n=1000):
        return np.concatenate(self.Ek_circles(n=n))


    @cached_property
    def known_values(self):
        if self._known_values is None:
            return np.array([(-1)**(self.n-k) for k in range(self.n+1)])
        return np.array(self._known_values)


    @cached_property
    def maximum_points(self):
        return np.array([r for r in (self.polynomial - 1).r if np.isclose(r.imag, 0)])


    @cached_property
    def maxima(self):
        return np.ones(self.maximum_points.size)


    @cached_property
    def minimum_points(self):
        return np.array([r for r in (self.polynomial + 1).r if np.isclose(r.imag, 0)])


    @cached_property
    def minima(self):
        return -1*np.ones(self.minimum_points.size)


    @cached_property
    def extrema(self):
        return sorted(np.concatenate((self.maxima, self.minima)))


    @cached_property
    def coef(self):
        return self.polynomial.coef


    @cached_property
    def T(self):
        return self.polynomial/self.coef[0]


    @cached_property
    def norm(self):
        return norm(self.T(self.E), np.inf)


    @staticmethod
    def partial_fractions_expansion(E, p_numerator, p_denominator):
        r = np.sort(p_denominator.r)
        q, rem = np.polydiv(p_numerator, p_denominator)
        dp = p_denominator.deriv()
        coefficients = rem(r)/dp(r)
        diff = lambda z: [1/(z - pr) for pr in r]
        eta = norm(p_numerator(E), np.inf)/norm(p_denominator(E), np.inf)
        terms = lambda z: np.array([q(z)] + list(coef*d for coef, d in zip(coefficients, diff(z))))
        return PartialFractionsDecomposition(eta, coefficients, q, rem, diff, terms)


    def partial_fractions_comparison(self, r):
        p = np.poly1d(r, r=True)
        return self.partial_fractions_expansion(self.E, self.T, p)

    def partial_fractions_comparison2(self, r):
        p = np.poly1d(r, r=True)
        return self.partial_fractions_expansion(self.E, p, self.T)


    @cached_property
    def critical_points(self):
        raw = sorted(np.concatenate((self.minimum_points, self.maximum_points)))
        extremal_points = []
        seen = set()
        for i, pt in enumerate(raw):
            duplicate = False
            for j, other_pt in enumerate(raw):
                if (j, i) in seen:
                    continue
                if i == j:
                    continue
                elif np.isclose(pt, other_pt, atol=2e-3):
                    duplicate = True
                seen.add((i, j))
            if not duplicate:
                extremal_points.append(pt)
        return extremal_points

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
            interior = self.X[np.logical_and(minimum <= self.X, self.X <= maximum)]
            interval = np.concatenate([[minimum], interior, [maximum]])
            if not interval.size:
                continue
            within_E = interval[abs(self(interval)) <= 1]
            ratio_within_E = within_E.size/interval.size
            if ratio_within_E >= 0.90:
                Ek.append(interval)
            else:
                gaps.append(interval)
        return Ek, gaps


    @cached_property
    def roots(self):
        return np.sort(self.polynomial.r)


    @cached_property
    def r(self):
        return self.roots


    @cached_property
    def deriv(self):
        return self.polynomial.deriv()


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
            if not set_.size:
                continue
            intervals.append(
                (min(set_), max(set_))
            )
        return intervals


    @cached_property
    def Ek_intervals(self):
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


    def Ek_nodes(self, v):
        d = []
        for idx, start_end in enumerate(self.Ek_intervals):
            start, end = start_end
            nodes = []
            for i, node in enumerate(v):
                if (start <= node <= end) or (np.isclose(node, start, atol=1e-5) or np.isclose(end, node, atol=1e-5)):

                    nodes.append(i)
            d.append(nodes)
        return d



    @staticmethod
    def is_within_Ek(x):
        return abs(x) < 1


    @cached_property
    def Ek_radii(self):
        radii = []
        for left, right in self.Ek_intervals:
            radii.append(
                (right - left)/2
            )
        return radii


    @cached_property
    def Ek_midpoints(self):
        midpoints = []
        for left, right in self.Ek_intervals:
            midpoints.append(
                (left + right)/2
            )
        return midpoints


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
    def gap_critical_values(self):
        return list(sorted(self.polynomial.deriv().r))


    def group_nodes(self, v):
        v = np.array(v)
        in_Ek = self.is_within_Ek(v)
        v_in_Ek = v[in_Ek]
        v_in_gap = v[~in_Ek]
        Ek_grouping = self.group_nodes_Ek(v_in_Ek)
        gap_grouping = self.group_nodes_alternating(v_in_gap)
        return [*Ek_grouping, *gap_grouping]


    def group_nodes_Ek(self, v):
        intervals = self.Ek_intervals
        groupings = []
        for left, right in intervals:
            interval_points = []
            for i, point in enumerate(v):
                if np.isclose(point, left, atol=1e-7) or np.isclose(point, right, atol=1e-7):
                    interval_points.append(i)
            if interval_points:
                groupings.append(interval_points)
        return groupings


    def group_nodes_alternating(self, v):
        # L accumulates groups
        L = []
        # Index corresponding to largest index in L
        l = 0
        # Index corresponding to current node under consideration
        i = 0
        # Lists keeping track of where we've most recently placed negative
        # or positive nodes (in C_n)
        negative = []
        positive = []
        while i < len(v):

            # The current node under consideration, and its sign
            current = v[i]
            sgn_current = np.sign(self(current))

            if not i:
                # The base case -- if this is the first node under consideration,
                # place it in its own group, and keep track of whether it was positive
                # or negative in C_n

                if sgn_current == 1:
                    positive.append(l)
                else:
                    negative.append(l)

                L.append([i])
                l += 1
                i += 1
                continue

            if sgn_current == 1:
                # If the current node is positive, then place it inside a group where the
                # last node was negative in C_n if such a group is available

                if negative:

                    idx = negative.pop(0)
                    L[idx].append(i)
                    # Now that we've placed an element positive in C_n in the group,
                    # we need to move the group index to the "positive" pile
                    positive.append(idx)

                else:
                    # If a group where the last node was negative in C_n is not available,
                    # we create a new positive group containing the node
                    positive.append(l)
                    L.append([i])
                    l += 1

            elif sgn_current == -1:
                # The logic here is effectively the same as the positive case above
                # with the positive/negative roles reversed

                if positive:

                    idx = positive.pop(0)
                    L[idx].append(i)

                    negative.append(idx)

                else:
                    negative.append(l)
                    L.append([i])
                    l += 1
            i += 1

        return L



    @property
    def comparison_polynomials(self):
        polynomials = [polynomial_factory(self.n, self.E) for i in range(5)]
        return [self.normalize_polynomial(p) for p in polynomials]


    def initialize_plot(self, size=(25, 15)):
        if self.fig:
            self.clear_plot()
        if self.parent:
            self.fig, self.ax = self.parent.fig, self.parent.ax
        else:
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


    def special_indices(self, gap_index, left_endpoint=True):
        """ gap_index refers to the left endpoint of the gap
        """
        if left_endpoint:
            starting_index = 0
        else:
            starting_index = 1
        left_index_set = [i for i in range(starting_index, gap_index, 2)]
        gap_index_set = [gap_index, gap_index + 1]
        missing = self.n + 1 - (len(left_index_set) + len(gap_index_set))
        right_index_set = [i for i in range(gap_index + 2, gap_index + 2 + 2*missing, 2)]
        return np.array(left_index_set + gap_index_set + right_index_set)

    def special_nodes(self, gap_index, left_endpoint=True):
        indices = self.special_indices(gap_index, left_endpoint)
        return np.array(self.critical_points)[indices]


    @property
    def plot_color(self):
        return "indigo" if not self.parent else "salmon"


    def plot_gaps(self, ax=None):
        ax = self.ax if self.ax else ax
        for idx, gap in enumerate(self.gaps):
            y = self(gap)
            ax.plot(gap, y, "--", color=self.plot_color, label=None)


    def plot_Ek(self, ax=None):
        ax = self.ax if self.ax else ax
        hline = 0 if self.parent is None else 0.05
        p = "" if not self.parent else "'"
        for idx, E in enumerate(self.Ek):
            curve_label = self.label(idx, f"$C_n{p} :$ {self.p._repr_latex_()}")
            domain_label = self.label(idx, f"$E_k{p}$")
            y =  self(E)
            ax.plot(E, y, "-r", label=curve_label, color=self.plot_color)
            ax.hlines(hline, E.min(), E.max(), label=domain_label, colors=self.plot_color, linewidth=10)


    def plot_comparison_polynomials(self, comparison_polynomials=None):
        if not comparison_polynomials:
            comparison_polynomials = self.comparison_polynomials
        for idx, p in enumerate(comparison_polynomials):
            y = self.handle_absolute(p(self.X))
            if "_repr_latex_" in dir(p):
                self.ax.plot(self.X, y, "-", label=f"{p._repr_latex_()}")
            else:
                self.ax.plot(self.X, y, "-", label="$"+f"p_{idx}"+"$")


    def plot_left_and_right(self, ax=None):
        ax = self.ax if self.ax else ax
        left_y = self(self.left)
        right_y = self(self.right)
        ax.plot(self.left, left_y, "--", label=None, color=self.plot_color)
        ax.plot(self.right, right_y, "--", label=None, color=self.plot_color)


    def save(self):
        with open(f"{self.id}.poly", "wb") as f:
            pickle.dump(self.polynomial, f)

    @staticmethod
    def read(path, domain_minimum=-10, domain_maximum=10):
        with open(path, "rb") as f:
            polynomial = pickle.load(f)
        X = np.linspace(domain_minimum, domain_maximum, 1000000)
        return ChebyshevPolynomial(X=X, polynomial=polynomial)


    def plot_Cn(self, ax=None):
        ax = self.ax if self.ax else ax
        self.plot_left_and_right(ax=ax)
        self.plot_Ek(ax=ax)
        self.plot_gaps(ax=ax)


    def plot_extrema(self, ax=None):
        ax = ax if ax else self.ax
        real_maxima = self.maximum_points[np.isclose(abs(self(self.maximum_points.real)), 1)]
        real_minima = self.minimum_points[np.isclose(abs(self(self.minimum_points.real)), 1)]
        ax.plot(real_maxima, self(real_maxima), "o", color="red")
        minima = self.handle_absolute(self.minima)
        ax.plot(real_minima, self(real_minima), "o", color="red")


    def plot_guidelines(self):
        if not self.absolute:
            self.ax.hlines(-1, self.X.min(), self.X.max(), colors='grey')
        self.ax.hlines(1, self.X.min(), self.X.max(), colors='grey')


    def plot_classical(self):
        self.ax.plot(self.X, T(self.n, self.X), "-", label='$T_n$')


    def plot_critical_values(self, ax=None):
        ax = ax if ax else self.ax
        ax.plot(
            self.gap_critical_values,
            np.zeros(len(self.gap_critical_values)), "*",
            color="pink", label="Maxima of $C_n$"
        )


    def plot_roots(self, ax=None):
        ax = ax if ax else self.ax
        ax.plot(
            self.polynomial.r, np.zeros(len(self.polynomial.r)),
            "*", label="Roots of $C_n$"
        )


    @staticmethod
    def lagrange_polynomials_(p, Xp):
        V = np.array([[vi**i for i in range(len(Xp))] for vi in Xp])
        coefficients = (inv(V)*p(Xp)).transpose()
        return [np.poly1d(coef[::-1]) for coef in coefficients]


    @staticmethod
    def vandermonde(Xp):
        return np.array([[vi**i for i in range(len(Xp))] for vi in Xp])


    def lagrange_polynomials(self, Xp):
        return self.lagrange_polynomials_(self, Xp)


    @property
    def gap_disk_color(self):
        return "salmon" if not self.parent else "violet"

    @property
    def Ek_disk_color(self):
        return "green" if not self.parent else "darkseagreen"


    def plot_disks(self, ax=None, E_disk_color=None, gap_disk_color=None):
        if E_disk_color is None:
            E_disk_color = self.Ek_disk_color
        if gap_disk_color is None:
            gap_disk_color = self.gap_disk_color

        ax = ax if ax else self.ax
        for idx, disk_info in enumerate(zip(self.Ek_midpoints, self.Ek_radii)):
            midpoint, radius = disk_info
            disk = plt.Circle((midpoint, 0), radius, fill=False, color=E_disk_color, linestyle="--")
            ax.add_patch(disk)
        for idx, disk_info in enumerate(zip(self.gap_midpoints, self.gap_radii)):
            midpoint, radius = disk_info
            disk = plt.Circle((midpoint, 0), radius, fill=False, color=gap_disk_color, linestyle="--")
            ax.add_patch(disk)

        E_disk = plt.Circle((self.E_midpoint, 0), self.E_disk_radius, fill=False, linestyle="-")
        ax.add_patch(E_disk)


    def generate_diskplot(self, derivative=False):
        self.initialize_plot()
        if derivative:
            self.derivative.plot_features(derivative=False)
        self.plot_critical_values()
        self.plot_roots()
        self.plot_disks()
        self.fig.show()


    def configure_legend(self):
        self.ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        self.fig.subplots_adjust(right=0.6)


    def configure_plot(self, derivative=False):
        self.ax.grid()
        Emin = self.E.min()
        Emax = self.E.max()
        if derivative:
            xlim_left = min(Emin, self.derivative.E.min())
            xlim_right = max(Emax, self.derivative.E.max())
        else:
            xlim_left = Emin
            xlim_right = Emax
        self.ax.set_xlim(xlim_left, xlim_right)
        if not self.absolute:
            self.ax.set_ylim(-2, 2)
        else:
            self.ax.set_ylim(0, 1.5)


    def save_plot(self, path):
        self.fig.savefig(path)


    def show_plot(self):
        self.fig.show()


    def generate_plots(self, comparison_polynomials=None, show=True,
                       save_to=None, compare=False, derivative=False
        ):
        self.initialize_plot()
        self.plot_Ek()
        self.plot_gaps()
        if compare or comparison_polynomials is not None:
            self.plot_comparison_polynomials(
                comparison_polynomials
            )
        if derivative:
            d = self.derivative
            d.initialize_plot()
            d.plot_Ek()
            d.plot_gaps()
            d.plot_left_and_right()
        self.plot_left_and_right()
        self.plot_extrema()
        self.plot_guidelines()
        self.configure_legend()
        self.configure_plot(derivative=derivative)
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


    def _repr_latex_(self):
        # get the scaled argument string to the basis functions
        off, scale = self.mapparms()
        if off == 0 and scale == 1:
            term = 'x'
            needs_parens = False
        elif scale == 1:
            term = f"{self._repr_latex_scalar(off)} + x"
            needs_parens = True
        elif off == 0:
            term = f"{self._repr_latex_scalar(scale)}x"
            needs_parens = True
        else:
            term = (
                f"{self._repr_latex_scalar(off)} + "
                f"{self._repr_latex_scalar(scale)}x"
            )
            needs_parens = True

        mute = r"".format

        parts = []
        for i, c in enumerate(self.coef):
            # prevent duplication of + and - signs
            if i == 0:
                coef_str = f"{self._repr_latex_scalar(c)}"
            elif not isinstance(c, numbers.Real):
                coef_str = f" + ({self._repr_latex_scalar(c)})"
            elif not np.signbit(c):
                coef_str = f" + {self._repr_latex_scalar(c)}"
            else:
                coef_str = f" - {self._repr_latex_scalar(-c)}"

            # produce the string for the term
            term_str = self._repr_latex_term(i, term, needs_parens)
            if term_str == '1':
                part = coef_str
            else:
                part = rf"{coef_str}\,{term_str}"

            if c == 0:
                part = mute(part)

            parts.append(part)

        if parts:
            body = ''.join(parts)
        else:
            # in case somehow there are no coefficients at all
            body = '0'

        return rf"$x \mapsto {body}$"



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
