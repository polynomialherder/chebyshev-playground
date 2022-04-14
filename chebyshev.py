import hashlib
import random
import time
import warnings

from itertools import permutations, pairwise
from functools import partial, cached_property, lru_cache
from random import uniform

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lagrange import LagrangePolynomial
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


class ChebyshevPolynomial:

    """ Generates a random degree n Chebyshev polynomial and domain E given a discretized space X
    """

    def __init__(self, n, X, seed=1, nodes=None, known_values=None, absolute=False):
        self.n = n
        self.X = X
        self.seed = seed
        self.domain_size = X.size
        self.absolute = absolute
        # Figure and axis properties for plotting
        self.fig = None
        self.ax = None
        self._nodes = nodes
        self._known_values = known_values


    def normalize_polynomial(self, poly):
        return poly / norm(poly(self.E), np.Inf)


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


    @cached_property
    def known_values(self):
        if self._known_values is None:
            return np.array([(-1)**(self.n-k) for k in range(self.n+1)])
        return np.array(self._known_values)


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


    @property
    def left(self):
        lower_bound = min(self.critical_points)
        return self.X[self.X < lower_bound]


    @property
    def right(self):
        upper_bound = max(self.critical_points)
        return self.X[upper_bound < self.X]


    @property
    def calculate_intervals(self):
        critical_point_pairs = pairwise(self.critical_points)
        Ek, gaps = [], []
        for index, bounds in enumerate(critical_point_pairs):
            minimum, maximum = bounds
            interval = self.X[np.logical_and(minimum <= self.X, self.X <= maximum)]
            if not index % 2:
                if interval.size:
                    Ek.append(interval)
            else:
                gaps.append(interval)
        return Ek, gaps


    @property
    def E(self):
        return np.concatenate(self.Ek)


    @property
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


    @property
    def E_intervals(self):
        return self.intervals(self.Ek)


    @property
    def E_midpoint(self):
        return (self.E.min() + self.E.max())/2


    @property
    def E_disk_radius(self):
        return self.E.max() - self.E_midpoint


    @property
    def gap_intervals(self):
        return self.intervals(self.gaps)


    @property
    def gaps(self):
        _, gaps = self.calculate_intervals
        return gaps


    @lru_cache
    def Ek_nodes(self, v):
        d = []
        for idx, start_end in enumerate(self.E_intervals):
            start, end = start_end
            nodes = []
            for i, node in enumerate(v):
                if (start <= node <= end) or (np.isclose(node, start, atol=1e-5) or np.isclose(end, node, atol=1e-5)):

                    nodes.append(i)
            d.append(nodes)
        return d


    @lru_cache(maxsize=128)
    def gap_nodes(self, v):
        d = []

        aCv = abs(C(v))
        v_negative = v[aCv < -1]
        size_negative = len(v_negative)
        v_positive = v[aCv > 1]
        size_positive = len(v_positive)
        maximum = max(size_negative, size_positive)
        m = self.n + 1

        correction_term = 0 if maximum == size_negative else 1
        i = 0
        while i < m:
            pass

        return d


    @staticmethod
    def is_within_Ek(x):
        return abs(x) < 1 or np.isclose(x, 1)


    @property
    def Ek_radii(self):
        radii = []
        for left, right in self.E_intervals:
            radii.append(
                (right - left)/2
            )
        return radii


    @property
    def Ek_midpoints(self):
        midpoints = []
        for left, right in self.E_intervals:
            midpoints.append(
                (left + right)/2
            )
        return midpoints


    @property
    def gap_radii(self):
        radii = []
        for left, right in self.gap_intervals:
            radii.append(
                (right - left)/2
            )
        return radii


    @property
    def gap_midpoints(self):
        midpoints = []
        for left, right in self.gap_intervals:
            midpoints.append(
                (left + right)/2
            )
        return midpoints


    @property
    def gap_critical_values(self):
        return list(sorted(self.polynomial.deriv().r))


    def group_nodes(self, v):
        previous = None
        L = []
        l = 0
        i = 0
        negative = []
        positive = []
        while i < len(v):

            current = v[i]
            sgn_current = np.sign(self(current))

            if not i:

                if sgn_current == 1:
                    positive.append(l)
                else:
                    negative.append(l)

                L.append([i])
                l += 1
                i += 1
                continue

            if sgn_current == 1:

                if negative:

                    idx = negative.pop(0)
                    L[idx].append(i)
                    positive.append(idx)

                else:
                    positive.append(l)
                    L.append([i])
                    l += 1

            elif sgn_current == -1:

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
        self.ax.set_xlim(self.E.min(), self.E.max())
        if not self.absolute:
            self.ax.set_ylim(-2, 2)
        else:
            self.ax.set_ylim(0, 1.5)


    def save_plot(self, path):
        self.fig.savefig(path)


    def show_plot(self):
        self.fig.show()


    def generate_plots(self, comparison_polynomials=None, show=True, save_to=None, compare=False):
        self.initialize_plot()
        self.plot_Ek()
        self.plot_gaps()
        if compare or comparison_polynomials is not None:
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
