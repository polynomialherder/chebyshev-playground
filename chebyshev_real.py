import random
import warnings

from itertools import permutations, pairwise
from functools import partial, cached_property
from random import uniform

import matplotlib.pyplot as plt
import numpy as np

from numpy.polynomial.polynomial import Polynomial, polyfromroots
from scipy.linalg import norm
from scipy.interpolate import lagrange

warnings.filterwarnings('ignore')

class Poly(Polynomial):

    @staticmethod
    def _repr_latex_scalar(x):
        """ We're overriding these methods from numpy so that we can control
            the precision representation in visualizations
        """
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
    roots = [uniform(min(E), max(E)) for i in range(n)]
    coefficients = polyfromroots(roots)
    Pp = Poly(coefficients)
    return Pp / norm(Pp(E), np.Inf)

class ChebyshevPolynomial:

    def __init__(self, n, X, seed=1):
        self.n = n
        self.X = X
        self.seed = seed
        self.domain_size = len(X)

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

    @cached_property
    def gaps(self):
        _, gaps = self.calculate_intervals
        return gaps

    @cached_property
    def polynomial(self):
        return lagrange(self.nodes, self.known_values)

    @cached_property
    def p(self):
        return self.polynomial

    @cached_property
    def comparison_polynomials(self):
        return [polynomial_factory(self.n, self.E) for i in range(10)]

    def generate_plot(self, show_classical=True, comparison_polynomials=True, absolute=False):
        """ TODO: Decompose this into smaller methods
        """
        fig, ax = plt.subplots(figsize=(20, 16))
        for idx, gap in enumerate(self.gaps):
            if not idx:
                label = "Gap"
            else:
                label = None

            if not absolute:
                ax.plot(gap, self(gap), "--", color="indigo", label=label)
            else:
                ax.plot(gap, np.absolute(self(gap)), "--", color="indigo", label=label)

        for idx, E in enumerate(self.Ek):
            if not idx:
                curve_label = "$C_n$"
                domain_label = "$E_n$"
            else:
                curve_label = domain_label = None
            if not absolute:
                ax.plot(E, self(E), "-r", label=curve_label, color="indigo")
            else:
                ax.plot(E, np.absolute(self(E)), "-r", label=curve_label, color="indigo")
            ax.hlines(0, min(E), max(E), label=domain_label, colors="indigo", linewidth=10)

        if comparison_polynomials:
            for p in self.comparison_polynomials:
                if not absolute:
                    ax.plot(self.X, p(self.X), "-", label=f"{p}")
                else:
                    ax.plot(self.X, np.absolute(p(self.X)), "-", label=f"{p}")

        if not absolute:
            ax.plot(self.left, self(self.left), "--", label=None, color="indigo")
            ax.plot(self.right, self(self.right), "--", label=None, color="indigo")
            ax.plot(self.maximum_points, self.maxima, "o", color="red")
            ax.plot(self.minimum_points, self.minima, "o", color="red")
            ax.hlines(-1, min(self.X), max(self.X), colors='grey')

        else:
            ax.plot(self.left, np.absolute(self(self.left)), "--", label=None, color="indigo")
            ax.plot(self.right, np.absolute(self(self.right)), "--", label=None, color="indigo")
            ax.plot(self.maximum_points, self.maxima, "o", color="red")
            ax.plot(self.minimum_points, abs(self.minima), "o", color="red")

        ax.hlines(1, min(self.X), max(self.X), colors='grey')

        if show_classical:
            ax.plot(self.X, T(self.n, self.X), "-", label='$T_n$')

        #ax.legend()
        ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        fig.subplots_adjust(right=0.6)
        ax.grid()
        ax.set_xlim(min(self.X), max(self.X))
        if not absolute:
            ax.set_ylim(-2, 2)
        else:
            ax.set_ylim(0, 2)
        fig.show()


    def __call__(self, x):
        return self.polynomial(x)

    def __repr__(self):
        return f"{self.polynomial}"


def chebyshev(n, E):
    nodes = np.array(random.choices(E, k=n+1))
    print(nodes)
    nodes.sort()
    Y = np.array([(-1)**(n-k) for k in range(n+1)])
    H = lagrange(nodes, Y)
    return H

def extrema(X, Y, eta=1):
    extrema_indices = np.isclose(abs(Y), eta)
    eta = X[extrema_indices]
    return X[extrema_indices], Y[extrema_indices]

def functional_gap(X, Y, absolute_bound):
    indices = np.isclose(abs(Y), absolute_bound, atol=0.001)
    return X[indices], Y[indices]


if __name__ == '__main__':
    X = np.linspace(-1, 1, 10000)
    p = ChebyshevPolynomial(3, X)
    p.generate_plot(show_classical=False)
    p.generate_plot(absolute=True, show_classical=False)
