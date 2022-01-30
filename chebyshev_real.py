import random
import time
import warnings

from itertools import permutations, pairwise
from functools import partial, cached_property
from random import uniform

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from numpy.polynomial.polynomial import Polynomial, polyfromroots
from scipy.linalg import norm
from scipy.interpolate import lagrange

warnings.filterwarnings('ignore')

matplotlib.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r'\usepackage{amsmath}'
})

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

    def __init__(self, n, X, seed=1, absolute=False):
        self.n = n
        self.X = X
        self.seed = seed
        self.domain_size = X.size
        self.absolute = absolute
        # Figure and access properties for plotting
        self.fig = None
        self.ax = None

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
        return Poly(self.polynomial)

    @cached_property
    def comparison_polynomials(self):
        return [polynomial_factory(self.n, self.E) for i in range(10)]


    def initialize_plot(self, size=(20, 16)):
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
            #label = self.label(idx, "$C_n(G_k))$")
            y = self.handle_absolute(self(gap))
            self.ax.plot(gap, y, "--", color="indigo", label=None)

    def plot_Ek(self):
        for idx, E in enumerate(self.Ek):
            curve_label = self.label(idx, f"$C_n :$ {self.p._repr_latex_()}")
            domain_label = self.label(idx, "$E_k$")
            y =  self.handle_absolute(self(E))
            self.ax.plot(E, y, "-r", label=curve_label, color="indigo")
            self.ax.hlines(0, E.min(), E.max(), label=domain_label, colors="indigo", linewidth=10)

    def plot_comparison_polynomials(self):
        for p in self.comparison_polynomials:
            y = self.handle_absolute(p(self.X))
            self.ax.plot(self.X, y, "-", label=f"{p._repr_latex_()}")


    def plot_left_and_right(self):
        left_y = self.handle_absolute(self(self.left))
        right_y = self.handle_absolute(self(self.right))
        self.ax.plot(self.left, left_y, "--", label=None, color="indigo")
        self.ax.plot(self.right, right_y, "--", label=None, color="indigo")


    def plot_extrema(self):
        self.ax.plot(self.maximum_points, self.maxima, "o", color="red")
        minima = self.handle_absolute(self.minima)
        self.ax.plot(self.minimum_points, minima, "o", color="red")

    def plot_guidelines(self):
        if not self.absolute:
            self.ax.hlines(-1, self.X.min(), self.X.max(), colors='grey')
        self.ax.hlines(1, min(self.X), max(self.X), colors='grey')

    def plot_classical(self):
        self.ax.plot(self.X, T(self.n, self.X), "-", label='$T_n$')

    def configure_legend(self):
        self.ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        self.fig.subplots_adjust(right=0.6)

    def configure_plot(self):
        self.ax.grid()
        self.ax.set_xlim(min(self.X), max(self.X))
        if not self.absolute:
            self.ax.set_ylim(-2, 2)
        else:
            self.ax.set_ylim(0, 2)


    def save_plot(self, path):
        self.fig.savefig(path)

    def show_plot(self):
        self.fig.show()

    def generate_plot(self, show=True, save_to=None):
        self.initialize_plot()
        self.plot_Ek()
        self.plot_gaps()
        self.plot_comparison_polynomials()
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
        return self.polynomial(x)

    def __repr__(self):
        return f"{self.polynomial}"


if __name__ == '__main__':
    X = np.linspace(-1, 1, 10000)
    p = ChebyshevPolynomial(3, X)
    p.generate_plot(save_to="chebyshev-polynomial.png")
    p.absolute = True
    p.generate_plot(save_to="chebyshev-polynomial-absolute.png")
