import multiprocessing as mp
import random

from dataclasses import dataclass, field
from itertools import combinations, combinations_with_replacement, cycle, pairwise, product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from numpy.ma import make_mask

from chebyshev import ChebyshevPolynomial
from lagrange import LagrangePolynomial

from matplotlib.patches import Arc
from matplotlib.transforms import IdentityTransform, TransformedBbox, Bbox

COLOR_CYCLE = cycle(["green", "orange", "purple", "brown", "olive", "cyan", "pink", "cornflowerblue"])


def outside_gaps_inside_Edisk(C, zv):
    outside_gap = make_mask(zv)
    for midpoint, radius in zip(C.gap_midpoints, C.gap_radii):
        distance = abs(midpoint - zv)
        outside_gap = outside_gap & (distance > radius) & ~np.isclose(distance, radius, atol=0.001)

    outside_Ek = make_mask(zv)
    for midpoint, radius in zip(C.Ek_midpoints, C.Ek_radii):
        distance = abs(midpoint - zv)
        outside_Ek = outside_Ek & (distance > radius) & ~np.isclose(distance, radius, atol=0.001)

    inside_E_disk = abs(C.E_midpoint - zv) < C.E_disk_radius
    return outside_gap & outside_Ek & inside_E_disk


def interpolation_points_to_string(C, v1):
    v0 = []
    for v in v1:
        for i, point in enumerate(C.critical_points):
            if point == v:
                v0.append(f"CP{i}")
        for i, point in enumerate(C.polynomial.r):
            if point == v:
                v0.append(f"R{i}")
        for i, point in enumerate(C.gap_midpoints):
            if point == v:
                v0.append(f"G{i}")
        for i, point in enumerate(C.Ek_midpoints):
            if point == v:
                v0.append(f"Ek{i}")
    return v0


def plot_disks(axis, C):
    for idx, disk_info in enumerate(zip(C.Ek_midpoints, C.Ek_radii)):
        midpoint, radius = disk_info
        disk = plt.Circle((midpoint, 0), radius, fill=False, linestyle="--")
        axis.add_patch(disk)
    for idx, disk_info in enumerate(zip(C.gap_midpoints, C.gap_radii)):
        midpoint, radius = disk_info
        disk = plt.Circle((midpoint, 0), radius, fill=False, linestyle="--")
        axis.add_patch(disk)

    E_disk = plt.Circle((C.E_midpoint, 0), C.E_disk_radius, fill=False, linestyle="-")
    axis.add_patch(E_disk)


def plot_angles(C, z, store, i, title=""):
    fig, ax = plt.subplots()
    ax.plot(z.real, z.imag, "o", color="darkviolet", label="$z$")
    fig.suptitle(title)
    factors1 = store.factors1c[i]
    factors2 = store.factors2[i]
    Cv = store.Cv[i]
    lemmas = store.lemmas[i]
    v = store.v[i]
    sign_pattern = tuple(np.sign(Cv).astype(int))
    lemma_pattern = tuple(np.sign(lemmas).astype(int))
    ax.set_title(f"{sign_pattern}\n" + f"{lemma_pattern}")
    ax.plot(v, np.zeros(len(v)), "ro", label="$x_i$")

    for start, end in combinations(v, 2):
        midpoint = (start + end)/2
        radius = (end - start)/2
        circ = plt.Circle((midpoint, 0), radius, fill=False, linestyle="dashdot", color="blue")
        ax.add_patch(circ)

    ax.plot(C.polynomial.r, np.zeros(len(C.polynomial.r)), "*", label="Roots of $C_n$")
    plot_disks(ax, C)
    ax.grid()
    ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    fig.subplots_adjust(right=0.6)
    fig.savefig(f"Trials/{C.n}/{C.id}/{title.lower()}-{i}.png")


def label_points(v, C):
    labels = []
    roots = C.polynomial.r
    for x in v:
        if x < roots[0]:
            labels.append("Q1")
        elif roots[0] < x < roots[1]:
            labels.append("Q2")
        else:
            labels.append("Q3")
    return labels


def signed_intervals(E, roots, n):
    E_start = [min(E)]
    E_end = [max(E)]
    roots = list(roots)
    points = sorted(E_start + roots + E_end)
    negative_intervals = []
    positive_intervals = []
    correction_factor = 1 if n % 2 else 0
    for i, interval_info in enumerate(pairwise(points)):
        if (i + correction_factor) % 2:
            negative_intervals.append(interval_info)
        else:
            positive_intervals.append(interval_info)
    return negative_intervals, positive_intervals


def generate_nodes(combs):
    for configuration in combs:
        nodes = []
        for start, end in configuration:
            midpoint = (start + end)/2
            nodes.append(
                np.random.uniform(start, end)
            )
        yield np.array(nodes)


def spawn(func, *args):
    proc = mp.Process(target=func, args=args)
    proc.start()
    proc.join()


@dataclass
class Store:
    v: list = field(default_factory=list)
    Cv: list = field(default_factory=list)
    angles: list = field(default_factory=list)
    factors1c: list = field(default_factory=list)
    factors1: list = field(default_factory=list)
    factors2: list = field(default_factory=list)
    factors3: list = field(default_factory=list)
    products: list = field(default_factory=list)
    cos: list = field(default_factory=list)
    lemmas: list = field(default_factory=list)

    @property
    def n(self):
        return len(self.v)


@dataclass
class Experiment:

    z: complex
    C: ChebyshevPolynomial
    v: list = None
    winning: Store = Store()
    losing: Store = Store()
    success: int = 0
    discarded: int = 0


    @property
    def m(self):
        return self.C.n + 1


    @property
    def Cv(self):
        return self.C(self.v)

    def l(self, v, Cv):
        return LagrangePolynomial(v, Cv)


    @property
    def index_range(self):
        return range(self.m)

    @property
    def index_product(self):
        return product(self.index_range, self.index_range)


    def plot(self, store, title, n=100):
        for i in range(store.n):
            spawn(plot_angles, C, z, store, i, title)
            if i > n:
                break

    def plot_winning(self, n=100):
        self.plot(self.winning, "Winning", n)

    def plot_losing(self, n=100):
        self.plot(self.losing, "Losing", n)


    def check_points(self, v, z):

        self.v = v

        idx_range = range(self.m)

        C = self.C
        l = self.l(v, self.Cv)

        products = []
        lemmas = []
        factors1c = []
        factors1 = []
        factors2 = []
        factors3 = []
        angles = []
        cosines = []
        for i, j in self.index_product:

            ci = l.denominator(i)
            cj = l.denominator(j)
            num = l.numerator_special(i, j, z)
            xi = v[i]
            xj = v[j]
            Cxi = C(xi)
            Cxj = C(xj)
            factor1 = (z.conjugate() - xj.conjugate())
            factor2 = (z - xi)
            prod = factor1*factor2
            lemma = (prod).real if i != j else 1
            factor3 = Cxi*Cxj
            cos = prod.real/(abs(factor1)*abs(factor2))
            theta = np.rad2deg(np.arccos(cos))
            theta = theta if not np.isclose(theta, 0) else 0


            products.append(
                Cxi*Cxj*num*(lemma)*1/(ci*cj)
            )
            lemmas.append(lemma)
            factors1c.append(z - xj)
            factors1.append(factor1)
            factors2.append(factor2)
            factors3.append(factor3)
            angles.append(
                theta
            )
            cosines.append(cos)

        products = np.array(products)

        satisfies = np.all(products < 0) or np.all(products > 0) or np.all(products == 0)
        summed_products = sum(products)
        abs_Cz = abs(C(z))

        if not np.isclose(abs_Cz **2, summed_products):
            print(f"Discarding results for {v} since they are unreliable")
            self.discarded += 1
            return False

        if satisfies:

            self.success += 1
            self.winning.v.append(v)
            self.winning.products.append(np.array(products))
            self.winning.Cv.append(C(v))
            self.winning.angles.append(angles)
            self.winning.factors1c.append(np.array(factors1c))
            self.winning.factors1.append(np.array(factors1))
            self.winning.factors2.append(np.array(factors2))
            self.winning.cos.append(np.array(cosines))
            self.winning.factors3.append(np.array(factors3))
            self.winning.lemmas.append(np.array(lemmas))


        else:
            self.losing.v.append(v)
            self.losing.products.append(np.array(products))
            self.losing.Cv.append(C(v))
            self.losing.angles.append(np.array(angles))
            self.losing.factors1c.append(np.array(factors1c))
            self.losing.factors1.append(np.array(factors1))
            self.losing.factors2.append(np.array(factors2))
            self.losing.factors3.append(np.array(factors3))
            self.losing.cos.append(np.array(cosines))
            self.losing.lemmas.append(np.array(lemmas))

        return satisfies


    def check_points_vectorized(self, v, zv):

        t = self
        l = self.l(v, self.C(v))

        def check_points(z):

            idx_range = range(t.m)

            C = t.C

            products = []

            current = None
            for i, j in t.index_product:

                ci = l.denominator(i)
                cj = l.denominator(j)
                num = l.numerator_special(i, j, z)
                xi = v[i]
                xj = v[j]
                Cxi = C(xi)
                Cxj = C(xj)
                factor1 = (z.conjugate() - xj.conjugate())
                factor2 = (z - xi)
                prod = factor1*factor2
                lemma = (prod).real if i != j else 1
                product = Cxi*Cxj*num*(lemma)*1/(ci*cj)
                current_sign = np.sign(product)
                if current is None:
                    current = current_sign
                    continue

                if current_sign ==  current:
                    current = current_sign
                    continue

                return False
            return True

        return np.vectorize(check_points)(zv)


    def plot_lemma_(self, xv, yv, zv, v=None, holds=None, uniquifier=""):
        fig, ax = plt.subplots()
        if not holds:
            holds = self.check_points_vectorized(v, zv)
            ax.plot(v, np.zeros(len(v)), "ro", label="$x_i$")
        cmap = plt.contourf(xv, yv, holds)
        plot_disks(ax, self.C)
        ax.set_xlim(self.C.E.min(), self.C.E.max())
        fig.colorbar(cmap)
        ax.plot(self.C.polynomial.r, np.zeros(len(self.C.polynomial.r)), "*", label="Roots of $C_n$")
        
        fig.savefig(f"Trials/{C.n}/{C.id}/lemma-{uniquifier}.png")


    def plot_lemma(self, xv, yv, zv, v=None, holds=None, uniquifier=""):
        spawn(self.plot_lemma_, xv, yv, zv, v, uniquifier)



if __name__ == '__main__':
    n = 3
    X_ = np.linspace(-1, 1, 10000000)

    for _ in range(100):

        X = np.linspace(-1, 1, 10000)
        Y = np.linspace(-1, 1, 10000)
        xv,  yv = np.meshgrid(X,  Y)
        zv = xv + 1j*yv

        print(f"Generating a Chebyshev polynomial and set E on [-1, 1]")
        C = ChebyshevPolynomial(n, X_)
        print(f"Generated the Chebyshev polynomial {C.id}")

        try:
            print(f"Masking the regions of z we are interested in")
            mask = outside_gaps_inside_Edisk(C, zv) & (zv.real > C.E_intervals[1][0]) #& (zv.real < C.polynomial.r[1]) & (zv.imag > 0) & (zv.imag < 1.5*C.Ek_radii[1])
            z = random.choice(zv[mask])

            t = Experiment(z, C, C.n + 1)
        except:
            print(f"Couldn't find any z in the region of interest; retrying")
            continue

        trials = 200
        print(f"Masking was a success! Beginning {trials} trials")
        m = n+1
        for _ in range(trials):
            negative, positive = signed_intervals(C.E, C.polynomial.r, C.n)
            combs = combinations_with_replacement(negative + positive, m)
            # combs = [
            #     (negative[0], positive[0], positive[0], negative[1]),
            #     (positive[0], positive[0], negative[1], positive[1]),
            # ]
            X = np.linspace(-1, 1, 100)
            Y = np.linspace(-1, 1, 100)
            xv,  yv = np.meshgrid(X,  Y)
            zv = xv + 1j*yv
            holds = np.zeros(zv.shape)
            for i, v in enumerate(generate_nodes(combs)):
                v = np.sort(v)
                holds = np.logical_or(holds, t.check_points_vectorized(v, zv))
                t.check_points(v, z)

        t.plot_lemma(xv, yv, zv, holds=holds)

        # print(f"Finished trials; found {t.winning.n} sets of interpolation points satisfying the lemma for {C.id}, discarded {t.discarded}")

        # Path("Trials").mkdir(exist_ok=True)
        # Path(f"Trials/{C.n}/").mkdir(exist_ok=True)
        # Path(f"Trials/{C.n}/{C.id}").mkdir(exist_ok=True)

        # print(f"Generating plots of winning sets")
        # t.plot_winning(n=t.winning.n)
        # print(f"Generating plots of losing sets")
        # t.plot_losing(n=25)

        # X = np.linspace(-1, 1, 100)
        # Y = np.linspace(-1, 1, 100)
        # xv,  yv = np.meshgrid(X,  Y)
        # zv = xv + 1j*yv

        # for i, v in enumerate(t.winning.v):
        #     print(f"Generating plot of z for which winning set #{i} satisfies the lemma")
        #     t.plot_lemma(xv, yv, zv, v, uniquifier=f"-winning-{i}")


        # for i, v in enumerate(t.losing.v):
        #     print(f"Generating plot of z for which losing set #{i} satisfies the lemma")
        #     t.plot_lemma(xv, yv, zv, v, uniquifier=f"-losing-{i}")

            

        

        



