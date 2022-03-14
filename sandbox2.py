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


if __name__ == '__main__':

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


        @property
        def m(self):
            return self.C.n + 1


        @property
        def Cv(self):
            return self.C(v)

        @property
        def l(self):
            return LagrangePolynomial(self.v, self.Cv)


        @property
        def index_range(self):
            return range(self.m)

        @property
        def index_product(self):
            return product(self.index_range, self.index_range)


        def plot(self, store, title, n=100):
            for i in range(store.n):
                spawn(plot_angles, C, z, store, i, "Losing")
                if i > n:
                    break

        def plot_winning(self, n=100):
            self.plot(self.winning, "Winning", n)

        def plot_losing(self, n=100):
            self.plot(self.losing, "Losing", n)
                    

        def check_points(self, v):
            self.v = v

            idx_range = range(self.m)

            z = self.z
            l = self.l
            C = self.C

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



    X_ = np.linspace(-1, 1, 10000000)
    n = 3
    C = ChebyshevPolynomial(n, X_)

    X = np.linspace(-1, 1, 1000)
    Y = np.linspace(-1, 1, 1000)
    xv,  yv = np.meshgrid(X,  Y)
    zv = xv + 1j*yv


    winning_indices = {}
    losing_indices = {}
    Path("Trials").mkdir(exist_ok=True)
    Path(f"Trials/{C.n}/").mkdir(exist_ok=True)
    Path(f"Trials/{C.n}/{C.id}").mkdir(exist_ok=True)
    success = 0
    total = 0

    mask = outside_gaps_inside_Edisk(C, zv) & (zv.real > C.E_intervals[1][0]) & (zv.real < C.polynomial.r[1]) & (zv.imag > 0) & (zv.imag < 1.5*C.Ek_radii[1])
    z = random.choice(zv[mask])

    t = Experiment(z, C, C.n + 1)

    winning_indices = {}
    winning_indices["z"] = z
    winning_indices["indices"] = []
    winning_indices["v"] = []
    winning_indices["prod"] = []
    winning_indices["Cv"] = []
    winning_indices["angles"] = []
    winning_indices["factors1c"] = []
    winning_indices["factors1"] = []
    winning_indices["factors2"] = []
    winning_indices["cos"] = []
    winning_indices["factors3"] = []
    winning_indices["lemmas"] = []

    losing_indices = {}
    losing_indices["z"] = z
    losing_indices["indices"] = []
    losing_indices["v"] = []
    losing_indices["prod"] = []
    losing_indices["Cv"] = []
    losing_indices["angles"] = []
    losing_indices["factors1c"] = []
    losing_indices["factors1"] = []
    losing_indices["factors2"] = []
    losing_indices["cos"] = []
    losing_indices["factors3"] = []
    losing_indices["lemmas"] = []

    m = n+1
    for _ in range(200):
        negative, positive = signed_intervals(C.E, C.polynomial.r, C.n)
        #combs = combinations_with_replacement(negative + positive, n)
        combs = [
            (negative[0], positive[0], positive[0], negative[1]),
            (positive[0], positive[0], negative[1], positive[1]),
        ]
        for v in generate_nodes(combs):
            v = np.sort(v)
            t.check_points(v)

    #         Cv = C(v)
    #         l = LagrangePolynomial(v, Cv)

    #         l_seq = l.L_seq(z)

    #         idx_range = range(m)

    #         products = []
    #         angles = []
    #         cosines = []
    #         factors1c = []
    #         factors1 = []
    #         factors2 = []
    #         factors3 = []
    #         lemmas = []

    #         for i, j in product(idx_range, idx_range):

    #             ci = l.denominator(i)
    #             cj = l.denominator(j)
    #             num = l.numerator_special(i, j, z)
    #             xi = v[i]
    #             xj = v[j]
    #             Cxi = C(xi)
    #             Cxj = C(xj)
    #             factor1 = (z.conjugate() - xj.conjugate())
    #             factor2 = (z - xi)
    #             prod = factor1*factor2
    #             lemma = (prod).real if i != j else 1
    #             factor3 = Cxi*Cxj
    #             products.append(
    #                 Cxi*Cxj*num*(lemma)*1/(ci*cj)
    #             )
    #             lemmas.append(lemma)
    #             factors1c.append(z - xj)
    #             factors1.append(factor1)
    #             factors2.append(factor2)
    #             factors3.append(factor3)
    #             cos = prod.real/(abs(factor1)*abs(factor2))
    #             theta = np.rad2deg(np.arccos(cos))
    #             theta = theta if not np.isclose(theta, 0) else 0
    #             angles.append(
    #                 theta
    #             )
    #             cosines.append(cos)

    #         products = np.array(products)

    #         satisfies = np.all(products < 0) or np.all(products > 0) or np.all(products == 0)
    #         summed_products = sum(products)
    #         abs_Cz = abs(C(z))

    #         if not np.isclose(abs_Cz **2, summed_products):
    #             print(f"Discarding results for {v} since they are unreliable")
    #             continue

    #         if satisfies:

    #             success += 1
    #             winning_indices["v"].append(v)
    #             winning_indices["prod"].append(np.array(products))
    #             winning_indices["Cv"].append(C(v))
    #             winning_indices["angles"].append(angles)
    #             winning_indices["factors1c"].append(np.array(factors1c))
    #             winning_indices["factors1"].append(np.array(factors1))
    #             winning_indices["factors2"].append(np.array(factors2))
    #             winning_indices["cos"].append(np.array(cosines))
    #             winning_indices["factors3"].append(np.array(factors3))
    #             winning_indices["lemmas"].append(np.array(lemmas))


    #             # fig, ax = plt.subplots()
    #             # ax.plot(v, np.zeros(len(v)), "ro", label="$x_i$")
    #             # ax.plot(z.real, z.imag, "bo", label="$z$")
    #             # for factor1, factor2, color in zip(factors1, factors2, COLOR_CYCLE):
    #             #     ax.plot([0, factor1.real], [0, factor1.imag], "--o", color=color)
    #             #     ax.plot([0, factor2.real], [0, factor2.imag], "--o", color=color)

    #             # plot_disks(ax, C)
    #             # ax.set_title("Winning interpolation points")
    #             # ax.plot(C.Ek_midpoints, np.zeros(len(C.Ek_midpoints)), "bo")
    #             # fig.show()

    #         else:
    #             losing_indices["v"].append(v)
    #             losing_indices["prod"].append(np.array(products))
    #             losing_indices["Cv"].append(C(v))
    #             losing_indices["angles"].append(np.array(angles))
    #             losing_indices["factors1c"].append(np.array(factors1c))
    #             losing_indices["factors1"].append(np.array(factors1))
    #             losing_indices["factors2"].append(np.array(factors2))
    #             losing_indices["factors3"].append(np.array(factors3))
    #             losing_indices["cos"].append(np.array(cosines))
    #             losing_indices["lemmas"].append(np.array(lemmas))


    #         total += 1
    # print(C.critical_points, C.polynomial.r)
    # print(C.gap_midpoints, C.Ek_midpoints)
    # print(f"Found {success=} index sets out of {total=}")

    # for i in range(len(winning_indices["angles"])):
    #     spawn(plot_angles, C, z, winning_indices, i, "Winning")

    # for i in range(len(losing_indices["angles"])):
    #     spawn(plot_angles, C, z, losing_indices, i, "Losing")
    #     if i > 25:
    #         break

