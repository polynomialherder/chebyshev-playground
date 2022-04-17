""" What follows is a mess. It's not worth trying to make sense of. At the present it's a tangled web of uncommented numerical spaghetti,
    tight coupling, poor API design, inconsistent naming conventions, god objects, DRY violations, performance fences, and sprawling methods and functions.

    Also there's a concurrency bug that introduces a race condition that only hasn't corrupted my SQLite db because of sheer luck.
    Anyway caveat emptor, cave canem, use at your own risk. I'll clean it up eventually, promise :)
"""

import io
import multiprocessing as mp
import random
import sqlite3

from dataclasses import dataclass, field
from itertools import combinations, combinations_with_replacement, cycle, pairwise, product
from os.path import join
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

    ax.plot(C.gap_critical_values, np.zeros(len(C.gap_critical_values)), "*", color="pink", label="Maxima of $C_n$")
    ax.plot(C.polynomial.r, np.zeros(len(C.polynomial.r)), "*", label="Roots of $C_n$")
    plot_disks(ax, C)
    ax.grid()
    ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    fig.subplots_adjust(right=0.6)
    fig.savefig(f"Trials/{C.n}/{C.id}/{title.lower()}-{i}.png")


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


def generate_nodes(intervals):
    for interval in intervals:
        nodes = []
        for start, end in interval:
            midpoint = (start + end)/2
            nodes.append(
                np.random.uniform(start, end)
            )
        yield np.array(nodes)


def spawn(func, *args, **kwargs):
    proc = mp.Process(target=func, args=args, kwargs=kwargs)
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

    C: ChebyshevPolynomial
    m_: int = None
    winning: Store = Store()
    losing: Store = Store()
    all: Store = Store()
    success: int = 0
    discarded: int = 0
    trial_number: int = 0
    sqlite_db: str = "chebyshev.db"
    grouped: bool = True
    cached_real_plot: io.BytesIO = None


    @property
    def m(self):
        if self.m_:
            return self.m_
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



    def create_db(self):
        sql = """
        CREATE TABLE IF NOT EXISTS trials(
          node_plot BLOB,
          cumulative_plot BLOB,
          real_plot BLOB,
          polynomial_id TEXT,
          degree INTEGER,
          coefficients TEXT,
          m INTEGER,
          trial_number INTEGER,
          resolution TEXT,
          holds_for INTEGER,
          X TEXT,
          CX TEXT,
          sgn_CX TEXT,
          group_indices TEXT,
          group_nodes TEXT,
          C_group TEXT,
          sgn_C_group TEXT
        );
        """
        self.execute_sql(sql)


    def trial_record(self, v, zv, holds, trial_number, bytes_current, bytes_cumulative, bytes_real):
        CX = list(self.C(v))
        y, x = zv.shape
        resolution = x*y
        sgn_CX = list(np.sign(CX))
        coefficients = list(self.C.polynomial.coef)
        if self.grouped:
            group_indices = list(self.C.group_nodes(v))
            grouping = []
            for index_set in group_indices:
                group = [v[i] for i in index_set]
                grouping.append(group)
            C_group = [list(self.C(group)) for group in grouping]
            sgn_C_group = [list(np.sign(group)) for group in C_group]
            grouping_data = {
                "group_indices": str(group_indices),
                "group_nodes": str(grouping),
                "C_group": str(C_group),
                "sgn_C_group": str(sgn_C_group),
            }
        else:
            grouping_data = {
                "group_indices": None,
                "group_nodes": None,
                "C_group": None,
                "sgn_C_group": None
            }
        return {
            "polynomial_id": self.C.id,
            "degree": self.C.n,
            "m": self.m,
            "trial_number": trial_number,
            "resolution": resolution,
            "holds_for": int(holds.sum()),
            "X": str(list(v)),
            "CX": str(CX),
            "sgn_CX": str(sgn_CX),
            "coefficients": str(coefficients),
            "node_plot": sqlite3.Binary(bytes_current.getbuffer()),
            "cumulative_plot": sqlite3.Binary(bytes_cumulative.getbuffer()),
            "real_locations": sqlite3.Binary(bytes_real.getbuffer()),
            **grouping_data
        }


    def insert_trials(self, records):

        sql = """INSERT INTO trials(
                       polynomial_id, degree, m, trial_number, resolution, holds_for, X,
                       CX, sgn_CX, coefficients, group_indices, group_nodes, C_group, sgn_C_group,
                       node_plot, cumulative_plot, real_plot
                 )
                 VALUES(
                       :polynomial_id, :degree, :m, :trial_number, :resolution, :holds_for, :X,
                       :CX, :sgn_CX, :coefficients, :group_indices, :group_nodes, :C_group, :sgn_C_group,
                       :node_plot, :cumulative_plot, :real_locations
                 )
        """
        with sqlite3.connect(self.sqlite_db) as conn:
            print(f"Dumping {len(records)} to database {self.sqlite_db}")
            conn.executemany(sql, records)
            conn.commit()

    def execute_sql(self, sql):
        with sqlite3.connect(self.sqlite_db) as conn:
            conn.execute(sql)
            conn.commit()


    def plot_bytes(self, all):
        path = self.plot_lemma_path(all=all)
        with open(path, "rb") as f:
            return f.read()


    def check_points(self, v, z):

        idx_range = range(self.m)

        C = self.C
        l = self.l(v, C(v))

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


    def check_points_grouped(self, v, z):
        """ Return a boolean array indicating over which z the inequality
            G_k(z)*G_n(z) holds for all k, z pairs

            TODO: - Decompose this method into smaller methods/functions, moving logic into
                    the ChebyshevPolynomial as appropriate
                  - Lots of opportunities to improve memory and compute performance
        """

        t = self
        l = self.l(v, self.C(v))

        idx_range = range(t.m)

        C = t.C
        abs_Cz = np.abs(C(z))

        current = None

        Gk = []

        # TODO: We use a tuple here to take advantage of the method's lru cache
        #       In practice this logic should be internal to the method
        #v0 = tuple(v)

        # For each interpolation node in v, group its index with a gap or an interval Ek
        # together with other indices corresponding to nodes in that gap or Ek
        grouped = C.group_nodes(v)
        if len(grouped) == 1:
            return np.zeros(z.shape)

        for indices in grouped:

            if not indices:
                continue

            # Compute the sum G_k for each of the nodes in the group
            G = 0
            for idx in indices:

                G += C(v[idx])*l.l_piecemeal(idx, z)

            Gk.append(G)

        # Initialize a boolean array of True values. This will be our accumulator
        # that we'll be and'ing to check whether the inequality holds
        holds = np.ones(z.shape)
        # Compute conj(G_k) * G_n for all k, n pairs
        # Track each product term so that we can make sure the sum
        # aligns with |C(z)|^2
        products = []
        range_Gk = range(len(Gk))
        for i, j in product(range_Gk, range_Gk):

            prod_ = (Gk[i].conjugate()*Gk[j]).real
            products.append(prod_)

            # Adjust the boolean array tracking the region where the product is nonnegative
            holds = np.logical_and(holds, prod_ >= 0)

        # The sum of the products should be reasonably close to |C(z)|^2, otherwise
        # our results are not meaningful
        if not np.all(np.isclose(abs_Cz **2, sum(products), atol=1e-4)):
            print(f"Discarding results for {v} since they are unreliable")
            return np.zeros(z.shape)
        return holds



    def check_points_vectorized(self, v, zv):

        t = self
        l = self.l(v, self.C(v))
        self.all.v.append(v)

        def check_points(z):

            idx_range = range(t.m)

            C = t.C

            products = 0

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
                    products += product
                    continue

                if current_sign == current:
                    current = current_sign
                    products += product
                    continue

                return False

            abs_Cz = abs(C(z))

            if not np.all(np.isclose(abs_Cz **2, products, atol=0.1)):
                print(f"Discarding results for {v} since they are unreliable (got {abs_Cz=} vs {products=})")
                return False

            return True

        return np.vectorize(check_points)(zv)


    def plot_lemma_path(self, all=False):
        grouped = "-ungrouped-" if not self.grouped else "-grouped-"
        all = "-single-" if not all else "-cumulative-"
        slug = "lemma-{grouped}{all}{self.m}-{self.trial_number}"
        return self.plot_path(slug)


    def plot_real_locations_path(self):
        slug = f"real-locations"
        return self.plot_path(slug)


    def plot_path(self, slug):
        base_path = f"Trials/{self.C.n}/{self.C.id}/"
        return join(base_path, f"{slug}.png")


    def plot_lemma_(self, xv, yv, zv, v=None, holds=None, uniquifier=""):

        # Configure the axes separately so that we have enough room for
        # the caption in the grouped case (TODO: Don't add the extra spacing if we aren't
        # plotting the grouped case)
        fig = plt.figure()
        ax = fig.add_axes((.1, .20, 0.70, 0.70))

        # Plot and recalculate the groupings (TODO: We should cache the last grouping and reuse that here)
        if holds is None:
            holds = self.check_points_vectorized(v, zv)
        if v is not None:
            ax.plot(v, np.zeros(len(v)), "ro", label="$x_i$")
        cmap = plt.contourf(xv, yv, holds)
        plot_disks(ax, self.C)
        if v is not None:
            ax.set_title(f"X = {np.round(v, 3)}")
            if (2 in v) or (-2 in v):
                ax.set_xlim(-1, 1)
            else:
                ax.set_xlim(self.C.E.min(), self.C.E.max())
        else:
            ax.set_xlim(self.C.E.min(), self.C.E.max())
        ax.set_ylim(-self.C.E_disk_radius, self.C.E_disk_radius)
        fig.colorbar(cmap)
        ax.plot(self.C.gap_critical_values, np.zeros(len(self.C.gap_critical_values)), "*", color="pink", label="Maxima of $C_n$")
        ax.plot(self.C.polynomial.r, np.zeros(len(self.C.polynomial.r)), "*", label="Roots of $C_n$")

        if v is not None and self.grouped:
            groups = self.C.group_nodes(v)
            group_text = f""
            for i, group in enumerate(groups):
                x = [v[j] for j in group]
                group_text += "$I_" f"{i}" + "$" + f" = {group}" + "$; X_" + f"{i}" +  "=$" + f"{np.round(x, 3)}"+"; $C_n(X_" + f"{i})" +" =$" + f"{np.round(self.C(x), 3)}" + "\n"
            fig.text(0.1, 0.0, group_text, fontsize=10)


        bytestream=io.BytesIO()
        fig.savefig(bytestream)

        path = self.plot_lemma_path(v is None)
        with open(path, "wb") as f:
            f.write(bytestream.getbuffer())
            bytestream.seek(0)
        return bytestream


    def plot_lemma(self, xv, yv, zv, v=None, holds=None, uniquifier=""):
        spawn(self.plot_lemma_, xv, yv, zv, v=v, holds=holds, uniquifier=uniquifier)


    def plot_real_points_(self):
        if self.cached_real_plot is not None:
            return self.cached_real_plot

        fig, ax = plt.subplots()

        X = np.linspace(C.E.min(), C.E.max(), 10000)
        Y = np.linspace(-C.E_disk_radius, C.E_disk_radius, 10000)
        xv, yv = np.meshgrid(X, Y)
        zv = xv + yv*1j

        is_real = np.isclose(abs(self.C(zv).imag), 0, atol=1e-2)
        cmap = plt.contourf(xv, yv, is_real)
        ax.plot(self.C.gap_critical_values, np.zeros(len(self.C.gap_critical_values)), "*", color="pink", label="Maxima of $C_n$")
        ax.plot(self.C.polynomial.r, np.zeros(len(self.C.polynomial.r)), "*", label="Roots of $C_n$")
        fig.colorbar(cmap)
        plot_disks(ax, self.C)
        ax.set_xlim(self.C.E.min(), self.C.E.max())
        ax.set_ylim(-self.C.E_disk_radius, self.C.E_disk_radius)
        path = self.plot_real_locations_path()
        bytestream = io.BytesIO()
        fig.savefig(bytestream)
        # Only write the plot to disk on the first iteration (TODO: Make this write-to-disk logic a method)
        if not self.trial_number:
            with open(path, "wb") as f:
                f.write(bytestream.getbuffer())
                bytestream.seek(0)
        self.cached_real_plot = bytestream
        return bytestream


    def perform_trial_(self, xv, yv, zv, holds_current, holds_cumulative, v=None, trial_number=0):
        m = self.m
        grouped = "grouped" if self.grouped else "ungrouped"
        print(f"Plotting region where the lemma holds so far ({grouped}, {m=}, iteration #{trial_number})")
        bytes_current = self.plot_lemma_(xv, yv, zv, v=v, holds=holds_current)
        bytes_cumulative = self.plot_lemma_(xv, yv, zv, holds=holds_cumulative)
        bytes_real = self.plot_real_points_()

        if self.sqlite_db:

            record = self.trial_record(v, zv, holds_grouped_current, trial_number, bytes_current, bytes_cumulative, bytes_real)
            self.insert_trials([record])


    def perform_trial(self, xv, yv, zv, holds_current, holds_cumulative, v=None, trial_number=0):
        spawn(self.perform_trial_, xv, yv, zv, holds_current, holds_cumulative, v=v, trial_number=trial_number)



if __name__ == '__main__':
    n = 3
    X_ = np.linspace(-1, 1, 100000000)

    for _ in range(150):

        X = np.linspace(-1, 1, 10000)
        Y = np.linspace(-1, 1, 10000)
        xv,  yv = np.meshgrid(X,  Y)
        zv = xv + 1j*yv

        print(f"Generating a Chebyshev polynomial and set E on [-1, 1]")
        C = ChebyshevPolynomial(n, X_)
        E_interval = (C.E.min(), C.E.max())
        print(f"Generated the Chebyshev polynomial {C.id}")

        trials = 1000
        print(f"Beginning {trials} trials")
        X = np.linspace(-1, 1, 1000)
        Y = np.linspace(-1, 1, 1000)
        xv,  yv = np.meshgrid(X,  Y)
        zv = xv + 1j*yv

        for j in range(1, 2*n):
            m = n+j
            t = Experiment(C, m, grouped=True)
            #t2 = Experiment(C, m, grouped=False)
            t.create_db()
            roots = C.polynomial.r

            gap_points = list(np.array(C.gap_midpoints) + np.array(C.gap_radii)/2) + list(np.array(C.gap_midpoints) - np.array(C.gap_radii)/2)
            combs = combinations(C.critical_points, m)
            Path("Trials").mkdir(exist_ok=True)
            Path(f"Trials/{C.n}/").mkdir(exist_ok=True)
            Path(f"Trials/{C.n}/{C.id}").mkdir(exist_ok=True)
            holds_grouped_cumulative = np.zeros(zv.shape)
            holds_cumulative = np.zeros(zv.shape)
            for i, v in enumerate(combs):
                v = np.sort(v)
                holds_grouped_current = t.check_points_grouped(v, zv)
                holds_grouped_cumulative = np.logical_or(holds_grouped_cumulative, holds_grouped_current)

                t.perform_trial(xv, yv, zv, holds_grouped_current, holds_grouped_cumulative, v=v, trial_number=t.trial_number)

                t.trial_number += 1

                #holds_current = t.check_points_vectorized(v, zv)
                #holds_cumulative = np.logical_or(holds_cumulative, holds_current)

                # t2.perform_trial(xv, yv, zv, holds_current, holds_cumulative, v=v, trial_number=t2.trial_number)

                # t2.trial_number += 1

