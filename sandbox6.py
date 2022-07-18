import itertools
import random

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from chebyshev import ChebyshevPolynomial
from sandbox2 import Experiment

from numpy.ma import make_mask
from scipy.interpolate import lagrange
from scipy.linalg import norm

from lagrange import LagrangePolynomial


def get_region_of_interest(C, zv):
    outside_gap = make_mask(zv)
    for midpoint, radius in zip(C.gap_midpoints, C.gap_radii):
        distance = abs(midpoint - zv)
        outside_gap = outside_gap & (distance > radius)

    outside_Ek = make_mask(zv)
    for midpoint, radius in zip(C.Ek_midpoints, C.Ek_radii):
        distance = abs(midpoint - zv)
        outside_Ek = outside_Ek & (distance > radius)

    inside_E_disk = abs(C.E_midpoint - zv) < C.E_disk_radius
    return inside_E_disk & outside_Ek


def compute_grouping(C, v, groups, zv=None):
    Gk = []
    l = LagrangePolynomial(v, C(v))
    L = []
    for group in groups:
        G = 0
        for idx in group:
            l_ = C(v[idx])*l.l_piecemeal(idx, zv)
            G += l_
            L.append(l_)
        Gk.append(G)

    products = []
    range_Gk = range(len(Gk))
    holds = np.ones(zv.shape)
    for i, j in product(range_Gk, range_Gk):
        prod_ = (Gk[i].conjugate()*Gk[j]).real
        holds = np.logical_and(holds, prod_ >= 0)
        products.append(prod_)

    return holds, products, Gk, L


def plot_inequality(dx, scale_factor=1, fit=False):
    v = C.critical_points
    v = [v[0], C.critical_points[2], v[3], v[3] + dx]
    print(v)
    holds, _, _, _ = compute_grouping(C, v, groupings, zv=zv)
    fig, ax = plt.subplots()
    ax.contourf(xv, yv, holds)
    C.plot_disks(ax=ax)
    ax.plot(C.gap_critical_values, [0], "o")
    ax.plot(v, np.zeros(len(v)), "o", color="red")
    if fit:
        ax.set_xlim(scale_factor*C.E.min(), scale_factor*C.E.max())
        ax.set_ylim(-scale_factor*C.E_disk_radius, scale_factor*C.E_disk_radius)
    fig.show()



def check_points(C, v, zv):
    l = LagrangePolynomial(v, C(v))
    L = [(l.values[i]*l.l_piecemeal(i, zv)*l.values[j].conjugate()*l.l_piecemeal(j,zv).conjugate()).real for i, j in product(range(len(v)), range(len(v)))]
    abs_Cz = abs(C(zv))
    m = len(L)
    sum_ = np.zeros(zv.shape)
    if not np.all(np.isclose(abs_Cz**2, sum(L), atol=0.001)):
        print(f"Discarding results for {v} since they are unreliable (got {abs_Cz=} vs {sum(L)=})")
        return sum_.astype(bool)
    for term in L:
        sum_ += np.sign(term)
    return np.isclose(abs(sum_), m), L


def circle_points(r, q, N=100):
    theta = np.linspace(0, 2 * np.pi, N+1)
    theta = theta[0:-1]
    X = q + r*np.cos(theta)
    Y = r*np.sin(theta)
    return X + 1j*Y


def draw_disks(ax, nodes, groupings):
    for grouping in groupings:
        for left, right in itertools.combinations(grouping, 2):
            r = abs(right - left)/2
            q_real = (right.real + left.real)/2
            q_imag = (right.imag + left.imag)/2
            q = q_real + q_imag*1j
            d = plt.Circle((q.real, q.imag), r, fill=False, linestyle="--", color="blue")
            ax.add_patch(d)


def generate_testplots():
     for i in range(1, 150):
        v = C.critical_points
        v = [v[0] - i*0.01, v[0], v[2], v[4], v[5], v[5] + i*0.01]
        holds = compute_grouping(C, v, groupings, zv)
        fig, ax = plt.subplots()
        ax.contourf(xv, yv, holds[0])
        C.plot_disks(ax=ax)
        ax.set_xlim(q - r, q + r)
        ax.set_ylim(-r, r)
        fig.savefig(f"Experiments/1/{i}.png")


def demonstrate_points(Cn, q, r, v, grouping, extra=None):
     X = np.linspace(q - r, q + r, 1000)
     Y = np.linspace(-r, r, 1000)
     xv, yv = np.meshgrid(X, Y)
     zv = xv + yv*1j
     holds = compute_grouping(Cn, v, grouping, zv)
     fig, ax = plt.subplots()
     ax.set_ylim(-r, r)
     ax.set_xlim(q - r, q + r)
     ax.contourf(xv, yv, holds[0])
     Cn.plot_disks(ax=ax)
     ax.plot(v, np.zeros(len(v)), "o", color="red")
     draw_disks(ax, v, grouping)
     if extra is not None:
         ax.plot(extra.real, extra.imag, "o", color="blue")
     fig.savefig(f"Experiments/{Cn.id}-{int(q)}-{int(r)}.png")




if __name__ == '__main__':
    # Harmonic measure - way of measuring size in relation
    # to the whole set
    X = np.linspace(-5, 5, 10000000)
    p = np.poly1d([-2.35, -0.95, 1], r=True)
    p = p/p.coef[0]
    #p = np.poly1d([ 5.16946816e+02, -3.95728706e+02,  3.63319434e+01,  1.91041047e-01])
    #np.poly1d([6.22671448, -2.50559731, -34.90056338, 35.89397685])
    p = np.poly1d([ 1.20591564,  2.10019598, -1.62893573, -1.10327262])
    for _ in range(1):
        C = ChebyshevPolynomial(X=X, polynomial=np.poly1d([2.31235651, -5.9913754, -8.0986466, 21.21137011]))

        #region = get_region_of_interest(C, zv)
        #max_in_region = abs(C(zv[region])).max()

        # E disk center is inside a gap
        # [  6.22671448  -2.50559731 -34.90056338  35.89397685]
        dx = 1000 #C.E_disk_radius/100
        print(f"Generating plots polynomial#{C.id} with coefficients {C.polynomial.coef}")
        print(f"{C.Ek_radii=}, {C.Ek_midpoints}, {C.E_disk_radius=}, {C.E_midpoint=}, {C.polynomial.r}")
        #print(f"{max_in_region=}")
        for elem in [
                (C.E_disk_radius, C.E_midpoint, "left-centered-on-Edisk"),
                #(C.gap_radii[1], C.gap_midpoints[1], "left-centered-on-G1")
                #(3*C.E_disk_radius, C.E_midpoint, "left"),
                #(1.5*C.Ek_radii[0], C.Ek_midpoints[0], "left-centered-on-E0"),
                #(1.5*C.Ek_radii[1], C.Ek_midpoints[1], "left-centered-on-E1")
        ]:
            print(f"Generating plots for experiment with {elem}")
            r, x, experiment_name = elem
            xlim_left, xlim_right = (x - r, x + r)
            ylim_below, ylim_above = (-r, r)
            X = np.linspace(xlim_left, xlim_right, 1000)
            Y = np.linspace(ylim_below, ylim_above, 1000)
            xv, yv = np.meshgrid(X, Y)
            zv = xv + yv*1j
            Xl = np.linspace(C.gaps[1].min(), C.gaps[1].max(), 10000)
            Yl = np.linspace(-C.gap_radii[1], C.gap_radii[1], 10000)
            xvl, yvl = np.meshgrid(Xl, Yl)
            zvl = xvl + yvl*1j


            Xp = np.linspace(C.gaps[1].min(), C.gaps[1].max(), 10000)
            Yp = np.linspace(C.gaps[1].min(), C.gaps[1].max(), 10000)
            xvp, yvp = np.meshgrid(Xp, Yp)
            zvp = xvp + yvp*1j
            real_cv = zv[np.logical_and(np.isclose(C(zv).imag, 0, atol=0.01), ~np.isclose(zv.imag, 0, atol=0.001))]
            v0 = C.critical_points

            Path(f"Groupings/{C.id}/").mkdir(exist_ok=True)
            Path(f"Groupings/{C.id}/{experiment_name}/").mkdir(exist_ok=True)
            complete = np.zeros(zv.shape).astype(bool)
            d = circle_points(C.E_disk_radius, C.E_midpoint, N=100)
            g0 = circle_points(C.gap_radii[0], C.gap_midpoints[0], N=1000)
            g1 = circle_points(C.gap_radii[1], C.gap_midpoints[1], N=1000)
            f = circle_points(1.00015*C.E_disk_radius, C.E_midpoint, N=1000)
            for i in range(0, 1000):
                # Degree 3 case
                # v = [v0[0] - dx*i, v0[0], v0[2], v0[5],  v0[5] + dx*i]
                # #v = [v0[0] - dx*i, v0[1], v0[2], v0[4], v0[7] + dx*i]
                # g0 = [0, 3, 4]
                # g1 = [1, 2]
                # groups = [g0, g1]

                # Right hand side
                # n = 3, m = 6
                # [1, -1, 1]
                # [1]
                #v = [v0[0], v0[2], v0[4], v0[5]]
                #v = [v0[0] + 0j, v0[1] + 0j, v0[3] + 0j, v0[5] + 0j]
                disks = [
                    (C.gap_midpoints[0], C.gap_radii[0]),
                    (C.gap_midpoints[1], C.gap_radii[1]),
                    (C.E_midpoint, C.E_disk_radius),
                    (C.Ek_midpoints[0], C.Ek_radii[0]),
                    (C.Ek_midpoints[1], C.Ek_radii[1])
                ]
                q, r = random.choice(disks)
                roots = sorted(C.polynomial.r)
                r = C.E_disk_radius
                q = C.E_midpoint
                z = q+r*np.cos(7*np.pi/8) + r*np.sin(7*np.pi/8)*1j #np.random.choice(circle_points(r, q))
                q0 = C.gap_midpoints[1]
                r0 = C.gap_radii[1]
                z0 = q0+r0*np.cos(7*np.pi/8) + r0*np.sin(7*np.pi/8)*1j

                #v = list(critical_points) + list(C.gap_critical_values)
                critical_points = np.random.choice(C.critical_points, 4, replace=False)
                theta = np.random.uniform(0, 2*np.pi)
                t = lambda z, theta: z.real*np.cos(theta) - z.imag*np.sin(theta) + (z.real*np.sin(theta) + z.imag*np.cos(theta))*1j
                #v = sorted([x for x in critical_points])
                # v = []
                # for x in critical_points:
                #     imag_factor = np.random.choice([0, (-1)**(np.random.randint(2))])
                #     v.append(x + imag_factor*0.5j)
                #x_g1 = np.random.choice(zv[abs(zv - C.gap_midpoints[0]) < C.gap_radii[0]])
                x_g0 = np.random.choice(zv[abs(zv - C.gap_midpoints[0]) >= C.gap_radii[0]])
                x_g1 = np.random.choice(g1)
                d_ = np.random.choice(zv[abs(zv - C.E_midpoint) >= C.E_disk_radius])
                node = np.random.choice(critical_points)
                # v.sort()
                v = [
                    x_g0,
                    x_g1,
                    d_,
                    node
                ]
                groups = []
                print(f"(Iteration #{i}) Checking group {v=} with {C(v)} and grouping strategy {groups}")

                for group_idx, group in enumerate(groups):
                    g = [v[m] for m in group]
                    print(f"C(G_{group_idx}) = {C(g)}")
                grouping = False
                if grouping:
                    holds, products, _ = compute_grouping(C, v, groups)

                    acz = abs(C(zv))**2
                    prods = sum(products)
                    is_reasonable = np.all(np.isclose(acz, prods, atol=1e-4))
                    if not is_reasonable:
                        print(f"Iteration #{i} produced an unreasonable result -- discarding and not plotting")
                else:
                    holds = check_points(C, v, zv)
                    complete = np.logical_or(complete, holds)


                fig, ax = plt.subplots()
                cm = ax.contourf(xv, yv, holds)
                fig.colorbar(cm)
                ax.plot(np.array(v).real, np.array(v).imag, "o", color="red")
                ax.plot(C.E_midpoint, 0, "o", color="blue")
                C.plot_disks(ax=ax)
                C.plot_roots(ax=ax)
                C.plot_critical_values(ax=ax)
                draw_disks(ax, v)
                ax.set_xlim(xlim_left, xlim_right)
                ax.set_ylim(ylim_below, ylim_above)
                v_display = [round(item.real, 3) + round(item.imag)*1j for item in v]
                ax.set_title(f"X={v_display}")
                path = f"Groupings/{C.id}/{experiment_name}/{i:04}.png"
                #fig.show()
                fig.savefig(path)

                fig, ax = plt.subplots()
                cm = ax.contourf(xv, yv, complete)
                fig.colorbar(cm)
                ax.plot(C.E_midpoint, 0, "x", color="blue")
                C.plot_disks(ax=ax)
                C.plot_roots(ax=ax)
                C.plot_critical_values(ax=ax)
                ax.set_xlim(xlim_left, xlim_right)
                ax.set_ylim(ylim_below, ylim_above)
                path = f"Groupings/{C.id}/{experiment_name}/extent-{i:04}.png"
                fig.savefig(path)

