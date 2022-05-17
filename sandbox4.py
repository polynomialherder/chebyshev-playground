import numpy as np
import matplotlib.pyplot as plt

from chebyshev import ChebyshevPolynomial
from sandbox2 import plot_disks

import cplot

if __name__ == '__main__':

    X = np.linspace(-1.5, 1.5, 100000000)
    Cn = ChebyshevPolynomial(3, X)

    fig, ax = plt.subplots(figsize=(14, 12))
    print(f"Now generating plot of Re(C) where {C.id=}")

    X = np.linspace(-1.5, 1.5, 10000)
    Y = np.linspace(-1.5, 1.5, 10000)

    xv, yv = np.meshgrid(X, Y)
    zv = xv + yv*1j


    Czv = C(zv)

    imaginary_part_near_zero = np.logical_and(np.isclose(abs(Czv.imag), 0, atol=1e-2), np.logical_and(C.E.min() < zv.real, zv.real < C.E.max()))
    cm = ax.contourf(xv, yv, imaginary_part_near_zero)
    fig.colorbar(cm)

    test_point = zv[imaginary_part_near_zero][0]
    x = test_point.real
    y = test_point.imag
    G = C.gap_critical_values
    # Degree 3 case
    # a = (G[1] - G[0])/2
    # h = (G[0] + G[1])/2

    a = (G[-1] - G[0])/2
    h = (G[0] + G[-1])/2
    b = np.sqrt((y**2 * a**2)/(abs((x-h)**2 - a**2)))
    c = np.sqrt(a**2 + b**2)
    f1 = h + c
    f2 = h - c
    positive = lambda x: b*np.sqrt((x - h)**2/a**2 - 1 + 0*1j)
    negative = lambda x: -b*np.sqrt((x - h)**2/a**2 - 1 + 0*1j)


    left = X[X <= G[0]]
    right = X[G[-1] <= X]

    curve_label = r"$\frac{x^2}{" + f"{x:.3}" + r"^2} - \frac{y^2}{" + f"{c:.2}" + r"^2} = 1$"
    curve_label = f"$\\frac{{x^2}}{{{{{a:.2}}}^2}} - \\frac{{y^2}}{{{{{b:.2}}}^2}} = 1$"
    ax.plot(X, b*(X-h)/a, "--", color="white")
    ax.plot(X, -b*(X-h)/a, "--", color="white")
    ax.plot(left, negative(left), "--", color="red")
    ax.plot(left, positive(left), "--", color="red", label=curve_label)
    ax.plot(right, positive(right), "--", color="red")
    ax.plot(right, negative(right), "--", color="red")
    ax.plot(f1, 0, "*", color="fuchsia", label="Foci")
    ax.plot(f2, 0, "*", color="fuchsia")
    ax.plot(h, 0, "*", color="cyan", label="M")
    print(f"{f1=}, {f2=}")


    eps = C.E_disk_radius/10
    border = abs(f1) + eps

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    plot_disks(ax, C)
    fig.legend()
    fig.savefig(f"degree-{C.n}-hyperbola-{C.id}.png")

        # Compare derivatives of Chebyshev with derivatives of arbitrary normalized polynomials
