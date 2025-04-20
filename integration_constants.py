import numpy as np

import matplotlib.pyplot as plt

from chebyshev import ChebyshevPolynomial


x_vals = np.linspace(14, 18, 100)


T = ChebyshevPolynomial(polynomial=1.55*np.poly1d([-6, -5, -2.9], r=True))

xv, yv, zv = T.grid()
Tzv = np.abs(T(zv))

Q = np.poly1d([T.Ek_midpoints[2], T.Ek_midpoints[1]], r=True)

Z = np.array(T.Ek_midpoints) + np.array(T.Ek_radii)*1j


def extremal_polynomial(T):
    # Construct a polynomial with roots at specified midpoints
    Q = np.poly1d([T.Ek_midpoints[1], T.Ek_midpoints[2]], r=True)
    
    # Integrate Q once to get QQ
    QQ = Q.integ(1)
    
    # Determine evaluation points: roots of derivative + critical points
    eval_pts = list(QQ.deriv().r) + T.critical_points
    
    # Compute min and max of QQ over evaluation points
    min_ = min(QQ(eval_pts))
    max_ = max(QQ(eval_pts))
    
    # Normalize the polynomial so it has zero average on the interval
    return T.normalize_polynomial(Q.integ(1, -(min_ + max_) / 2))


for k, x in enumerate(x_vals):
    # Build polynomial and integrate with constant x
    QQ = T.normalize_polynomial(Q.integ(1, x))
    
    # Compare magnitude against target
    diff = abs(QQ(Z)) - abs(T(Z))
    
    if any(diff > 0):
        val = f"{x:.3f} :: {max(diff):.5e} :: <<< found one at #{np.argmax(diff)} >>>"
    else:
        val = f"{x:.3f} :: {max(diff):.5e} (# {np.argmax(diff)})"
    
    print(val)

    # Plot and save frame
    fig, ax = plt.subplots()
    ax.contourf(xv, yv, Tzv > np.abs(QQ(zv)))
    T.plot_disks(ax=ax)
    fig.savefig(f"Animations/IntegrationConstantsGeneral/{k:03}.png")
    plt.close(fig)
