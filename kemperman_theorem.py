import numpy as np
import matplotlib.pyplot as plt 

from scipy.interpolate import lagrange
from scipy.linalg import norm
from chebyshev import ChebyshevPolynomial

if __name__ == '__main__':
    X = np.linspace(-10, 10)

    T = ChebyshevPolynomial.classical(4)
    xv, yv, zv = T.grid()

    Pzv = np.abs(T.deriv.deriv()(zv))

    holds = np.ones(zv.shape)
    for k in range(1000):
        r = np.random.uniform(-1, -1, 4)
        Q = np.poly1d(r, r=True)
        QQ = Q / norm(Q(T.E), np.inf)

        holds = np.logical_and(holds, Pzv > np.abs(QQ.deriv(2)(zv)))
        if not all(abs(T.deriv.deriv()(T.deriv.r)) > np.abs(QQ.deriv(2)(T.deriv.r))):
            print(f"Found one")
            break

    fig, ax = plt.subplots()
    ax.contourf(xv, yv, holds)
    ax.plot(T.deriv.deriv().r.real, T.deriv.deriv().r.imag, "o", color="yellow", label="Roots of P")
    ax.plot(T.deriv.r.real, T.deriv.r.imag, "o", color="green", label="Roots of P")
    ax.plot(T.r.real, T.r.imag, "o", color="blue", label="Roots of P")

    T.plot_disks(ax=ax) 
    ax.set_aspect('equal')
    fig.show()
