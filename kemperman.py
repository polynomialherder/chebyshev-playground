import numpy as np
import matplotlib.pyplot as plt 

from scipy.interpolate import lagrange
from scipy.linalg import norm
from chebyshev import ChebyshevPolynomial

if __name__ == '__main__':
    X = np.array([-2, -1, 1, 2])
    P = lagrange(X, [1, -1, -1, 1])
    P = np.poly1d(P.coef[1::])

    V = np.array([-2, -1.5, -1, 2])

    T = ChebyshevPolynomial(n=2, X=X)
    xv, yv, zv = T.calculate_grid(2, 0, 1000)

    Pzv = np.abs(P(zv))

    holds = np.ones(zv.shape)
    for _ in range(1000):
        Q = np.poly1d(np.random.uniform(-50, 50, 2), r=True)
        QQ = Q / norm(Q(V)/P(V), np.inf)

        holds = np.logical_and(holds, Pzv > np.abs(QQ(zv)))

    fig, ax = plt.subplots()
    ax.contourf(xv, yv, holds)
    ax.plot(V.real, V.imag, "o", color="red", label="Extremal points")
    ax.plot(P.r.real, P.r.imag, "o", color="green", label="Roots of P")
    
    # Adding dashed circles
    for i in range(0, len(V), 2):
        center = (V[i] + V[i+1]) / 2
        radius = abs(V[i+1] - V[i]) / 2
        circle = plt.Circle((center, 0), radius, color='blue', linestyle='dashed', fill=False)
        ax.add_patch(circle)
    
    ax.legend()
    ax.set_aspect('equal')
    fig.show()
