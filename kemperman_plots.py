import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from sympy.polys.polyfuncs import ChebyshevPolynomial

T = ChebyshevPolynomial.classical(5).polynomial

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.ravel()

for i, k in enumerate(range(1, 5)):
    ax = axes[i]
    Tp = T.deriv(k)
    Tf = T.deriv(k - 1)
    E = np.linspace(Tf.r.min(), Tf.r.max(), 1000)

    if i == 0:
        # Collect labels only in the first subplot
        ax.plot(Tf.r.real, Tf.r.imag, 'o', color='green', 
                label=fr'Roots of $T_5^{{({k-1})}}(x)$')
        ax.plot(E, Tp(E), '--', color='red', 
                label=fr'$T_5^{{({k})}}(x)$')
        ax.plot(E, Tf(E), '--', color='blue', 
                label=fr'$T_5^{{({k-1})}}(x)$')
    else:
        ax.plot(Tf.r.real, Tf.r.imag, 'o', color='green')
        ax.plot(E, Tp(E), '--', color='red')
        ax.plot(E, Tf(E), '--', color='blue')

    N = round(norm(Tp(E), np.inf), 2)
    ax.set_ylim(-N, N)
    ax.set_title(f'k = {k}')
    ax.grid(True)

plt.tight_layout()

# Fetch handles/labels from the first subplot
handles, labels = axes[0].get_legend_handles_labels()

# Place a single legend in the upper-right corner
fig.legend(handles, labels, loc='upper right')

fig.savefig('T5_all.png')
plt.show()
