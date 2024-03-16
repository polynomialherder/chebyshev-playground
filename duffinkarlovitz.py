from scipy.linalg import norm

from chebyshev import ChebyshevPolynomial
import numpy as np

from itertools import combinations, combinations_with_replacement

import matplotlib.pyplot as plt


def lebesgue_from_nodes(nodes):
    # Form the monic polynomial (z - x_0)(z - x_1)...(z - x_n)
    ell = np.poly1d(nodes, r=True)
    ellp = ell.deriv()
    # Take l/(z - x_k) for k in 0 ... n
    numerators = [(ell/np.poly1d([r], r=True))[0] for r in nodes]
    # Define the l_k:
    l = [num/ellp(node) for num, node in zip(numerators, nodes)]
    # Return a function. In Python the lambda keyword signals an inline-defined function -- its unrelated
    # to the usual denotation of the Lebesgue function using the Greek letter lambda
    return lambda x: [abs(f.deriv()(x)) for f in l]


if __name__ == '__main__':
    X = np.linspace(-10, 10, 1_000_000)
    T = ChebyshevPolynomial.read("6118e86.poly")
    norms = []
    # For each k in (n+1, 2*n), generate a k-length list of indices
    # These indices will correspond to Ek sets from which we'll select
    # a random point. We do this rather than selecting randomly from E
    # directly so that we can keep track of patterns in "winning" node-sets
    for ekcomb in combinations_with_replacement(range(T.n), T.n+1):
        # We'll generate 10 nodesets, choosing each from the Ek sets listed in
        # ekcomb. For example, if ekcomb is (1
        for _ in range(1000):
            try:
                v = []
                for ek in ekcomb:
                    Ek = T.Ek[ek]
                    v.append(np.random.uniform(Ek.min(), Ek.max()))

                v.sort()

                lebesgue = lebesgue_from_nodes(v)
                norms.append(
                    {
                        "norm": norm(lebesgue(T.E), np.inf),
                        "v": v,
                        "comb": ekcomb
                    }
                )
            except Exception as e:
                print(f"Got {e}, continuing")

    norms.sort(key=lambda item: item["norm"])
    print(norms[0])

    v = np.array(norms[0]["v"])

    fig, ax = plt.subplots()
    T.plot_disks(ax=ax)
    ax.plot(v.real, v.imag, "o")
    fig.show()