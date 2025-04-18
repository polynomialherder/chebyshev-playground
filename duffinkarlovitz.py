from scipy.linalg import norm
from scipy.interpolate import lagrange

from chebyshev import ChebyshevPolynomial
import numpy as np

from itertools import combinations, combinations_with_replacement

import matplotlib.pyplot as plt

from scipy.linalg import norm


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
    return lambda x: sum([abs(f.deriv()(x)) for f in l])


def lebesgue_from_nodes_signed(nodes):
    # Form the monic polynomial (z - x_0)(z - x_1)...(z - x_n)
    ell = np.poly1d(nodes, r=True)
    ellp = ell.deriv()
    # Take l/(z - x_k) for k in 0 ... n
    numerators = [(ell/np.poly1d([r], r=True))[0] for r in nodes]
    # Define the l_k:
    l = [num/ellp(node) for num, node in zip(numerators, nodes)]
    # Return a function. In Python the lambda keyword signals an inline-defined function -- its unrelated
    # to the usual denotation of the Lebesgue function using the Greek letter lambda
    return [f.deriv() for f in l]


def construct_extremal(T):
    V = T.critical_points

    extremal = []

    for alternating_set in T.alternating_sets():
        Leb = lebesgue_from_nodes(alternating_set)
        max_idx = np.argmax(Leb(V))
        z0 = V[max_idx]

        Leb_derivs = lebesgue_from_nodes_signed(alternating_set)

        Y = [np.sign(f(z0)) for f in Leb_derivs]
        
        extremal.append(
            lagrange(alternating_set, Y)
        )
    return extremal
        






def lebesgue_function(A, z):
    A = np.array(A, dtype=float)
    n = len(A)
    total = 0.0
    
    for k in range(n):
        # Compute the derivative of the k-th Lagrange basis polynomial at z
        l_k_prime = 0.0
        for m in range(n):
            if m == k:
                continue
            
            # Indices j != k, j != m
            idx = [j for j in range(n) if j != k and j != m]
            
            # Partial product for derivative term
            numerator = np.prod(z - A[idx])
            denominator = np.prod(A[k] - A[idx]) * (A[k] - A[m])
            
            l_k_prime += numerator / denominator
        
        total += abs(l_k_prime)
    
    return total


def old():
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


import numpy as np
import matplotlib.pyplot as plt

def plot_extremal_construction(T):
    """Generate and save visualizations for a ChebyshevPolynomial T and its extremal constructions."""
    
    # First plot: original Chebyshev polynomial T
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    x = np.array(T.critical_points)
    for ax, (left, right) in zip(axes, [(0, 1), (2, 3)]):
        T.plot_Ek(ax=ax)
        ax.plot(x.real, x.imag, "o", color="green", label="Critical points for T")
        ax.set_xlim(T.Ek[left].min(), T.Ek[right].max())
        ax.set_ylim(-1, 1)
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.1), ncol=1)
    fig.subplots_adjust(bottom=0.25)
    fig.savefig("T-base-polynomial.png", bbox_inches="tight")
    plt.close(fig)

    # Plot each extremal polynomial constructed from T
    alternating_sets = T.alternating_sets()
    for k, p in enumerate(construct_extremal(T)):
        T_p = ChebyshevPolynomial(polynomial=p)
        alternating_set = np.array(alternating_sets[k])
        x = np.array(T_p.critical_points)

        fig, axes = plt.subplots(1, 2, figsize=(10, 6))
        for ax, (left, right) in zip(axes, [(0, 1), (2, 3)]):
            T_p.plot_Ek(ax=ax)
            ax.plot(x.real, x.imag, "o", color="green", label="Critical points for P")
            ax.plot(alternating_set.real, alternating_set.imag, "o", color="red", label="Alternating set")
            ax.set_xlim(T_p.Ek[left].min(), T_p.Ek[right].max())
            ax.set_ylim(-1, 1)
            ax.grid(True)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.1), ncol=2)
        fig.subplots_adjust(bottom=0.25)
        fig.savefig(f"extremal-polynomials-{k:03}.png", bbox_inches="tight")
        plt.close(fig)




if __name__ == '__main__':

    # Generate a Chebyshev polynomial and check that for at least one alternating set,
    # the Lebesgue function for that alternating set attains its maximum at max {min(E), max(E)} = max {min(A), max(A)}
    for trial in range(500):
        # Generate a Chebyshev polynomial
        T = ChebyshevPolynomial(n=4)

        # All extremal points of T
        E0 = T.critical_points 

        max_indices = []

        for alternating_set in T.alternating_sets():

            Leb = lebesgue_from_nodes(alternating_set)

            # Get the index at which the Lebesgue function attains 
            # its max absolute value on E
            #
            # We hope for this to be 0 or len(E) - 1
            max_indices.append(np.argmax(Leb(T.E)))

        admissible = [T.E.size - 1, 0]
        if not any(it in admissible for it in max_indices):
            print("Found a counterexample")

            # Generate plots of the counterexample Lebesgue functions on E
            alt_sets = list(T.alternating_sets())
            fig, axes = plt.subplots(nrows=len(alt_sets), figsize=(8, 3 * len(alt_sets)))

            # If there's only one alternating set, ensure 'axes' is iterable
            if len(alt_sets) == 1:
                axes = [axes]

            x = T.E  # domain points used for evaluation
            for i, alt_set in enumerate(alt_sets):
                ax = axes[i]

                Leb = lebesgue_from_nodes(alt_set)
                y = Leb(x)


                ax.plot(x, y, label='Lebesgue function')
                ax.plot(T.critical_points, [0]*len(T.critical_points), 'go', label='Critical points')
                ax.plot(alt_set, [0]*len(alt_set), 'ro', label='Alternating set')

                ax.set_title(f"Alternating set {i}")
                ax.legend()

            plt.tight_layout()
            plt.savefig("counterexample.png")
            plt.close(fig)


            break

        print(max_indices)

        if not (trial % 10):
            print(f"Checked {trial+1} examples")





