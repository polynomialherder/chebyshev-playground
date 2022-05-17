from itertools import takewhile

import matplotlib.pyplot as plt
import numpy as np

from numpy.ma import make_mask

from chebyshev import ChebyshevPolynomial


if __name__ == '__main__':

    X_ = np.linspace(-5, 5, 100000000)
    X = np.linspace(-100, 100, 100000)
    for n in (3, 4, 5):
        comparison_polynomials = [ChebyshevPolynomial(n, X_).polynomial for _ in range(50000)]
        print(f"Generated {len(comparison_polynomials)} comparison polynomials")
        for j in range(10):
            C = ChebyshevPolynomial(n=n, X=X_)
            print(f"Generated {C.id}, a degree {n} polynomial")
            print(f"Generated polynomial {C.id}")
            D = C.polynomial.deriv()

            holds = make_mask(np.ones(X.size))
            fig, ax = plt.subplots()
            ax.plot(X, D(X), color="springgreen")
            n_comparisons = 50000
            plotted_polynomials = []
            print(f"Performing {len(comparison_polynomials)} comparisons")
            for p in comparison_polynomials:
                pn = C.normalize_polynomial(p)
                holds = np.logical_and(holds, abs(D(X)) > pn.deriv()(X))

            intervals = []
            current = holds[0]
            i = 0
            print(f"Calculating intervals")
            while i < len(holds):
                pred = lambda x: x == current
                w = list(takewhile(pred, holds[i:]))
                new_idx = i + len(w)
                if current:
                    intervals.append((X[i], X[new_idx-1]))
                i = new_idx
                current = not current

            C.plot_Cn(ax=ax)
            C.plot_critical_values(ax=ax)

            if intervals:
                print(f"D_n is greater on the intervals {intervals}")
                start, end = intervals[0]
                ax.hlines(-2, start, end, colors="springgreen", linewidth=10, label=f"Region where $\\lvert C_n' \\rvert > \\lvert p'_n \\rvert$")
                for start, end in intervals[1:]:
                    ax.hlines(-2, start, end, colors="springgreen", linewidth=10)

            if not intervals:
                continue

            if intervals:
                start_x = -100 #min(intervals[0][0], -15)
                end_x = 100 #max(intervals[-1][-1], 15)
            else:
                start_x = -15
                end_x = 15

            print(f"D_n is larger for {sum(holds)} points")
            ax.set_xlim(start_x, end_x)
            ax.set_ylim(-2, 2)
            ax.grid()
            fig.legend()
            path = f"Derivatives/9-derivative-order-{n}-{j}.png"
            fig.savefig(path)
            print(path)

