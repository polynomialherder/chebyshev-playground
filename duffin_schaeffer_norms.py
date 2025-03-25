import numpy as np
import matplotlib.pyplot as plt 

from scipy.interpolate import lagrange
from scipy.linalg import norm
from chebyshev import ChebyshevPolynomial

from numexp import ExperimentLogger
from cli import build_static_site


if __name__ == '__main__':
    logger = ExperimentLogger()


    lower, upper = 2, 6
    comparisons = 1000

    parameters = {
        "normalized_on": "[-1, 1]",
        "comparisons": comparisons,
        "degrees": f"{lower}-{upper}"
    }

    logger.start_experiment(
        name="Duffin-Schaeffer Estimates (classical, normalized on E)",
        tags=["kemperman-theorem", "inequalities"],
        comment="""This script checks that
$$||T_n'||_E \ge ||P'||_E$$

holds, where $P$ is degree $n$ and has norm $1$ on a set $E = T^{-1}([-1, 1])$.

Polynomials $P$ are chosen by choosing n random roots from $[-1, 1]$, then normalizing the resulting monic polynomial on $E$.

This experiment checks the results proven by Duffin & Schaeffer as a consistency check for the overall process. It outputs a datafile 

**Parameters**: {{PARAMETERS}}.

### Duffin, R. J. and A. C. Schaeffer
[A refinement of an inequality of the brothers Markoff](https://www.jstor.org/stable/1990125), *Trans. Amer. Math. Soc., **50** (1941), 517-528
        """.replace("{{PARAMETERS}}", f"{parameters}")
    )

    results = {}

    for n in range(lower, upper):

        print(f"Checking degree {n} Chebyshev polynomials")

        T = ChebyshevPolynomial.classical(n)
        E = T.E
        Tp_norm = norm(T.deriv(T.E), np.inf)

        norms = []
        failures = 0

        for i in range(comparisons):
            r = np.random.uniform(-1, -1, n)
            Q = np.poly1d(r, r=True)
            QQ = Q / norm(Q(E), np.inf)

            QQp_norm = norm(QQ.deriv()(E), np.inf)

            passes = QQp_norm < Tp_norm

            if not passes:
                failures += 1

            norms.append(QQp_norm)
            
            if not (i % 25):
                print(f"Checked {i+1} polynomials")
        
        results[n] = {
            "T_n' norm": Tp_norm,
            "failures": failures,
            "max norm in comparison polynomial": max(norms)
        }
        print(f"Finished checking degree {n}:\n\n {results[n]}")


    print(results)
    logger.save_dict(results, "output.json")
    logger.end_experiment()
    build_static_site("experiments.db", "dist")