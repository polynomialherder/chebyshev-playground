import json

import numpy as np
import matplotlib.pyplot as plt 

from scipy.interpolate import lagrange
from scipy.linalg import norm
from chebyshev import ChebyshevPolynomial

from numexp import ExperimentLogger
from cli import build_static_site


if __name__ == '__main__':
    logger = ExperimentLogger()


    comparisons = 200
    n = 4
    T = ChebyshevPolynomial(n=n)

    V = T.critical_points

    V = [V[0], V[1], V[2], V[4], V[5]]

    Tp = T.polynomial.deriv()
    Tp_norm = norm(Tp(V), np.inf)


    parameters = {
        "normalized_on": "all critical points",
        "comparisons": comparisons,
        "degrees": f"{n}",
        "E": list(T.Ek_intervals),
        "T": list(T.coef),
        "extremal_points": list(T.critical_points),
        "alternating_set": list(V), 
        "T' norm": Tp_norm
    }

    logger.start_experiment(
        name="Duffin-Schaeffer Estimates (multiple interval, normalized on 2n critical points)",
        tags=["duffin-schaeffer"],
        comment="""This script checks that
$$||T_n'||_E \ge ||P'||_E$$

holds, where $P$ is degree $n$ and has norm $1$ on a set $B$ is the full set of extremal points for $T_n$.

Polynomials $P$ are chosen by choosing n random roots from $[- \inf E, \sup E]$, then normalizing the resulting monic polynomial on the set $B$.

This process outputs a datafile 

**Parameters**: {{PARAMETERS}}.

### Duffin, R. J. and A. C. Schaeffer
[A refinement of an inequality of the brothers Markoff](https://www.jstor.org/stable/1990125), *Trans. Amer. Math. Soc., **50** (1941), 517-528
        """.replace("{{PARAMETERS}}", json.dumps(parameters, indent=4))
    )

    print("Parameters", json.dumps(parameters, indent=4))
    results = {}

    failures = 0


    norms = []


    for i in range(comparisons):
        r = np.random.uniform(T.E.min(), T.E.max(), n)
        Q = np.poly1d(r, r=True)
        QQ = Q / norm(Q(V), np.inf)

        QQp_norm = norm(QQ.deriv()(V), np.inf)

        passes = QQp_norm < Tp_norm

        if not passes:
            failures += 1

        norms.append(QQp_norm)
        
        if not (i % 25):
            print(f"Checked {i+1} polynomials")
    
    results = {
        "T_n' norm": Tp_norm,
        "failures": failures,
        "max norm in comparison polynomial": max(norms)
    }
    print(f"Finished checking degree {n}:\n\n {results}")


print(results)
logger.save_dict(results, "output.json")
logger.end_experiment()
build_static_site("experiments.db", "dist")