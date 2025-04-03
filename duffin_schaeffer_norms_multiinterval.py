import json
import time

import numpy as np
import matplotlib.pyplot as plt 

from scipy.interpolate import lagrange
from scipy.linalg import norm
from chebyshev import ChebyshevPolynomial

from cli import build_static_site


if __name__ == '__main__':
    start_time = time.time()

    comparisons = 25000
    n = 4
    T = ChebyshevPolynomial.read("dc804fa.poly")
    #T = ChebyshevPolynomial(n=n)

    V = T.critical_points

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
        
        if not (i % 500):
            print(f"Checked {i+1} polynomials")
    
    results = {
        "T_n' norm": Tp_norm,
        "failures": failures,
        "max norm in comparison polynomial": max(norms)
    }
    print(f"Finished checking degree {n}:\n\n {results}")


print(results)
print(f"Run took {round(time.time() - start_time, 2)}s")