import numpy as np
import matplotlib.pyplot as plt 

from scipy.interpolate import lagrange
from scipy.linalg import norm
from chebyshev import ChebyshevPolynomial

if __name__ == '__main__':
    # Initialize a Chebyshev polynomial
    n = 3
    T = ChebyshevPolynomial.classical(n) 
    TT = T.polynomial

    # 100k trials
    counterexample = False
    for _ in range(1000):
        # Initialize random trial polynomial and normalize 
        # at the extremal points of T
        Q = np.poly1d(np.random.uniform(-1, 1, n), r=True)
        QQ = T.normalize_polynomial(Q)
        for k in range(1, n):
            # Grab the roots of the (k-1)st derivative of T_n
            r = np.sort(TT.deriv(k-1).r)
            M = min(abs(TT.deriv(k)(r)))
            min_ = np.min(abs(QQ.deriv(k)(r) - M))
            if (counterexample := not (np.all(abs(QQ.deriv(k)(r)) < M))):
                print(f"Found one")
                break
        if counterexample:
            break


