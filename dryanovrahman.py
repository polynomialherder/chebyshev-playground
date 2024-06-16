from functools import cached_property

import numpy as np

import matplotlib.pyplot as plt

from scipy.linalg import norm

class DRPolynomials:

    def __init__(self, n):
        self.n = n


    def weight(self, k):
        n = self.n
        return n**n/((2**n)*(k**k)*(n-k)**(n-k))


    def q(self, k):
        return np.poly1d([1, 1])**(k) * np.poly1d([-1, 1])  ** (self.n - k) * self.weight(k)

    @cached_property
    def polynomials(self):
        return [self.q(k) for k in range(self.n+1)]


    @cached_property
    def eta(self):
        return np.array([-1 + 2*k/self.n for k in range(self.n+1)])



    def normalize_polynomial(self, p):
        return p/norm(p(self.eta), np.inf)


    @staticmethod
    def calculate_grid(r, q, n):
        X = np.linspace(q - r, q + r, n)
        Y = np.linspace(-r, r, n)
        xv, yv = np.meshgrid(X, Y)
        return xv, yv, xv + 1j*yv


    def grid(self, n=1000):
        return self.calculate_grid(2., 0, n)


    def plot(self, saveto=None):
        fig, ax = plt.subplots()
        
        X = np.linspace(-1, 1, 1000)

        for p in self.polynomials:

            ax.plot(X, p(X), "-")

        self.plot_eta(ax)

        ax.grid()

        if saveto is None:
            fig.show()
            return

        fig.savefig(saveto)


    def plot_eta(self, ax):
        ax.plot(self.eta.real, self.eta.imag, "o", color="pink")



    def plot_disks(self, ax):
        disk = plt.Circle((0, 1/np.sqrt(3)), 2/np.sqrt(3), fill=False, color="salmon", linestyle="--")
        ax.add_patch(disk)

        disk = plt.Circle((0, -1/np.sqrt(3)), 2/np.sqrt(3), fill=False, color="salmon", linestyle="--")
        ax.add_patch(disk)

        E_disk = plt.Circle((0, 0), 1, fill=False, linestyle="-")
        ax.add_patch(E_disk)


    def __call__(self, x):
        return [p(x) for p in self.polynomials]


def generate_random_roots(n):

    domains = [
        (-100, -1),
        (1, 100)
    ]

    roots = []
    for k in range(n):
        idx = np.random.choice([0, 1])

        l, r = domains[idx]

        roots.append(
            np.random.uniform(l, r)
        )
    return roots


def random_polynomial(n):
    roots = generate_random_roots(n)
    return np.poly1d(roots, r=True)


def trials():
    for N in range(1, 15):
        P = DRPolynomials(N)

        fig, ax = plt.subplots()

        xv, yv, zv = P.grid()
        holds = np.ones(zv.shape)

        Pzv = np.max(np.abs(P(zv)), axis=0)

        for _ in range(100):
            p = random_polynomial(N)
            pp = P.normalize_polynomial(p)
            holds = np.logical_and(holds, Pzv > np.abs(pp(zv)))
    
        ax.contourf(xv, yv, holds)

        ax.plot(P.eta.real, P.eta.imag, "o", color="yellow")
        P.plot_disks(ax=ax)

        fig.suptitle(f"n={N}")

        fig.savefig(f"Animations/Comparisons/{N:03}.png")

        plt.close()

        fig, ax = plt.subplots()

        P.plot_disks(ax=ax)

        ax.contourf(xv, yv, np.argmax(np.abs(P(zv)), axis=0))
        
        ax.plot(P.eta.real, P.eta.imag, "o", color="yellow")
        P.plot_disks(ax=ax) 

        fig.savefig(f"Animations/PointMaximizer/{N:03}.png")

        plt.close()


def plot_real():
    for N in range(1, 10):
        P = DRPolynomials(N)
        P.plot(saveto=f"DRP{N:03}.png")

        plt.close()


if __name__ == '__main__':
    trials()