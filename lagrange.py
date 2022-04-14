import numpy as np

from functools import cache, cached_property

class LagrangePolynomial:

    def __init__(self, nodes, values):
        self.nodes = np.array(nodes)
        self.values = list(values)

    @cached_property
    def indices(self):
        return np.arange(len(self.nodes))


    def numerator_piecemeal(self, i, z):
        prod = 1
        for idx in self.indices:
            if idx == i:
                continue
            prod *= (z - self.nodes[idx])
        return prod

    def denominator_piecemeal(self, i):
        prod = 1
        xi = self.nodes[i]
        for idx in self.indices:
            if idx == i:
                continue
            prod *= xi - self.nodes[idx]
        return prod

    def l_piecemeal(self, i, z):
        return self.numerator_piecemeal(i, z) / self.denominator_piecemeal(i)


    def L_piecemeal(self, z):
        sum_ = 0
        for i, x in enumerate(self.nodes):
            sum_ += self.values[i]*self.l_piecemeal(i, z)
        return sum_


    def numerator(self, i, z, j=None):
        where = self.indices != i
        return np.prod(z - self.nodes, where=where)


    def numerator_special(self, i, j, z):
        where = (self.indices != i) & (self.indices != j)
        return np.prod(np.abs(z - self.nodes)**2, where=where)


    def denominator_differences(self, i):
        xi = self.nodes[i]
        return xi - self.nodes[self.indices != i]


    def numerator_differences(self, i, z):
        return z - self.nodes[self.indices != i]


    def denominator(self, i):
        xi = self.nodes[i]
        return np.prod(xi - self.nodes, where=self.indices != i)


    def numerators(self, z):
        return [self.numerator(i, z) for i, _ in enumerate(self.nodes)]


    @property
    def denominators(self):
        return [self.denominator(i) for i, _ in enumerate(self.nodes)]


    def l(self, i, z):
        return self.numerator(i, z)/self.denominator(i)


    def lv(self, z):
        num = np.array(self.numerators(z))
        return num/np.array(self.denominators)


    def L(self, z):
        return self.L_seq(z).sum()


    def L_seq(self, z):
        return np.array(self.values)*self.lv(z)


    def __call__(self, z):
        return np.vectorize(self.L_piecemeal)(z)
