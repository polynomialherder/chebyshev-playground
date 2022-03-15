import numpy as np

class LagrangePolynomial:

    def __init__(self, nodes, values):
        self.nodes = np.array(nodes)
        self.values = list(values)

    @property
    def indices(self):
        return np.arange(len(self.nodes))


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
