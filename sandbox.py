import numpy as np

import matplotlib.pyplot as plt

from chebyshev import T, ChebyshevPolynomial

from scipy.linalg import norm

def l(x, nodes, j):
    prod = 1
    for i, xi in enumerate(nodes):
        if i == j:
            continue
        prod *= (x-xi)/(nodes[j] - xi)
    return prod


def plot_lagrange_polynomials(X, nodes):
    for i, _ in enumerate(nodes):
        plt.plot(X, l(X, nodes, i), label="$\\ell_" + f"{i}$")

    for i, node in enumerate(nodes):
        plt.plot(node, 0, "o", label="$x_" + f"{i}$")

    plt.grid()
    plt.legend()
    plt.show()


def Ln(f, x, nodes, i):
    return f(nodes[i])*l(x, nodes, i)

def alternating_Ln(x, nodes, i):
    return (-1)**(nodes.size-1-i)*l(x, nodes, i)

def chebyshev(X, nodes):
    vec = np.zeros(X.size)
    for i, node in enumerate(nodes):
        vec += alternating_Ln(X, nodes, i)
    return vec

def interaction():
    X = np.linspace(-1, 1, 1000)
    nodes = np.array([ 1.,  0.80901699,  0.30901699, -0.30901699, -0.80901699, -1.])
    TX = T(nodes.size-1, X)
    DX = TX/norm(TX, np.Inf)
    ell = chebyshev(X, nodes)
    plt.plot(X, ell, label="$\\ell$")
    plt.plot(X, DX, label="$D_n$")
    for i, node in enumerate(nodes):
        plt.plot(node, 0, "ro", label="$x_i$")
    plt.grid()
    plt.legend()
    plt.show()


def visualize_proof_of_52():
    theta = np.linspace(0, 1, 1000, endpoint=True)
    w1 = -3
    w2 = 3
    d = (1-theta)*w1 + theta*w2
    m = (w2.real + w1.real)/2 + (w2.imag + w1.imag)*1j/2
    diameter_length = np.sqrt((w2.real - w1.real)**2 + (w2.imag - w1.imag)**2)
    fig, ax = plt.subplots()
    disk = plt.Circle((m.real, m.imag), diameter_length/2, fill=False, linestyle="-")
    z = 4
    factor1 = z.conjugate() - w1.conjugate()
    factor2 = z - w2
    product = factor1*factor2
    real_part = product.real
    cos = real_part / (abs(factor1) * abs(factor2))
    rads = np.arccos(cos)
    degrees = np.rad2deg(rads)
    ax.add_patch(disk)
    ax.plot(d.real, d.imag, "-", label="D")
    ax.plot(w1.real, w1.imag, "o", label="$W_1$")
    ax.plot(w2.real, w2.imag, "o", label="$W_2$")
    ax.plot(m.real, m.imag, "o", label="Midpoint")
    ax.plot(z.real, z.imag, "o", label="$z$")
    ax.plot([0, factor1.real], [0, factor1.imag], "--o", label="$\\overline{z} - \\overline{w_1}$")
    ax.plot([0, factor2.real], [0, factor2.imag], "--o", label="$z - w_2$")

    ax.plot(product.real, product.imag, "o", label="$(\\overline{z} - \\overline{w_1})(z - w_2)$")
    ax.plot(real_part, 0, "o", label="Real part")
    ax.grid()
    fig.legend()
    fig.show()


if __name__ == '__main__':
    X = np.linspace(-5, 5, 100000)

    C = ChebyshevPolynomial(6, X, compare=False)
    C.generate_plots()
