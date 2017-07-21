from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize


# For reproducibility
np.random.seed(1000)

# Number of samples
nb_samples = 200


def loss(v):
    e = 0.0
    for i in range(nb_samples):
        e += np.square(v[0] + v[1]*X[i] - Y[i])
    return 0.5 * e


def gradient(v):
    g = np.zeros(shape=2)
    for i in range(nb_samples):
        g[0] += (v[0] + v[1]*X[i] - Y[i])
        g[1] += ((v[0] + v[1]*X[i] - Y[i]) * X[i])
    return g


def show_dataset(X, Y):
    fig, ax = plt.subplots(1, 1, figsize=(30, 25))

    ax.scatter(X, Y)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid()

    plt.show()


if __name__ == '__main__':
    # Create dataset
    X = np.arange(-5, 5, 0.05)

    Y = X + 2
    Y += np.random.uniform(-0.5, 0.5, size=nb_samples)

    # Show the dataset
    show_dataset(X, Y)

    # Minimize loss function
    result = minimize(fun=loss, x0=np.array([0.0, 0.0]), jac=gradient, method='L-BFGS-B')

    print('Interpolating rect:')
    print('y = %.2fx + %2.f' % (result.x[1], result.x[0]))

    # Compute the absolute error
    err = 0.0

    for i in range(nb_samples):
        err += np.abs(Y[i] - (result.x[1]*X[i] + result.x[0]))

    print('Absolute error: %.2f' % err)