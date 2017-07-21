from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection

from sklearn.isotonic import IsotonicRegression


# For reproducibility
np.random.seed(1000)

nb_samples = 100


def show_dataset(X, Y):
    fig, ax = plt.subplots(1, 1, figsize=(30, 25))

    ax.plot(X, Y, 'b.-')
    ax.grid()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.show()


def show_isotonic_regression_segments(X, Y, Yi, segments):
    lc = LineCollection(segments, zorder=0)
    lc.set_array(np.ones(len(Y)))
    lc.set_linewidths(0.5 * np.ones(nb_samples))

    fig, ax = plt.subplots(1, 1, figsize=(30, 25))

    ax.plot(X, Y, 'b.', markersize=8)
    ax.plot(X, Yi, 'g.-', markersize=8)
    ax.grid()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.show()


if __name__ == '__main__':
    # Create dataset
    X = np.arange(-5, 5, 0.1)
    Y = X + np.random.uniform(-0.5, 1, size=X.shape)

    # Show original dataset
    show_dataset(X, Y)

    # Create an isotonic regressor
    ir = IsotonicRegression(-6, 10)
    Yi = ir.fit_transform(X, Y)

    # Create a segment list
    segments = [[[i, Y[i]], [i, Yi[i]]] for i in range(nb_samples)]

    # Show isotonic interpolation
    show_isotonic_regression_segments(X, Y, Yi, segments)
