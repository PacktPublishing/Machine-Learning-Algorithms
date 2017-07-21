from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score


# For reproducibility
np.random.seed(1000)

nb_samples = 50


def show_dataset(X, Y):
    fig, ax = plt.subplots(1, 1, figsize=(30, 25))

    ax.grid()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    ax.scatter(X, Y)

    plt.show()


if __name__ == '__main__':
    # Create dataset
    X = np.arange(-nb_samples, nb_samples, 1)
    Y = np.zeros(shape=(2 * nb_samples,))

    for x in X:
        Y[int(x) + nb_samples] = np.power(x * 6, 2.0) / 1e4 + np.random.uniform(-2, 2)

    # Show dataset
    #show_dataset(X, Y)

    # Create and train a Support Vector regressor
    svr = SVR(kernel='poly', degree=2, C=1.5, epsilon=0.5)
    svr_scores = cross_val_score(svr, X.reshape((nb_samples*2, 1)), Y, scoring='neg_mean_squared_error', cv=10)
    print('SVR CV average negative squared error: %.3f' % svr_scores.mean())

