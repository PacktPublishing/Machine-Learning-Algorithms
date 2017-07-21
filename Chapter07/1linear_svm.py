from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


# For reproducibility
np.random.seed(1000)

nb_samples = 500


def show_dataset(X, Y):
    fig, ax = plt.subplots(1, 1, figsize=(30, 25))

    ax.grid()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    for i in range(nb_samples):
        if Y[i] == 0:
            ax.scatter(X[i, 0], X[i, 1], marker='o', color='r')
        else:
            ax.scatter(X[i, 0], X[i, 1], marker='^', color='b')

    plt.show()


if __name__ == '__main__':
    # Create dataset
    X, Y = make_classification(n_samples=nb_samples, n_features=2, n_informative=2, n_redundant=0,
                               n_clusters_per_class=1)

    # Show dataset
    show_dataset(X, Y)

    # Create a SVM with linear kernel
    svc = SVC(kernel='linear')

    # Compute CV score
    svc_scores = cross_val_score(svc, X, Y, scoring='accuracy', cv=10)
    print('Linear SVM CV average score: %.3f' % svc_scores.mean())

