from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

from sklearn.datasets import make_circles
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


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
    # Create datasets
    X, Y = make_circles(n_samples=nb_samples, noise=0.1)

    # Show dataset
    show_dataset(X, Y)

    # Define a param grid
    param_grid = [
        {
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'C': [0.1, 0.2, 0.4, 0.5, 1.0, 1.5, 1.8, 2.0, 2.5, 3.0]
        }
    ]

    # Create a train grid search on SVM classifier
    gs = GridSearchCV(estimator=SVC(), param_grid=param_grid,
                      scoring='accuracy', cv=10, n_jobs=multiprocessing.cpu_count())
    gs.fit(X, Y)

    print(gs.best_estimator_)
    print('Kernel SVM score: %.3f' % gs.best_score_)



