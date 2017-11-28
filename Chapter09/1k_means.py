from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


# For reproducibility
np.random.seed(1000)

nb_samples = 1000


def show_dataset(X):
    fig, ax = plt.subplots(1, 1, figsize=(30, 25))

    ax.grid()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    ax.scatter(X[:, 0], X[:, 1], marker='o', color='b')

    plt.show()


def show_clustered_dataset(X, km):
    fig, ax = plt.subplots(1, 1, figsize=(30, 25))

    ax.grid()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    for i in range(nb_samples):
        c = km.predict(X[i].reshape(1, -1))
        if c == 0:
            ax.scatter(X[i, 0], X[i, 1], marker='o', color='r')
        elif c == 1:
            ax.scatter(X[i, 0], X[i, 1], marker='^', color='b')
        else:
            ax.scatter(X[i, 0], X[i, 1], marker='d', color='g')

    plt.show()


if __name__ == '__main__':
    # Create dataset
    X, _ = make_blobs(n_samples=nb_samples, n_features=2, centers=3, cluster_std=1.5)

    # Show dataset
    show_dataset(X)

    # Create and train K-Means
    km = KMeans(n_clusters=3)
    km.fit(X)

    # Show the centroids
    print(km.cluster_centers_)

    # Show clustered dataset
    show_clustered_dataset(X, km)
    