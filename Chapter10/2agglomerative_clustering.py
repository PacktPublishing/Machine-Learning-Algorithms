from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering

# For reproducibility
np.random.seed(1000)

nb_samples = 3000


def plot_clustered_dataset(X, Y):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    ax.grid()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    markers = ['o', 'd', '^', 'x', '1', '2', '3', 's']
    colors = ['r', 'b', 'g', 'c', 'm', 'k', 'y', '#cccfff']

    for i in range(nb_samples):
        ax.scatter(X[i, 0], X[i, 1], marker=markers[Y[i]], color=colors[Y[i]])

    plt.show()


if __name__ == '__main__':
    # Create the dataset
    X, _ = make_blobs(n_samples=nb_samples, n_features=2, centers=8, cluster_std=2.0)

    # Show the dataset
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    ax.grid()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    ax.scatter(X[:, 0], X[:, 1], marker='o', color='b')
    plt.show()

    # Complete linkage
    print('Complete linkage')
    ac = AgglomerativeClustering(n_clusters=8, linkage='complete')
    Y = ac.fit_predict(X)

    # Show the clustered dataset
    plot_clustered_dataset(X, Y)

    # Average linkage
    print('Average linkage')
    ac = AgglomerativeClustering(n_clusters=8, linkage='average')
    Y = ac.fit_predict(X)

    # Show the clustered dataset
    plot_clustered_dataset(X, Y)

    # Ward linkage
    print('Ward linkage')
    ac = AgglomerativeClustering(n_clusters=8)
    Y = ac.fit_predict(X)

    # Show the clustered dataset
    plot_clustered_dataset(X, Y)


