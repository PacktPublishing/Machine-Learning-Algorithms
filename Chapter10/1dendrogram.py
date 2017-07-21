from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

# For reproducibility
np.random.seed(1000)

nb_samples = 25

if __name__ == '__main__':
    # Create the dataset
    X, Y = make_blobs(n_samples=nb_samples, n_features=2, centers=3, cluster_std=1.5)

    # Show the dataset
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    ax.grid()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    ax.scatter(X[:, 0], X[:, 1], marker='o', color='b')
    plt.show()

    # Compute the distance matrix
    Xdist = pdist(X, metric='euclidean')

    # Compute the linkage
    Xl = linkage(Xdist, method='ward')

    # Compute and show the dendrogram
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    Xd = dendrogram(Xl)
    plt.show()