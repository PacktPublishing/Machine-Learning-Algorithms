from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets.samples_generator import make_blobs
from sklearn.decomposition import KernelPCA

# For reproducibility
np.random.seed(1000)

if __name__ == '__main__':
    # Create a dummy dataset
    Xb, Yb = make_blobs(n_samples=500, centers=3, n_features=3)

    # Show the dataset
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(Xb[:, 0], Xb[:, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid()

    plt.show()

    # Perform a kernel PCA (with radial basis function)
    kpca = KernelPCA(n_components=2, kernel='rbf', fit_inverse_transform=True)
    X_kpca = kpca.fit_transform(Xb)

    # Plot the dataset after PCA
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(kpca.X_transformed_fit_[:, 0], kpca.X_transformed_fit_[:, 1])
    ax.set_xlabel('First component: Variance')
    ax.set_ylabel('Second component: Mean')
    ax.grid()

    plt.show()