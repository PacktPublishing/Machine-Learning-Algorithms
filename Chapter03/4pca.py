from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

# For reproducibility
np.random.seed(1000)

if __name__ == '__main__':
    # Load MNIST digits
    digits = load_digits()

    # Show some random digits
    selection = np.random.randint(0, 1797, size=100)

    fig, ax = plt.subplots(10, 10, figsize=(10, 10))

    samples = [digits.data[x].reshape((8, 8)) for x in selection]

    for i in range(10):
        for j in range(10):
            ax[i, j].set_axis_off()
            ax[i, j].imshow(samples[(i * 8) + j], cmap='gray')

    plt.show()

    # Perform a PCA on the digits dataset
    pca = PCA(n_components=36, whiten=True)
    X_pca = pca.fit_transform(digits.data / 255)

    print('Explained variance ratio')
    print(pca.explained_variance_ratio_)

    # Plot the explained variance ratio
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    ax[0].set_xlabel('Component')
    ax[0].set_ylabel('Variance ratio (%)')
    ax[0].bar(np.arange(36), pca.explained_variance_ratio_ * 100.0)

    ax[1].set_xlabel('Component')
    ax[1].set_ylabel('Cumulative variance (%)')
    ax[1].bar(np.arange(36), np.cumsum(pca.explained_variance_)[::-1])

    plt.show()

    # Rebuild from PCA and show the result
    fig, ax = plt.subplots(10, 10, figsize=(10, 10))

    samples = [pca.inverse_transform(X_pca[x]).reshape((8, 8)) for x in selection]

    for i in range(10):
        for j in range(10):
            ax[i, j].set_axis_off()
            ax[i, j].imshow(samples[(i * 8) + j], cmap='gray')

    plt.show()

