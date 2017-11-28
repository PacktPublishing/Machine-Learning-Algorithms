from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering


# For reproducibility
np.random.seed(1000)

nb_samples = 1000


if __name__ == '__main__':
    # Create dataset
    X, Y = make_moons(n_samples=nb_samples, noise=0.05)

    # Try different gammas with a RBF affinity
    Yss = []
    gammas = np.linspace(0, 12, 4)

    for gamma in gammas:
        sc = SpectralClustering(n_clusters=2, affinity='rbf', gamma=gamma)
        Yss.append(sc.fit_predict(X))

    # Show data
    fig, ax = plt.subplots(1, 4, figsize=(30, 10), sharey=True)

    for x in range(4):
        ax[x].grid()
        ax[x].set_title('Gamma = %.0f' % gammas[x])

        for i in range(nb_samples):
            c = Yss[x][i]

            if c == 0:
                ax[x].scatter(X[i, 0], X[i, 1], marker='o', color='r')
            else:
                ax[x].scatter(X[i, 0], X[i, 1], marker='^', color='b')

    plt.show()