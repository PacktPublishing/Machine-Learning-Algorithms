from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import VarianceThreshold

# For reproducibility
np.random.seed(1000)

if __name__ == '__main__':
    # Create a dummy dataset
    X = np.ndarray(shape=(100, 3))

    X[:, 0] = np.random.normal(0.0, 5.0, size=100)
    X[:, 1] = np.random.normal(0.5, 5.0, size=100)
    X[:, 2] = np.random.normal(1.0, 0.5, size=100)

    # Show the dataset
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.grid()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    ax.plot(X[:, 0], label='STD = 5.0')
    ax.plot(X[:, 1], label='STD = 5.0')
    ax.plot(X[:, 2], label='STD = 0.5')

    plt.legend()
    plt.show()

    # Impose a variance threshold
    print('Samples before variance thresholding')
    print(X[0:3, :])

    vt = VarianceThreshold(threshold=1.5)
    X_t = vt.fit_transform(X)

    # After the filter has removed the componenents
    print('Samples after variance thresholding')
    print(X_t[0:3, :])