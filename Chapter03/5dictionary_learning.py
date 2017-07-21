from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.decomposition import DictionaryLearning

# For reproducibility
np.random.seed(1000)

if __name__ == '__main__':
    # Load MNIST digits
    digits = load_digits()

    # Perform a dictionary learning (and atom extraction) from the MNIST dataset
    dl = DictionaryLearning(n_components=36, fit_algorithm='lars', transform_algorithm='lasso_lars')
    X_dict = dl.fit_transform(digits.data)

    # Show the atoms that have been extracted
    fig, ax = plt.subplots(6, 6, figsize=(8, 8))

    samples = [dl.components_[x].reshape((8, 8)) for x in range(34)]

    for i in range(6):
        for j in range(6):
            ax[i, j].set_axis_off()
            ax[i, j].imshow(samples[(i * 5) + j], cmap='gray')

    plt.show()

