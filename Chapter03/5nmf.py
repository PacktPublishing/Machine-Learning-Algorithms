from __future__ import print_function

import numpy as np

from sklearn.datasets import load_iris
from sklearn.decomposition import NMF

# For reproducibility
np.random.seed(1000)

if __name__ == '__main__':
    # Load iris dataset
    iris = load_iris()
    print('Irid dataset shape')
    print(iris.data.shape)

    # Perform a non-negative matrix factorization
    nmf = NMF(n_components=3, init='random', l1_ratio=0.1)
    Xt = nmf.fit_transform(iris.data)

    print('Reconstruction error')
    print(nmf.reconstruction_err_)

    print('Original Iris sample')
    print(iris.data[0])

    print('Compressed Iris sample (via Non-Negative Matrix Factorization)')
    print(Xt[0])

    print('Rebuilt sample')
    print(nmf.inverse_transform(Xt[0]))