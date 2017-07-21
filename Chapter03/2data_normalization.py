from __future__ import print_function

import numpy as np

from sklearn.preprocessing import Normalizer

# For reproducibility
np.random.seed(1000)

if __name__ == '__main__':
    # Create a dummy dataset
    data = np.array([1.0, 2.0])
    print(data)

    # Max normalization
    n_max = Normalizer(norm='max')
    nm = n_max.fit_transform(data.reshape(1, -1))
    print(nm)

    # L1 normalization
    n_l1 = Normalizer(norm='l1')
    nl1 = n_l1.fit_transform(data.reshape(1, -1))
    print(nl1)

    # L2 normalization
    n_l2 = Normalizer(norm='l2')
    nl2 = n_l2.fit_transform(data.reshape(1, -1))
    print(nl2)