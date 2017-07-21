from __future__ import print_function

import numpy as np

from sklearn.neighbors import NearestNeighbors

# For reproducibility
np.random.seed(1000)

nb_items = 1000

if __name__ == '__main__':
    # Create the item dataset
    items = np.zeros(shape=(nb_items, 4))

    for i in range(nb_items):
        items[i, 0] = np.random.randint(0, 100)
        items[i, 1] = np.random.randint(0, 100)
        items[i, 2] = np.random.randint(0, 100)
        items[i, 3] = np.random.randint(0, 100)

    metrics = ['euclidean', 'hamming', 'jaccard']

    for metric in metrics:
        print('Metric: %r' % metric)

        # Fit k-nearest neighbors
        nn = NearestNeighbors(n_neighbors=10, radius=5.0, metric=metric)
        nn.fit(items)

        # Create a test product
        test_product = np.array([15, 60, 28, 73])

        # Determine the neighbors with different radiuses
        d, suggestions = nn.radius_neighbors(test_product.reshape(1, -1), radius=20)

        print('Suggestions (radius=10):')
        print(suggestions)

        d, suggestions = nn.radius_neighbors(test_product.reshape(1, -1), radius=30)

        print('Suggestions (radius=15):')
        print(suggestions)