from __future__ import print_function

import numpy as np

from sklearn.neighbors import NearestNeighbors

# For reproducibility
np.random.seed(1000)

nb_users = 1000
nb_product = 20

if __name__ == '__main__':
    # Create the user dataset
    users = np.zeros(shape=(nb_users, 4))

    for i in range(nb_users):
        users[i, 0] = np.random.randint(0, 4)
        users[i, 1] = np.random.randint(0, 2)
        users[i, 2] = np.random.randint(0, 5)
        users[i, 2] = np.random.randint(0, 5)

    # Create user-product dataset
    user_products = np.random.randint(0, nb_product, size=(nb_users, 5))

    # Fit k-nearest neighbors
    nn = NearestNeighbors(n_neighbors=20, radius=2.0)
    nn.fit(users)

    # Create a test user
    test_user = np.array([2, 0, 3, 2])

    # Determine the neighbors
    d, neighbors = nn.kneighbors(test_user.reshape(1, -1))

    print('Neighbors:')
    print(neighbors)

    # Determine the suggested products
    suggested_products = []

    for n in neighbors:
        for products in user_products[n]:
            for product in products:
                if product != 0 and product not in suggested_products:
                    suggested_products.append(product)

    print('Suggested products:')
    print(suggested_products)



