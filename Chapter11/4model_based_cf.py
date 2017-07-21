from __future__ import print_function

import numpy as np

from scipy.linalg import svd

# For reproducibility
np.random.seed(1000)

if __name__ == '__main__':
    # Create a dummy user-item matrix
    M = np.random.randint(0, 6, size=(20, 10))

    print('User-Item matrix:')
    print(M)

    # Decompose M
    U, s, V = svd(M, full_matrices=True)
    S = np.diag(s)

    print('U -> %r' % str(U.shape))
    print('S -> %r' % str(S.shape))
    print('V -> %r' % str(V.shape))

    # Select the first 8 singular values
    Uk = U[:, 0:8]
    Sk = S[0:8, 0:8]
    Vk = V[0:8, :]

    # Compute the user and product vectors
    Su = Uk.dot(np.sqrt(Sk).T)
    Si = np.sqrt(Sk).dot(Vk).T

    # Compute the average rating per user
    Er = np.mean(M, axis=1)

    # Perform a prediction for the user 5 and item 2
    r5_2 = Er[5] + Su[5].dot(Si[2])
    print(r5_2)

