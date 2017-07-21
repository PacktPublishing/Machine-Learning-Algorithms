from __future__ import print_function

import numpy as np
import warnings

from scikits.crab.models import MatrixPreferenceDataModel
from scikits.crab.similarities import UserSimilarity
from scikits.crab.metrics import euclidean_distances
from scikits.crab.recommenders.knn import UserBasedRecommender

# For reproducibility
np.random.seed(1000)

if __name__ == '__main__':
    # Define a user-item matrix
    user_item_matrix = {
        1: {1: 2, 2: 5, 3: 3},
        2: {1: 5, 4: 2},
        3: {2: 3, 4: 5, 3: 2},
        4: {3: 5, 5: 1},
        5: {1: 3, 2: 3, 4: 1, 5: 3}
    }

    # Build a matrix preference model
    model = MatrixPreferenceDataModel(user_item_matrix)

    # Build a similarity matrix
    similarity_matrix = UserSimilarity(model, euclidean_distances)

    # Create a recommender
    recommender = UserBasedRecommender(model, similarity_matrix, with_preference=True)

    # Test the recommender for user 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print(recommender.recommend(2))
