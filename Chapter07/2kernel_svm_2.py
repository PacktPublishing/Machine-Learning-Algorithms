from __future__ import print_function

import numpy as np
import multiprocessing

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


# For reproducibility
np.random.seed(1000)

# Set a local folder here
olivetti_home = ''


if __name__ == '__main__':
    # Load dataset

    faces = fetch_olivetti_faces(data_home=olivetti_home)
    # Define a param grid
    param_grid = [
        {
            'kernel': ['rbf', 'poly'],
            'C': [0.1, 0.5, 1.0, 1.5],
            'degree': [2, 3, 4, 5],
            'gamma': [0.001, 0.01, 0.1, 0.5]
        }
    ]

    # Create a train grid search on SVM classifier
    gs = GridSearchCV(estimator=SVC(), param_grid=param_grid,
                      scoring='accuracy', cv=10, n_jobs=multiprocessing.cpu_count())
    gs.fit(faces.data, faces.target)

    print(gs.best_estimator_)
    print('Kernel SVM score: %.3f' % gs.best_score_)
