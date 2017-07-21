from __future__ import print_function

import numpy as np
import multiprocessing

from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


# For reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Load dataset
    digits = load_digits()

    # Define a param grid
    param_grid = [
        {
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'C': [0.1, 0.2, 0.4, 0.5, 1.0, 1.5, 1.8, 2.0, 2.5, 3.0]
        }
    ]

    # Create a train grid search on SVM classifier
    gs = GridSearchCV(estimator=SVC(), param_grid=param_grid,
                      scoring='accuracy', cv=10, n_jobs=multiprocessing.cpu_count())
    gs.fit(digits.data, digits.target)

    print(gs.best_estimator_)
    print('Kernel SVM score: %.3f' % gs.best_score_)