from __future__ import print_function

import numpy as np
import multiprocessing

from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


# For reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Load dataset
    digits = load_digits()

    # Define a param grid
    param_grid = [
        {
            'criterion': ['gini', 'entropy'],
            'max_features': ['auto', 'log2', None],
            'min_samples_split': [2, 10, 25, 100, 200],
            'max_depth': [5, 10, 15, None]
        }
    ]

    # Create and train a grid searh
    gs = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param_grid,
                      scoring='accuracy', cv=10, n_jobs=multiprocessing.cpu_count())
    gs.fit(digits.data, digits.target)

    print(gs.best_estimator_)
    print('Decision tree score: %.3f' % gs.best_score_)