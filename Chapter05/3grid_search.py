from __future__ import print_function

import numpy as np
import multiprocessing

from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression


# For reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Load dataset
    iris = load_iris()

    # Define a param grid
    param_grid = [
        {
            'penalty': ['l1', 'l2'],
            'C': [0.5, 1.0, 1.5, 1.8, 2.0, 2.5]
        }
    ]

    # Create and train a grid search
    gs = GridSearchCV(estimator=LogisticRegression(), param_grid=param_grid,
                      scoring='accuracy', cv=10, n_jobs=multiprocessing.cpu_count())
    gs.fit(iris.data, iris.target)

    # Best estimator
    print(gs.best_estimator_)

    gs_scores = cross_val_score(gs.best_estimator_, iris.data, iris.target, scoring='accuracy', cv=10)
    print('Best estimator CV average score: %.3f' % gs_scores.mean())

