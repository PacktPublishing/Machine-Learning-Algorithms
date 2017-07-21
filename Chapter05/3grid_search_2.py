from __future__ import print_function

import numpy as np
import multiprocessing

from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import SGDClassifier


# For reproducibility
np.random.seed(1000)

if __name__ == '__main__':
    # Load dataset
    iris = load_iris()

    # Define a param grid
    param_grid = [
        {
            'penalty': ['l1', 'l2', 'elasticnet'],
            'alpha': [1e-5, 1e-4, 5e-4, 1e-3, 2.3e-3, 5e-3, 1e-2],
            'l1_ratio': [0.01, 0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 0.75, 0.8]
        }
    ]

    # Create SGD classifier
    sgd = SGDClassifier(loss='perceptron', learning_rate='optimal')

    # Create and train a grid search
    gs = GridSearchCV(estimator=sgd, param_grid=param_grid, scoring='accuracy', cv=10,
                      n_jobs=multiprocessing.cpu_count())
    gs.fit(iris.data, iris.target)

    # Best estimator
    print(gs.best_estimator_)

    gs_scores = cross_val_score(gs.best_estimator_, iris.data, iris.target, scoring='accuracy', cv=10)
    print('Best estimator CV average score: %.3f' % gs_scores.mean())