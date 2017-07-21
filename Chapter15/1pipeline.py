from __future__ import print_function

import numpy as np

from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# For reproducibility
np.random.seed(1000)

nb_samples = 500


if __name__ == '__main__':
    # Create the dataset
    X, Y = make_classification(n_samples=nb_samples, n_informative=15, n_redundant=5, n_classes=2)

    # Create the steps for the pipeline
    pca = PCA(n_components=10)
    scaler = StandardScaler()
    svc = SVC(kernel='poly', gamma=3)

    steps = [
                ('pca', pca),
                ('scaler', scaler),
        ('classifier', svc)
    ]

    # Create the pipeline
    pipeline = Pipeline(steps)

    # Perform a grid search
    param_grid = {
        'pca__n_components': [5, 10, 12, 15, 18, 20],
        'classifier__kernel': ['rbf', 'poly'],
        'classifier__gamma': [0.05, 0.1, 0.2, 0.5],
        'classifier__degree': [2, 3, 5]
    }

    gs = GridSearchCV(pipeline, param_grid)
    gs.fit(X, Y)

    print('Best estimator:')
    print(gs.best_estimator_)

    print('Best score:')
    print(gs.best_score_)
