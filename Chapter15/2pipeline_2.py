from __future__ import print_function

import numpy as np
import warnings

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# For reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    warnings.simplefilter("ignore")

    # Load the dataset
    digits = load_digits()

    # Create the steps for the pipeline
    pca = PCA()
    nmf = NMF()
    scaler = StandardScaler()
    kbest = SelectKBest(f_classif)
    lr = LogisticRegression()
    svc = SVC()

    pipeline_steps = [
        ('dimensionality_reduction', pca),
        ('normalization', scaler),
        ('classification', lr)
    ]

    # Create the pipeline
    pipeline = Pipeline(pipeline_steps)

    # Perform a grid search
    pca_nmf_components = [10, 20, 30]

    param_grid = [
        {
            'dimensionality_reduction': [pca],
            'dimensionality_reduction__n_components': pca_nmf_components,
            'classification': [lr],
            'classification__C': [1, 5, 10, 20]
        },
        {
            'dimensionality_reduction': [pca],
            'dimensionality_reduction__n_components': pca_nmf_components,
            'classification': [svc],
            'classification__kernel': ['rbf', 'poly'],
            'classification__gamma': [0.05, 0.1, 0.2, 0.5, 1.0],
            'classification__degree': [2, 3, 5],
            'classification__C': [1, 5, 10, 20]
        },
        {
            'dimensionality_reduction': [nmf],
            'dimensionality_reduction__n_components': pca_nmf_components,
            'classification': [lr],
            'classification__C': [1, 5, 10, 20]
        },
        {
            'dimensionality_reduction': [nmf],
            'dimensionality_reduction__n_components': pca_nmf_components,
            'classification': [svc],
            'classification__kernel': ['rbf', 'poly'],
            'classification__gamma': [0.05, 0.1, 0.2, 0.5, 1.0],
            'classification__degree': [2, 3, 5],
            'classification__C': [1, 5, 10, 20]
        },
        {
            'dimensionality_reduction': [kbest],
            'classification': [svc],
            'classification__kernel': ['rbf', 'poly'],
            'classification__gamma': [0.05, 0.1, 0.2, 0.5, 1.0],
            'classification__degree': [2, 3, 5],
            'classification__C': [1, 5, 10, 20]
        },
    ]

    gs = GridSearchCV(pipeline, param_grid)
    gs.fit(digits.data, digits.target)

    print('Best estimator:')
    print(gs.best_estimator_)

    print('Best score:')
    print(gs.best_score_)
