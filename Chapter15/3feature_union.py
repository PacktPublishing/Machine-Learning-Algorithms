from __future__ import print_function

import numpy as np
import warnings

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# For reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    warnings.simplefilter("ignore")

    # Load the dataset
    digits = load_digits()

    # Create the steps for a feature union
    steps_fu = [
        ('pca', PCA(n_components=10)),
        ('kbest', SelectKBest(f_classif, k=5)),
    ]

    # Create the steps for the pipeline
    fu = FeatureUnion(steps_fu)
    scaler = StandardScaler()
    svc = SVC(kernel='rbf', C=5.0, gamma=0.05)

    pipeline_steps = [
        ('fu', fu),
        ('scaler', scaler),
        ('classifier', svc)
    ]

    pipeline = Pipeline(pipeline_steps)

    print('Cross-validation score:')
    print(cross_val_score(pipeline, digits.data, digits.target, cv=10).mean())