from __future__ import print_function

import numpy as np

from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score


# For reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Load dataset
    iris = load_iris()

    # Create and train an AdaBoost classifier
    ada = AdaBoostClassifier(n_estimators=100, learning_rate=1.0)
    ada_scores = cross_val_score(ada, iris.data, iris.target, scoring='accuracy', cv=10)
    print('AdaBoost score: %.3f' % ada_scores.mean())

