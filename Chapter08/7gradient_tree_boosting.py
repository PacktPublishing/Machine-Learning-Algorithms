from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

# For reproducibility
np.random.seed(1000)

nb_samples = 500

if __name__ == '__main__':
    # Create the dataset
    X, Y = make_classification(n_samples=nb_samples, n_features=4, n_informative=3, n_redundant=1, n_classes=3)

    # Collect the scores for n_estimators in (1, 50)
    a = []
    max_estimators = 50

    for i in range(1, max_estimators):
        score = cross_val_score(GradientBoostingClassifier(n_estimators=i, learning_rate=10.0 / float(i)), X, Y,
                                     cv=10, scoring='accuracy').mean()
        a.append(score)

    # Plot the results
    plt.figure(figsize=(30, 25))
    plt.xlabel('Number of estimators')
    plt.ylabel('Average CV accuracy')
    plt.grid(True)
    plt.plot(a)
    plt.show()