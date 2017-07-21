from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score


# For reproducibility
np.random.seed(1000)

nb_classifications = 100


if __name__ == '__main__':
    # Load dataset
    digits = load_digits()

    # Collect accuracies
    et_accuracy = []

    for i in range(1, nb_classifications):
        a = cross_val_score(ExtraTreesClassifier(n_estimators=i), digits.data, digits.target, scoring='accuracy',
                            cv=10).mean()
        et_accuracy.append(a)

    # Show results
    plt.figure(figsize=(30, 25))
    plt.xlabel('Number of trees')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.plot(et_accuracy)
    plt.show()
