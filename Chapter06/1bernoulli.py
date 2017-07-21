from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import BernoulliNB


# For reproducibility
np.random.seed(1000)

nb_samples = 300


def show_dataset(X, Y):
    fig, ax = plt.subplots(1, 1, figsize=(30, 25))

    ax.grid()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    for i in range(nb_samples):
        if Y[i] == 0:
            ax.scatter(X[i, 0], X[i, 1], marker='o', color='r')
        else:
            ax.scatter(X[i, 0], X[i, 1], marker='^', color='b')

    plt.show()


if __name__ == '__main__':
    # Create dataset
    X, Y = make_classification(n_samples=nb_samples, n_features=2, n_informative=2, n_redundant=0)

    # Show dataset
    show_dataset(X, Y)

    # Split dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

    # Create and train Bernoulli Naive Bayes classifier
    bnb = BernoulliNB(binarize=0.0)
    bnb.fit(X_train, Y_train)

    print('Bernoulli Naive Bayes score: %.3f' % bnb.score(X_test, Y_test))

    # Compute CV score
    bnb_scores = cross_val_score(bnb, X, Y, scoring='accuracy', cv=10)
    print('Bernoulli Naive Bayes CV average score: %.3f' % bnb_scores.mean())

    # Predict some values
    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Yp = bnb.predict(data)
    print(Yp)

