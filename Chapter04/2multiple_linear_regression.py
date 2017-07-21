from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score


# For reproducibility
np.random.seed(1000)


def show_dataset(data):
    fig, ax = plt.subplots(4, 3, figsize=(20, 15))

    for i in range(4):
        for j in range(3):
            ax[i, j].plot(data.data[:, i + (j + 1) * 3])
            ax[i, j].grid()

    plt.show()


if __name__ == '__main__':
    # Load dataset
    boston = load_boston()

    # Show dataset
    show_dataset(boston)

    # Create a linear regressor instance
    lr = LinearRegression(normalize=True)

    # Split dataset
    X_train, X_test, Y_train, Y_test = train_test_split(boston.data, boston.target, test_size=0.1)

    # Train the model
    lr.fit(X_train, Y_train)

    print('Score %.3f' % lr.score(X_test, Y_test))

    # CV score
    scores = cross_val_score(lr, boston.data, boston.target, cv=7, scoring='neg_mean_squared_error')
    print('CV Negative mean squared errors mean: %.3f' % scores.mean())
    print('CV Negative mean squared errors std: %.3f' % scores.std())

    # CV R2 score
    r2_scores = cross_val_score(lr, boston.data, boston.target, cv=10, scoring='r2')
    print('CV R2 score: %.3f' % r2_scores.mean())



