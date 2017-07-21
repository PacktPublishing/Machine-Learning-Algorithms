from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


# For reproducibility
np.random.seed(1000)

nb_samples = 200


def show_dataset(X, Y):
    fig, ax = plt.subplots(1, 1, figsize=(30, 25))

    ax.scatter(X, Y)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid()

    plt.show()


if __name__ == '__main__':
    # Create dataset
    X = np.arange(-5, 5, 0.05)

    Y = X + 2
    Y += X**2 + np.random.uniform(-0.5, 0.5, size=nb_samples)

    # Show the dataset
    show_dataset(X, Y)

    # Split dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X.reshape(-1, 1), Y.reshape(-1, 1), test_size=0.25)

    lr = LinearRegression(normalize=True)
    lr.fit(X_train, Y_train)
    print('Linear regression score: %.3f' % lr.score(X_train, Y_train))

    # Create polynomial features
    pf = PolynomialFeatures(degree=2)
    X_train = pf.fit_transform(X_train)
    X_test = pf.fit_transform(X_test)

    lr.fit(X_train, Y_train)
    print('Second degree polynomial regression score: %.3f' % lr.score(X_train, Y_train))