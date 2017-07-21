from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc


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

    # Create and train Gaussian Naive Bayes classifier
    gnb = GaussianNB()
    gnb.fit(X_train, Y_train)

    # Create and train a Logistic regressor (for comparison)
    lr = LogisticRegression()
    lr.fit(X_train, Y_train)

    # Compute ROC Curve
    Y_gnb_score = gnb.predict_proba(X_test)
    Y_lr_score = lr.decision_function(X_test)

    fpr_gnb, tpr_gnb, thresholds_gnb = roc_curve(Y_test, Y_gnb_score[:, 1])
    fpr_lr, tpr_lr, thresholds_lr = roc_curve(Y_test, Y_lr_score)

    # Plot ROC Curve
    plt.figure(figsize=(30, 25))

    plt.plot(fpr_gnb, tpr_gnb, color='red', label='Naive Bayes (AUC: %.2f)' % auc(fpr_gnb, tpr_gnb))
    plt.plot(fpr_lr, tpr_lr, color='green', label='Logistic Regression (AUC: %.2f)' % auc(fpr_lr, tpr_lr))
    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    plt.show()
