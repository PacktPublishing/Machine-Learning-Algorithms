from __future__ import print_function

import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, zero_one_loss, jaccard_similarity_score, confusion_matrix, \
    precision_score, recall_score, fbeta_score


# For reproducibility
np.random.seed(1000)

nb_samples = 500


if __name__ == '__main__':
    # Create dataset
    X, Y = make_classification(n_samples=nb_samples, n_features=2, n_informative=2, n_redundant=0,
                               n_clusters_per_class=1)

    # Split dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

    # Create and train logistic regressor
    lr = LogisticRegression()
    lr.fit(X_train, Y_train)

    print('Accuracy score: %.3f' % accuracy_score(Y_test, lr.predict(X_test)))
    print('Zero-one loss (normalized): %.3f' % zero_one_loss(Y_test, lr.predict(X_test)))
    print('Zero-one loss (unnormalized): %.3f' % zero_one_loss(Y_test, lr.predict(X_test), normalize=False))
    print('Jaccard similarity score: %.3f' % jaccard_similarity_score(Y_test, lr.predict(X_test)))

    # Compute confusion matrix
    cm = confusion_matrix(y_true=Y_test, y_pred=lr.predict(X_test))
    print('Confusion matrix:')
    print(cm)

    print('Precision score: %.3f' % precision_score(Y_test, lr.predict(X_test)))
    print('Recall score: %.3f' % recall_score(Y_test, lr.predict(X_test)))
    print('F-Beta score (1): %.3f' % fbeta_score(Y_test, lr.predict(X_test), beta=1))
    print('F-Beta score (0.75): %.3f' % fbeta_score(Y_test, lr.predict(X_test), beta=0.75))
    print('F-Beta score (1.25): %.3f' % fbeta_score(Y_test, lr.predict(X_test), beta=1.25))



