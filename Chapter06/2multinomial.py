from __future__ import print_function

import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB


# For reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Prepare a dummy dataset
    data = [
        {'house': 100, 'street': 50, 'shop': 25, 'car': 100, 'tree': 20},
        {'house': 5, 'street': 5, 'shop': 0, 'car': 10, 'tree': 500, 'river': 1}
    ]

    # Create and train a dictionary vectorizer
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(data)
    Y = np.array([1, 0])

    # Create and train a Multinomial Naive Bayes classifier
    mnb = MultinomialNB()
    mnb.fit(X, Y)

    # Create dummy test data
    test_data = data = [
        {'house': 80, 'street': 20, 'shop': 15, 'car': 70, 'tree': 10, 'river': 1},
        {'house': 10, 'street': 5, 'shop': 1, 'car': 8, 'tree': 300, 'river': 0}
    ]

    Yp = mnb.predict(dv.fit_transform(test_data))
    print(Yp)
