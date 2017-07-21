from __future__ import print_function

import numpy as np

from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder
from sklearn.feature_extraction import DictVectorizer, FeatureHasher


# For reproducibility
np.random.seed(1000)

if __name__ == '__main__':
    Y = np.random.choice(('Male', 'Female'), size=(10))

    # Encode the labels
    print('Label encoding')
    le = LabelEncoder()
    yt = le.fit_transform(Y)
    print(yt)

    # Decode a dummy output
    print('Label decoding')
    output = [1, 0, 1, 1, 0, 0]
    decoded_output = [le.classes_[i] for i in output]
    print(decoded_output)

    # Binarize the labels
    print('Label binarization')
    lb = LabelBinarizer()
    yb = lb.fit_transform(Y)
    print(yb)

    # Decode the binarized labels
    print('Label decoding')
    lb.inverse_transform(yb)

    # Define some dictionary data
    data = [
        {'feature_1': 10, 'feature_2': 15},
        {'feature_1': -5, 'feature_3': 22},
        {'feature_3': -2, 'feature_4': 10}
    ]

    # Vectorize the dictionary data
    print('Dictionary data vectorization')
    dv = DictVectorizer()
    Y_dict = dv.fit_transform(data)
    print(Y_dict.todense())

    print('Vocabulary:')
    print(dv.vocabulary_)

    # Feature hashing
    print('Feature hashing')
    fh = FeatureHasher()
    Y_hashed = fh.fit_transform(data)

    # Decode the features
    print('Feature decoding')
    print(Y_hashed.todense())

    # One-hot encoding
    data1 = [
        [0, 10],
        [1, 11],
        [1, 8],
        [0, 12],
        [0, 15]
    ]

    # Encode data
    oh = OneHotEncoder(categorical_features=[0])
    Y_oh = oh.fit_transform(data1)
    print(Y_oh.todense())
