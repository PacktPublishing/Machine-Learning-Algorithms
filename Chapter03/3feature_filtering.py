from __future__ import print_function

import numpy as np

from sklearn.datasets import load_boston, load_iris
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2, f_regression

# For reproducibility
np.random.seed(1000)

if __name__ == '__main__':
    # Load Boston data
    regr_data = load_boston()
    print('Boston data shape')
    print(regr_data.data.shape)

    # Select the best k features with regression test
    kb_regr = SelectKBest(f_regression)
    X_b = kb_regr.fit_transform(regr_data.data, regr_data.target)
    print('K-Best-filtered Boston dataset shape')
    print(X_b.shape)
    print('K-Best scores')
    print(kb_regr.scores_)

    # Load iris data
    class_data = load_iris()
    print('Iris dataset shape')
    print(class_data.data.shape)

    # Select the best k features using Chi^2 classification test
    perc_class = SelectPercentile(chi2, percentile=15)
    X_p = perc_class.fit_transform(class_data.data, class_data.target)
    print('Chi2-filtered Iris dataset shape')
    print(X_p.shape)
    print('Chi2 scores')
    print(perc_class.scores_)

