from __future__ import print_function

import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import cross_val_score


# For reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    diabetes = load_diabetes()

    # Create a linear regressor and compute CV score
    lr = LinearRegression(normalize=True)
    lr_scores = cross_val_score(lr, diabetes.data, diabetes.target, cv=10)
    print('Linear regression CV score: %.6f' % lr_scores.mean())

    # Create a Ridge regressor and compute CV score
    rg = Ridge(0.005, normalize=True)
    rg_scores = cross_val_score(rg, diabetes.data, diabetes.target, cv=10)
    print('Ridge regression CV score: %.6f' % rg_scores.mean())

    # Create a Lasso regressor and compute CV score
    ls = Lasso(0.01, normalize=True)
    ls_scores = cross_val_score(ls, diabetes.data, diabetes.target, cv=10)
    print('Lasso regression CV score: %.6f' % ls_scores.mean())

    # Create ElasticNet regressor and compute CV score
    en = ElasticNet(alpha=0.001, l1_ratio=0.8, normalize=True)
    en_scores = cross_val_score(en, diabetes.data, diabetes.target, cv=10)
    print('ElasticNet regression CV score: %.6f' % en_scores.mean())

    # Find the optimal alpha value for Ridge regression
    rgcv = RidgeCV(alphas=(1.0, 0.1, 0.01, 0.001, 0.005, 0.0025, 0.001, 0.00025), normalize=True)
    rgcv.fit(diabetes.data, diabetes.target)
    print('Ridge optimal alpha: %.3f' % rgcv.alpha_)

    # Find the optimal alpha value for Lasso regression
    lscv = LassoCV(alphas=(1.0, 0.1, 0.01, 0.001, 0.005, 0.0025, 0.001, 0.00025), normalize=True)
    lscv.fit(diabetes.data, diabetes.target)
    print('Lasso optimal alpha: %.3f' % lscv.alpha_)

    # Find the optimal alpha and l1_ratio for Elastic Net
    encv = ElasticNetCV(alphas=(0.1, 0.01, 0.005, 0.0025, 0.001), l1_ratio=(0.1, 0.25, 0.5, 0.75, 0.8), normalize=True)
    encv.fit(diabetes.data, diabetes.target)
    print('ElasticNet optimal alpha: %.3f and L1 ratio: %.4f' % (encv.alpha_, encv.l1_ratio_))






