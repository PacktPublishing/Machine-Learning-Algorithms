from __future__ import print_function

import numpy as np

from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import cross_val_score


# For reproducibility
np.random.seed(1000)

nb_samples = 500

# Set a folder to store the graph in
graph_folder = ''


if __name__ == '__main__':
    # Create dataset
    X, Y = make_classification(n_samples=nb_samples, n_features=3, n_informative=3, n_redundant=0, n_classes=3,
                               n_clusters_per_class=1)

    # Create a Decision tree classifier
    dt = DecisionTreeClassifier()
    dt_scores = cross_val_score(dt, X, Y, scoring='accuracy', cv=10)
    print('Decision tree score: %.3f' % dt_scores.mean())

    # Save in Graphviz format
    dt.fit(X, Y)

    with open('dt.dot', 'w') as df:
        df = export_graphviz(dt, out_file=df,
                             feature_names=['A', 'B', 'C'],
                             class_names=['C1', 'C2', 'C3'])
