from __future__ import print_function

import matplotlib.pyplot as plt
import multiprocessing
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.lancaster import LancasterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_curve, auc

# For reproducibility
np.random.seed(1000)

# Path to the dataset (http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/)
dataset = 'dataset.csv'

rt = RegexpTokenizer('[a-zA-Z0-9\.]+')
sw = set(stopwords.words('english'))
ls = LancasterStemmer()


def tokenizer(sentence):
    tokens = rt.tokenize(sentence)
    return [ls.stem(t.lower()) for t in tokens if t not in sw]


if __name__ == '__main__':
    # Load corpus and labels
    corpus = []
    labels = []

    with open(dataset, 'r') as df:
        for i, line in enumerate(df):
            if i == 0:
                continue

            parts = line.strip().split(',')
            labels.append(float(parts[1].strip()))
            corpus.append(parts[3].strip())

    # Vectorize the corpus (only 100000 records)
    tfv = TfidfVectorizer(tokenizer=tokenizer, sublinear_tf=True, ngram_range=(1, 2), norm='l2')
    X = tfv.fit_transform(corpus[0:100000])
    Y = np.array(labels[0:100000])

    # Prepare train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

    # Create and train a Random Forest
    rf = RandomForestClassifier(n_estimators=20, n_jobs=multiprocessing.cpu_count())
    rf.fit(X_train, Y_train)

    # Compute scores
    print('Precision: %.3f' % precision_score(Y_test, rf.predict(X_test)))
    print('Recall: %.3f' % recall_score(Y_test, rf.predict(X_test)))

    # Compute the ROC curve
    y_score = rf.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(Y_test, y_score[:, 1])

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='red', label='Random Forest (AUC: %.2f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()







