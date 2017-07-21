from __future__ import print_function

import numpy as np

from nltk.corpus import reuters, stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# For reproducibility
np.random.seed(1000)

ret = RegexpTokenizer('[a-zA-Z0-9\']+')
sw = set(stopwords.words('english'))
ess = SnowballStemmer('english', ignore_stopwords=True)


def tokenizer(sentence):
    tokens = ret.tokenize(sentence)
    return [ess.stem(t) for t in tokens if t not in sw]


if __name__ == '__main__':
    # Compose the corpus
    Xr = np.array(reuters.sents(categories=['rubber']))
    Xc = np.array(reuters.sents(categories=['cotton']))
    Xw = np.concatenate((Xr, Xc))
    X = []

    for document in Xw:
        X.append(' '.join(document).strip().lower())

    # Create the label vectors
    Yr = np.zeros(shape=Xr.shape)
    Yc = np.ones(shape=Xc.shape)
    Y = np.concatenate((Yr, Yc))

    # Vectorize
    tfidfv = TfidfVectorizer(tokenizer=tokenizer, ngram_range=(1, 2), norm='l2')
    Xv = tfidfv.fit_transform(X)

    # Prepare train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(Xv, Y, test_size=0.25)

    # Create and train a Random Forest classifier
    rf = RandomForestClassifier(n_estimators=25)
    rf.fit(X_train, Y_train)

    # Test classifier
    score = rf.score(X_test, Y_test)
    print('Score: %.3f' % score)

    test_newsline = [
        'Trading tobacco is reducing the amount of requests for cotton and this has a negative impact on our economy']
    yvt = tfidfv.transform(test_newsline)
    category = rf.predict(yvt)
    print('Predicted category: %d' % int(category[0]))

