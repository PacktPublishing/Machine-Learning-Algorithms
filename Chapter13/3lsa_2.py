from __future__ import print_function

import numpy as np

from nltk.corpus import brown

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


# For reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Compose a corpus
    sentences = sentences = brown.sents(categories=['news', 'fiction'])
    corpus = []

    for s in sentences:
        corpus.append(' '.join(s))

    # Vectorize the corpus
    vectorizer = TfidfVectorizer(strip_accents='unicode', stop_words='english', sublinear_tf=True, use_idf=True)
    Xc = vectorizer.fit_transform(corpus)

    rank = 2

    # Performed a truncated SVD
    tsvd = TruncatedSVD(n_components=rank)
    Xt = tsvd.fit_transform(Xc)

    # Check the top-10 word per topic
    Mwts = np.argsort(tsvd.components_, axis=1)[::-1]

    for t in range(rank):
        print('\nTopic ' + str(t))
        for i in range(10):
            print(vectorizer.get_feature_names()[Mwts[t, i]])