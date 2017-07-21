from __future__ import print_function

import numpy as np

from nltk.corpus import brown

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# For reproducibility
np.random.seed(1000)

if __name__ == '__main__':
    # Compose a corpus
    sentences_1 = brown.sents(categories=['reviews'])[0:1000]
    sentences_2 = brown.sents(categories=['government'])[0:1000]
    sentences_3 = brown.sents(categories=['fiction'])[0:1000]
    sentences_4 = brown.sents(categories=['news'])[0:1000]
    corpus = []

    for s in sentences_1 + sentences_2 + sentences_3 + sentences_4:
        corpus.append(' '.join(s))

    # Vectorize the corpus
    cv = CountVectorizer(strip_accents='unicode', stop_words='english', analyzer='word')
    Xc = cv.fit_transform(corpus)

    # Perform LDA
    lda = LatentDirichletAllocation(n_topics=8, learning_method='online', max_iter=25)
    Xl = lda.fit_transform(Xc)

    # Show the top 5 words per topic
    Mwts_lda = np.argsort(lda.components_, axis=1)[::-1]

    for t in range(8):
        print('\nTopic ' + str(t))
        for i in range(5):
            print(cv.get_feature_names()[Mwts_lda[t, i]])

    # Test the model with new document
    print('Document 0:')
    print(corpus[0])
    print(Xl[0])

    print('Document 2500:')
    print(corpus[2500])
    print(Xl[2500])

    test_doc = corpus[0] + ' ' + corpus[2500]
    y_test = lda.transform(cv.transform([test_doc]))
    print(y_test)

