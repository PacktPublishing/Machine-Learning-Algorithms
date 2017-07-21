from __future__ import print_function

import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# For reproducibility
np.random.seed(1000)

ret = RegexpTokenizer('[a-zA-Z0-9\']+')
sw = set(stopwords.words('english'))
ess = SnowballStemmer('english', ignore_stopwords=True)


def tokenizer(sentence):
    tokens = ret.tokenize(sentence)
    return [ess.stem(t) for t in tokens if t not in sw]


if __name__ == '__main__':
    # Create a corpus
    corpus = [
        'This is a simple test corpus',
        'A corpus is a set of text documents',
        'We want to analyze the corpus and the documents',
        'Documents can be automatically tokenized'
    ]

    # Create a count vectorizer
    print('Count vectorizer:')
    cv = CountVectorizer()

    vectorized_corpus = cv.fit_transform(corpus)
    print(vectorized_corpus.todense())

    print('CV Vocabulary:')
    print(cv.vocabulary_)

    # Perform an inverse transformation
    vector = [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1]
    print(cv.inverse_transform(vector))

    # Use a complete external tokenizer
    print('CV with external tokenizer:')
    cv = CountVectorizer(tokenizer=tokenizer)
    vectorized_corpus = cv.fit_transform(corpus)
    print(vectorized_corpus.todense())

    # Use an n-gram range equal to (1, 2)
    print('CV witn n-gram range (1, 2):')
    cv = CountVectorizer(tokenizer=tokenizer, ngram_range=(1, 2))
    vectorized_corpus = cv.fit_transform(corpus)
    print(vectorized_corpus.todense())

    print('N-gram range (1,2) vocabulary:')
    print(cv.vocabulary_)

    # Create a Tf-Idf vectorizer
    print('Tf-Idf vectorizer:')
    tfidfv = TfidfVectorizer()
    vectorized_corpus = tfidfv.fit_transform(corpus)
    print(vectorized_corpus.todense())

    print('Tf-Idf vocabulary:')
    print(tfidfv.vocabulary_)

    # Use n-gram range equal to (1, 2) and L2 normalization
    print('Tf-Idf witn n-gram range (1, 2) and L2 normalization:')
    tfidfv = TfidfVectorizer(tokenizer=tokenizer, ngram_range=(1, 2), norm='l2')
    vectorized_corpus = tfidfv.fit_transform(corpus)
    print(vectorized_corpus.todense())

