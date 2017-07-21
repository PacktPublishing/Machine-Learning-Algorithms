from __future__ import print_function

import numpy as np

from nltk.corpus import brown

from sklearn.feature_extraction.text import CountVectorizer


# For reproducibility
np.random.seed(1000)

rank = 2
alpha_1 = 1000.0
alpha_2 = 10.0

# Compose a corpus
sentences_1 = brown.sents(categories=['editorial'])[0:20]
sentences_2 = brown.sents(categories=['fiction'])[0:20]
corpus = []

for s in sentences_1 + sentences_2:
    corpus.append(' '.join(s))

# Vectorize the corpus
cv = CountVectorizer(strip_accents='unicode', stop_words='english')
Xc = np.array(cv.fit_transform(corpus).todense())

# Define the probability matrices
Ptd = np.random.uniform(0.0, 1.0, size=(len(corpus), rank))
Pwt = np.random.uniform(0.0, 1.0, size=(rank, len(cv.vocabulary_)))
Ptdw = np.zeros(shape=(len(cv.vocabulary_), len(corpus), rank))

# Normalize the probability matrices
for d in range(len(corpus)):
    nf = np.sum(Ptd[d, :])
    for t in range(rank):
        Ptd[d, t] /= nf

for t in range(rank):
    nf = np.sum(Pwt[t, :])
    for w in range(len(cv.vocabulary_)):
        Pwt[t, w] /= nf


def log_likelihood():
    value = 0.0

    for d in range(len(corpus)):
        for w in range(len(cv.vocabulary_)):
            real_topic_value = 0.0

            for t in range(rank):
                real_topic_value += Ptd[d, t] * Pwt[t, w]

            if real_topic_value > 0.0:
                value += Xc[d, w] * np.log(real_topic_value)

    return value


def expectation():
    global Ptd, Pwt, Ptdw

    for d in range(len(corpus)):
        for w in range(len(cv.vocabulary_)):
            nf = 0.0

            for t in range(rank):
                Ptdw[w, d, t] = Ptd[d, t] * Pwt[t, w]
                nf += Ptdw[w, d, t]

            Ptdw[w, d, :] = (Ptdw[w, d, :] / nf) if nf != 0.0 else 0.0


def maximization():
    global Ptd, Pwt, Ptdw

    for t in range(rank):
        nf = 0.0

        for d in range(len(corpus)):
            ps = 0.0

            for w in range(len(cv.vocabulary_)):
                ps += Xc[d, w] * Ptdw[w, d, t]

            Pwt[t, w] = ps
            nf += Pwt[t, w]

        Pwt[:, w] /= nf if nf != 0.0 else alpha_1

    for d in range(len(corpus)):
        for t in range(rank):
            ps = 0.0
            nf = 0.0

            for w in range(len(cv.vocabulary_)):
                ps += Xc[d, w] * Ptdw[w, d, t]
                nf += Xc[d, w]

            Ptd[d, t] = ps / (nf if nf != 0.0 else alpha_2)


if __name__ == '__main__':
    print('Initial Log-Likelihood: %f' % log_likelihood())

    for i in range(30):
        expectation()
        maximization()
        print('Step %d - Log-Likelihood: %f' % (i, log_likelihood()))

    # Show the top 5 words per topic
    Pwts = np.argsort(Pwt, axis=1)[::-1]

    for t in range(rank):
        print('\nTopic ' + str(t))
        for i in range(5):
            print(cv.get_feature_names()[Pwts[t, i]])