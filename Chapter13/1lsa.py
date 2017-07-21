from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import brown

from scipy.linalg import svd

from sklearn.feature_extraction.text import TfidfVectorizer


# For reproducibility
np.random.seed(1000)


def scatter_documents(X):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.scatter(X[:, 0], X[:, 1])
    ax.set_xlabel('t0')
    ax.set_ylabel('t1')
    ax.grid()
    plt.show()


if __name__ == '__main__':
    # Compose a corpus
    sentences = brown.sents(categories=['news'])[0:500]
    corpus = []

    for s in sentences:
        corpus.append(' '.join(s))

    # Vectorize the corpus
    vectorizer = TfidfVectorizer(strip_accents='unicode', stop_words='english', sublinear_tf=True, use_idf=True)
    Xc = vectorizer.fit_transform(corpus).todense()

    # Perform SVD
    U, s, V = svd(Xc, full_matrices=False)

    # Extract a sub-space with rank=2
    rank = 2

    Uk = U[:, 0:rank]
    sk = np.diag(s)[0:rank, 0:rank]
    Vk = V[0:rank, :]

    # Check the top-10 word per topic
    Mwts = np.argsort(np.abs(Vk), axis=1)[::-1]

    for t in range(rank):
        print('\nTopic ' + str(t))
        for i in range(10):
            print(vectorizer.get_feature_names()[Mwts[t, i]])

    # Compute the structure of a document
    print('\nSample document:')
    print(corpus[0])

    Mdtk = Uk.dot(sk)
    print('\nSample document in the topic sub-space:')
    print('d0 = %.2f*t1 + %.2f*t2' % (Mdtk[0][0], Mdtk[0][1]))

    # Show a scatter plot of all documents
    scatter_documents(Mdtk)
