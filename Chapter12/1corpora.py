from __future__ import print_function

from nltk.corpus import gutenberg

if __name__ == '__main__':
    # Print all Gutenberg corpus documents
    print('Gutenberg corpus files:')
    print(gutenberg.fileids())

    # Print a raw corpus
    print(gutenberg.raw('milton-paradise.txt'))

    # Print 2 sentences from a corpus
    print(gutenberg.sents('milton-paradise.txt')[0:2])

    # Print 20 words from a corpus
    print(gutenberg.words('milton-paradise.txt')[0:20])