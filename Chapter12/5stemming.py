from __future__ import print_function

from nltk.stem.snowball import SnowballStemmer
from nltk.stem.snowball import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer

if __name__ == '__main__':
    print('English Snowball stemming:')
    ess = SnowballStemmer('english', ignore_stopwords=True)
    print(ess.stem('flies'))

    print('French Snowball stemming:')
    fss = SnowballStemmer('french', ignore_stopwords=True)
    print(fss.stem('courais'))

    print('English Snowball stemming:')
    print(ess.stem('teeth'))

    print('Porter stemming:')
    ps = PorterStemmer()
    print(ps.stem('teeth'))

    print('Lancaster stemming:')
    ls = LancasterStemmer()
    print(ls.stem('teeth'))

    print('Porter stemming:')
    print(ps.stem('teen'))
    print(ps.stem('teenager'))

    print('Lancaster stemming:')
    print(ls.stem('teen'))
    print(ls.stem('teenager'))