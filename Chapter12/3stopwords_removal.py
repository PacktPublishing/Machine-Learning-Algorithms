from __future__ import print_function

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

if __name__ == '__main__':
    # Load English stopwords
    sw = set(stopwords.words('english'))

    print('English stopwords:')
    print(sw)

    # Tokenize and remove stopwords
    complex_text = 'This isn\'t a simple text. Count 1, 2, 3 and then go!'

    ret = RegexpTokenizer('[a-zA-Z\']+')
    tokens = ret.tokenize(complex_text)
    clean_tokens = [t for t in tokens if t not in sw]

    print('Tokenized and cleaned complex text')
    print(clean_tokens)