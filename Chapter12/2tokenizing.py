# coding=utf-8

from __future__ import print_function

from nltk.tokenize import sent_tokenize
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import RegexpTokenizer

if __name__ == '__main__':
    # Sentence tokenizing
    print('Generic text:')
    generic_text = 'Lorem ipsum dolor sit amet, amet minim temporibus in sit. Vel ne impedit consequat intellegebat.'
    print(sent_tokenize(generic_text))

    print('English text:')
    english_text = 'Where is the closest train station? I need to reach London'
    print(sent_tokenize(english_text, language='english'))

    print('Spanish text:')
    spanish_text = u'¿Dónde está la estación más cercana? Inmediatamente me tengo que ir a Barcelona.'
    for sentence in sent_tokenize(spanish_text, language='spanish'):
        print(sentence)

    # Word tokenizing
    # Create a Treebank word tokenizer
    tbwt = TreebankWordTokenizer()

    print('Simple text:')
    simple_text = 'This is a simple text.'
    print(tbwt.tokenize(simple_text))

    print('Complex text:')
    complex_text = 'This isn\'t a simple text'
    print(tbwt.tokenize(complex_text))

    # Create a Regexp tokenizer
    ret = RegexpTokenizer('[a-zA-Z0-9\'\.]+')
    print(ret.tokenize(complex_text))

    # Create a more restrictive Regexp tokenizer
    ret = RegexpTokenizer('[a-zA-Z\']+')

    complex_text = 'This isn\'t a simple text. Count 1, 2, 3 and then go!'
    print(ret.tokenize(complex_text))