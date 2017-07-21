from __future__ import print_function

from langdetect import detect, detect_langs

if __name__ == '__main__':
    # Simple language detection
    print(detect('This is English'))
    print(detect('Dies ist Deutsch'))

    # Probabilistic language detection
    print(detect_langs('I really love you mon doux amour!'))