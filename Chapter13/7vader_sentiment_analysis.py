from __future__ import print_function

from nltk.sentiment.vader import SentimentIntensityAnalyzer

if __name__ == '__main__':
    text = 'This is a very interesting and quite powerful sentiment analyzer'

    vader = SentimentIntensityAnalyzer()
    print(vader.polarity_scores(text))