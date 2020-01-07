# *_*coding:utf-8 *_*
#
from sklearn.feature_extraction.text import CountVectorizer

text = ["The quick brown fox jumped over the lazy dog."]

vectorizer = CountVectorizer()

vectorizer.fit(text)

print(vectorizer)

print(vectorizer.vocabulary_)
