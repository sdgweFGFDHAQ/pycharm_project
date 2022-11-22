import numpy
from keras.models import Sequential
import nltk
import sys
import jieba
import pandas
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score
