# coding=UTF-8

import re

from nltk.corpus import stopwords as nltk_stopwords
nltk_stopwords = frozenset(nltk_stopwords.words('english'))
from gensim.parsing import STOPWORDS as gensim_stopwords
stopwords = nltk_stopwords | gensim_stopwords

from nltk.stem.lancaster import LancasterStemmer as Stemmer
stemmer = Stemmer()
def stem(word):
    word = stemmer.stem(word)
    return word

def tokenize_text(text):
    text = text.lower()
    text = re.sub(r'''
        (?P<tag>==+)
        \ +
        ( references 
        | notes
        | notes\ +and\ +references
        | see\ +also
        | further\ +reading
        | bibliography
        | sources
        | footnotes
        | literature
        | external\ +links )
        \ +
        (?P=tag)
        .*
        $
    ''', '', text, flags=re.VERBOSE | re.DOTALL)
    text = re.sub(r'[^a-z]+', ' ', text)
    text = text.split()

    i = 0
    while i < len(text):
        word = text[i]
        if word in stopwords or len(word) <= 2:
            del text[i]
            continue

        text[i] = stem(word)
        i += 1

    return text

