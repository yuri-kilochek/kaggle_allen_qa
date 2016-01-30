# coding=UTF-8

import re

from nltk.corpus import stopwords
stopwords = frozenset(stopwords.words('english'))

from nltk.stem.lancaster import LancasterStemmer as Stemmer
stemmer = Stemmer()

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

        text[i] = stemmer.stem(word)
        i += 1

    return text

