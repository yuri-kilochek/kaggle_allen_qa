#!/usr/bin/env python
# coding=UTF-8

import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LsiModel

texts = pd.read_pickle('tokenized_ck12wiki_texts.pkl')

lexicon = Dictionary(texts['text'])
lsi = LsiModel(
    corpus=(lexicon.doc2bow(t) for t in texts['text']),
    id2word=lexicon,
)

pd.to_pickle(lsi, 'lsi.pkl')

