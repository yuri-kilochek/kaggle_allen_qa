#!/usr/bin/env python
# coding=UTF-8

import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from multiprocessing import cpu_count

texts = pd.read_pickle('tokenized_ck12wiki_texts.pkl')

doc2vec = Doc2Vec(
    [TaggedDocument(t, [i]) for i, t in enumerate(texts['text'])],
    workers=cpu_count(),
)

pd.to_pickle(doc2vec, 'doc2vec.pkl')

