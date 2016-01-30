#!/usr/bin/env python
# coding=UTF-8

import pandas as pd
from tqdm import tqdm
from pprint import pprint

from tokenize_text import tokenize_text

texts = pd.read_pickle('data/ck12wiki/texts.pkl')

for title in tqdm(texts.index, 'Tokenizing texts'):
    text = texts['text'][title]
    text = tokenize_text(text)
    texts.set_value(title, 'text', text)

pprint(texts)

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from multiprocessing import cpu_count

doc2vec = Doc2Vec(
    [TaggedDocument(t, [i]) for i, t in enumerate(texts['text'])],
    workers=cpu_count(),
)

pd.to_pickle(doc2vec, 'doc2vec.pkl')

