#!/usr/bin/env python
# coding=UTF-8

import pandas as pd
from tqdm import tqdm

from tokenize_text import tokenize_text

texts = pd.read_pickle('data/ck12wiki/texts.pkl')

for title in tqdm(texts.index, 'Tokenizing'):
    text = texts['text'][title]
    text = tokenize_text(text)
    texts.set_value(title, 'text', text)

pd.to_pickle(texts, 'tokenized_ck12wiki_texts.pkl')

