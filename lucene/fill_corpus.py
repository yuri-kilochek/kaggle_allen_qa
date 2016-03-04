#!/usr/bin/env python
# coding=UTF-8

import pandas as pd
import re
from tqdm import tqdm

texts = pd.read_pickle('../tokenized_ck12wiki_texts.pkl')

def clean_name(name):
    name = name.lower()
    name = re.sub('[^a-z0-9]+', '_', name)
    return name

for name, text in tqdm(zip(texts.index, texts['text'])):
    name = clean_name(name)
    text = ' '.join(text)
    with open('corpus/{}.txt'.format(name), 'w', encoding='UTF-8') as text_file:
        text_file.write(text)

