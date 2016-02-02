#!/usr/bin/env python
# coding=UTF-8

import pandas as pd
from tqdm import tqdm

from tokenize_text import tokenize_text

training_set = pd.read_csv('data/challenge/training_set.tsv', '\t')
training_set.set_index('id')

training_set['correctAnswerTag'] = training_set['correctAnswer']
del training_set['correctAnswer']

for id in tqdm(training_set.index, 'Tokenizing'):
    for thing in ('question', 'answerA', 'answerB', 'answerC', 'answerD'):
        text = training_set[thing][id]
        text = tokenize_text(text)
        training_set.set_value(id, thing, text)

pd.to_pickle(training_set, 'tokenized_challenge_training_set.pkl')

