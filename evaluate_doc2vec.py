#!/usr/bin/env python
# coding=UTF-8

import pandas as pd
import numpy as np
from tqdm import tqdm
from pprint import pprint

qas = pd.read_csv('data/challenge/training_set.tsv', '\t')
qas.set_index('id')

from tokenize_text import tokenize_text

for id in tqdm(qas.index, 'Tokenizing'):
    for thing in ('question', 'answerA', 'answerB', 'answerC', 'answerD'):
        text = qas[thing][id]
        text = tokenize_text(text)
        qas.set_value(id, thing, text)

doc2vec = pd.read_pickle('doc2vec.pkl')

for id in tqdm(qas.index, 'Vectorizing'):
    for thing in ('question', 'answerA', 'answerB', 'answerC', 'answerD'):
        text = qas[thing][id]
        vector = doc2vec.infer_vector(text)
        qas.set_value(id, thing, vector)

from scipy.spatial import distance

total_count = 0
correct_count = 0
for id in tqdm(qas.index, 'Evaluating'):
    question = qas['question'][id]
    answers = [qas[a][id] for a in ('answerA', 'answerB', 'answerC', 'answerD')]
    correct_answer_index = np.argmin([distance.cosine(a, question) for a in answers])
    correct_answer_tag = chr(ord('A') + correct_answer_index)

    total_count+= 1
    if qas['correctAnswer'][id] == correct_answer_tag:
        correct_count+= 1

print('{:.2%}'.format(correct_count / total_count))

