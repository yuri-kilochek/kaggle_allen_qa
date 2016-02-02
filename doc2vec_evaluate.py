#!/usr/bin/env python
# coding=UTF-8

import pandas as pd
import numpy as np
from tqdm import tqdm
from pprint import pprint

training_set = pd.read_pickle('tokenized_challenge_training_set.pkl')
doc2vec = pd.read_pickle('doc2vec.pkl')

for id in tqdm(training_set.index, 'Vectorizing'):
    for thing in ('question', 'answerA', 'answerB', 'answerC', 'answerD'):
        text = training_set[thing][id]
        vector = doc2vec.infer_vector(text)
        training_set.set_value(id, thing, vector)

from scipy.spatial import distance

total_count = 0
correct_count = 0
for id in tqdm(training_set.index, 'Evaluating'):
    question = training_set['question'][id]
    answers = [training_set[a][id] for a in ('answerA', 'answerB', 'answerC', 'answerD')]
    correct_answer_index = np.argmin([distance.cosine(a, question) for a in answers])
    correct_answer_tag = chr(ord('A') + correct_answer_index)

    total_count+= 1
    if training_set['correctAnswerTag'][id] == correct_answer_tag:
        correct_count+= 1

print('{:.2%}'.format(correct_count / total_count))

