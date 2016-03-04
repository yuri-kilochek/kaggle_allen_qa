#!/usr/bin/env python
# coding=UTF-8

import requests
import pandas as pd
from tqdm import tqdm
from pprint import pprint

def get_query_score(query):
    response = requests.get('http://localhost:8080/LuceneServer/server/get_score', params={
        'query': query,
        'nresult': 1,
        'pathtocorpus': '/home/yuri-kilochek/devel/kaggle_allen_qa/lucene/corpus',
        'returndocs': 'no',
        'analyzertype': 'standard',
        'similarity': 'default',
    })

    assert response.status_code == 200

    scores = response.json()['scoresList']
    if not scores:
        return 0.0
    return scores[0]

training_set = pd.read_pickle('../tokenized_challenge_training_set.pkl')

count, correct_count = 0, 0
for id in tqdm(training_set.index, 'Evaluating'):
    question = training_set['question'][id]
    answers = {
        'A': training_set['answerA'][id],
        'B': training_set['answerB'][id],
        'C': training_set['answerC'][id],
        'D': training_set['answerD'][id],
    }
    
    answerTag = max(answers, key=lambda at: get_query_score(' '.join(question + answers[at])))
    if answerTag == training_set['correctAnswerTag'][id]:
        correct_count += 1
    count += 1
    
print('{:.2%}:'.format(correct_count / count))

