#!/usr/bin/env python
# coding=UTF-8

import requests
from pprint import pprint

response = requests.get('http://localhost:8080/LuceneServer/server/create_index', params={
    'pathtocorpus': '/home/yuri-kilochek/devel/kaggle_allen_qa/lucene/corpus',
    'byline': 'yes',
    'similarity': 'default',
    'analyzertype': 'standard',
    'minlengthtext': 0,
})

assert response.status_code == 200
pprint(response.json())

