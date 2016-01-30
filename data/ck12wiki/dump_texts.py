#!/usr/bin/env python
# coding=UTF-8

from warnings import filterwarnings
filterwarnings('ignore', module='bs4')

import pandas as pd
from tqdm import tqdm
import wikipedia as wiki
import sys
from pprint import pprint

topics = pd.read_pickle('topics.pkl')

try:
    texts = pd.read_pickle('texts.pkl')
    pprint(texts)
except FileNotFoundError:
    texts = pd.DataFrame(columns=['title', 'text'])
    texts.set_index('title')

until_save = 10
bad_titles = set()
for topic in tqdm(topics.index):
    if topics.loc[topic]['is_loaded']:
        continue

    while True:
        try:
            count = 0
            for title in wiki.search(topic):
                if count == 3:
                    break

                if title in bad_titles:
                    continue

                try:
                    page = wiki.page(title)
                    texts.loc[title] = page.content
                    count += 1
                except wiki.exceptions.DisambiguationError as e:
                    bad_titles.add(title)

            topics.loc[topic]['is_loaded'] = True
            if until_save == 0:
                texts.to_pickle('texts.pkl')
                topics.to_pickle('topics.pkl')
                pprint(texts)
                until_save = 10
            else:
                until_save -= 1

            break
        except wiki.exceptions.HTTPTimeoutError:
            continue
        except Exception as e:
            print(e, file=sys.stderr)
            break

texts.to_pickle('texts.pkl')
topics.to_pickle('topics.pkl')
pprint(texts)
