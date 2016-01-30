#!/usr/bin/env python
# coding=UTF-8

from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from pprint import pprint

topic_category_urls = [
    'http://www.ck12.org/earth-science/',
    'http://www.ck12.org/life-science/',
    'http://www.ck12.org/physical-science/',
    'http://www.ck12.org/biology/',
    'http://www.ck12.org/chemistry/',
    'http://www.ck12.org/physics/',
]

topics = set()
for url in tqdm(topic_category_urls):
    html = requests.get(url).text
    for h3 in BeautifulSoup(html, 'html.parser').find_all('h3'):
        topic = ' '.join(map(str.strip, h3.li.a.get('href').strip('/').split('/')[-1].split('-')))
        topics.add(topic)

plucker = re.compile(r'''
    (.*?)
    (?:
      \ in\ Earth\ Sciences?
    | \ in\ Life\ Sciences?
    | \ in\ Physical\ Sciences?
    | \ in\ Biology
    | \ in\ Chemistry
    | \ in\ Physics
    )?
''', re.X | re.I)

topics = {plucker.fullmatch(t).group(1) for t in topics}

pages = pd.DataFrame({
    'topic': list(topics),    
})
pages.set_index('topic', inplace=True)
pages['is_loaded'] = False

pprint(pages)

pages.to_pickle('topics.pkl')

