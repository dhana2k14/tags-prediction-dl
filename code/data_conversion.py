import pandas as pd
import nltk
from nltk.corpus import stopwords
import re

wo_xls = pd.read_excel('./data/ltc_work_order_samples.xlsx', usecols = [1])
wo_xls.head()

# text cleaning
# Spell-Correct words 

words_to_replace = {'ani' :'any', 
                    'concret' : 'concrete', 
                    'charg' : 'charge', 
                    'inclus' :'includes',
                    'materi' : 'material',
                    'specifi': 'specified',
                    'centr' : 'centre',
                    'aggreg' : 'aggregate',
                    'posit' :'position',
                    'provid' : 'providing',
                    'lay' : 'laying',
                    'exclud' : 'excluding',
                    'center' : 'centering',
                    'shutter' : 'shuttering',
                    'nomin' :'nominal'}

def words_correction(document):
    for doc in document.strip().split():
        if doc in words_to_replace:
            document = re.sub(re.compile(r'\b%s\b' % doc, re.I), words_to_replace.get(doc), document)
        else:
            document
    return document
            
            
        