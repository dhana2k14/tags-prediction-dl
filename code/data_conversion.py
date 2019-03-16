import pandas as pd
import nltk
from nltk.corpus import stopwords
import re

excel_book = pd.ExcelWriter('../data/ltc_work_order_samples.xlsx')
wo_xls = pd.read_excel(excel_book)
wo_xls.head()

# text cleaning
# Spell-Correct words 

word_list = pd.read_excel("../data/word_list.xlsx", sheet_name = 0, index_col = None)
word_dict = dict(zip(word_list['source_word'].str.strip(), word_list['target_word'].str.strip()))

def words_correction(document):
    for doc in document.strip().split():
        if doc in word_dict:
            document = re.sub(re.compile(r'\b%s\b' % doc, re.I), word_dict.get(doc), document)
        else:
            document
    return document
 
# data pre-processing

wo_xls['FreeText_updated'] = wo_xls['FreeText'].apply(lambda x: words_correction(x))
wo_xls.to_excel("../data/wo_xls_free_text_updated.xlsx", index = False)






           
            
        