import pandas as pd
import nltk
from nltk.corpus import stopwords

wo_xls = pd.read_excel('../data/Sample_output_WO.xlsx', usecols = [0])
wo_xls.head()

# text cleaning


