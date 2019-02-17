# ham = 0; spam = 1

import os, string, re
import pandas as pd
import numpy as np

from nltk.corpus import stopwords

base_dir = 'D:/Codes/DataSets/trec07p/'
csv_name = 'processed_emails.csv'

def explore_data(data_set):
    print(data_set.describe())
    print(data_set.groupby('ham_or_spam').describe())

def process_text(text):
    removed_punc =[char.lower() for char in text if char not in string.punctuation]
    removed_punc=''.join(removed_punc)
    return [word for word in removed_punc.split() if word.lower() not in stopwords.words('english')]

if __name__=="__main__":
    ham_or_spam_df = pd.read_csv(os.path.join(base_dir, csv_name), usecols=['ham_or_spam', 'text'])
    print(ham_or_spam_df['text'].head(5).apply(process_text))
    # print(stopwords.words('english'))