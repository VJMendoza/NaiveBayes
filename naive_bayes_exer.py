# ham = 0; spam = 1

import os, string, re
import pandas as pd
import numpy as np

from nltk.corpus import stopwords, words
from bs4 import BeautifulSoup

base_dir = 'D:/Codes/DataSets/trec07p/'
csv_name = 'processed_emails.csv'

def remove_non_ascii(text):
    return ''.join(i for i in text if ord(i)<128)

def clean_mail_body(mail_body):
    mail_body = re.sub(r'\\[Uu][a-zA-Z0-9]{4}', '', mail_body)
    soup = BeautifulSoup(mail_body)
    for script in soup(['script', 'style']):
        script.extract()
    
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk)
    
    return text.replace('\t', '')

def tokenize_text(text):
    removed_punc =[char.lower() for char in text if char not in string.punctuation]
    removed_punc=''.join(removed_punc)
    return [word for word in removed_punc.split() if word.lower() not in stopwords.words('english')]

def get_words(frame):
    word_freq_df = pd.DataFrame(columns=['ham_or_spam'])
    for index, row in frame.iterrows():
        tokens = tokenize_text(row['text'])
        for token in tokens:
            if token not in word_freq_df.columns:
                word_freq_df[token] = ''
        
        curr_words = {feature:tokens.count(feature) for feature in list(word_freq_df)}
        curr_words['ham_or_spam'] = row['ham_or_spam']
        word_freq_df = word_freq_df.append(curr_words, ignore_index=True)
    
    return word_freq_df.replace('', 0)

if __name__=="__main__":
    ham_or_spam_df = pd.read_csv(os.path.join(base_dir, csv_name), usecols=['ham_or_spam', 'text'])
    ham_or_spam_df['text'] = ham_or_spam_df['text'].astype(str)

    # Remove any remaining html tags
    # ham_or_spam_df['text'] = [BeautifulSoup(text, features='html.parser').get_text() for text in ham_or_spam_df['text']]
    ham_or_spam_df['text'] = ham_or_spam_df['text'].apply(remove_non_ascii)
    ham_or_spam_df['text'] = ham_or_spam_df['text'].apply(clean_mail_body)
    ham_or_spam_df['text'].replace('', np.nan, inplace=True)
    # print(ham_or_spam_df.isna().sum())

    # Remove empty rows
    ham_or_spam_df.dropna(how='any', subset=['text'], inplace=True)

    # print(get_words(ham_or_spam_df.head(10)).columns)
    get_words(ham_or_spam_df.head(10)).to_csv(os.path.join(base_dir, 'vocab.csv'))
    # Get words
    # print(ham_or_spam_df.loc[125]['text'])
    # print(ham_or_spam_df.loc[125]['text'].decode('utf8').encode('ascii', 'ignore'))