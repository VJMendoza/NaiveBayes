# ham = 0; spam = 1

import os
import string
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from nltk.corpus import stopwords, words
from bs4 import BeautifulSoup

base_dir = 'data/'
csv_name = 'processed_emails.csv'


def remove_non_ascii(text):
    return ''.join(i for i in text if ord(i) < 128)


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


# Could've just used nltk.tokenize but whatever lol
def tokenize_text(text):
    removed_punc = [char.lower()
                    for char in text if char not in string.punctuation]
    removed_punc = ''.join(removed_punc)
    return [word for word in removed_punc.split() if word.lower()
            not in stopwords.words('english')]


if __name__ == "__main__":
    ham_or_spam_df = pd.read_csv(os.path.join(
        base_dir, csv_name), usecols=['ham_or_spam', 'text'])
    ham_or_spam_df['text'] = ham_or_spam_df['text'].astype(str)

    # Remove any remaining html tags
    # ham_or_spam_df['text'] = [BeautifulSoup(text, features='html.parser').get_text() for text in ham_or_spam_df['text']]
    ham_or_spam_df['text'] = ham_or_spam_df['text'].apply(remove_non_ascii)
    ham_or_spam_df['text'] = ham_or_spam_df['text'].apply(clean_mail_body)
    ham_or_spam_df['text'].replace('', np.nan, inplace=True)

    # Remove empty rows
    ham_or_spam_df.dropna(how='any', subset=['text'], inplace=True)

    # Tokenize and form into one space-separated string
    ham_or_spam_df['text'] = ham_or_spam_df['text'].apply(tokenize_text)
    ham_or_spam_df['text'] = ham_or_spam_df['text'].apply(
        lambda x: ' '.join(x))

    ham_or_spam_df.to_csv(os.path.join(
        base_dir, 'ham_or_spam_preprocessed.csv'))
