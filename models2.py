from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from time import gmtime, strftime, time
import os
import numpy as np
import pandas as pd

base_dir = 'data/'
csv_gv_name = 'ham_or_spam_preprocessed.csv'


def learn(bow_train, y_train):
    gv_model = Pipeline([
        ('countvect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB(alpha=0))
    ])
    gv_model.fit(bow_train, y_train)

    gvl_model = Pipeline([
        ('countvect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB())
    ])
    gvl_model.fit(bow_train, y_train)

    rv_model = Pipeline([
        ('countvect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('reduce', SelectKBest(chi2, k=200)),
        ('clf', MultinomialNB(alpha=0))
    ])
    rv_model.fit(bow_train, y_train)

    rvl_model = Pipeline([
        ('countvect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('reduce', SelectKBest(chi2, k=200)),
        ('clf', MultinomialNB())
    ])
    rvl_model.fit(bow_train, y_train)

    return gv_model, gvl_model, rv_model, rvl_model


def evaluate(model, model_name, bow_test, label_test, label_names):
    print(model_name)
    predicted = model.predict(bow_test)
    print('Accuracy: {:.05%}'.format(np.mean(predicted == label_test)))
    print(classification_report(label_test, predicted, target_names=label_names))


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(base_dir, csv_gv_name), header=0, index_col=0)
    df.dropna(how='any', subset=['text'], inplace=True)
    target_names = ['ham', 'spam']

    bow_train, bow_test, label_train, label_test = train_test_split(
        df['text'], df['ham_or_spam'], test_size=0.2)

    gv_model, gvl_model, rv_model, rvl_model = learn(bow_train, label_train)

    evaluate(gv_model, 'Gen Vocab (No laplace smoothing)',
             bow_test, label_test, target_names)
    evaluate(gvl_model, 'Gen Vocab (With laplace smoothing; alpha=1)',
             bow_test, label_test, target_names)
    evaluate(rv_model, 'Redux Vocab (No laplace smoothing)',
             bow_test, label_test, target_names)
    evaluate(rvl_model, 'Redux Vocab (With laplace smoothing; alpha=1)',
             bow_test, label_test, target_names)