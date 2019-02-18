import pandas as pd
import numpy as np
from os import path
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

base_dir = 'dataset'
csv_gv_name = 'vocab.csv'
# csv_rv_name = 'vocab.csv'


def report_model(model_name, label_test, model_predictions):
    print("Model: {}".format(model_name))
    print(classification_report(label_test, model_predictions))
    print(confusion_matrix(label_test, model_predictions))


if __name__ == "__main__":
    data_gv = pd.read_csv(path.join(base_dir, csv_gv_name), sep=",")
    # data_rv = pd.read_csv(path.join(base_dir, csv_rv_name), sep=",")
    bow_train, bow_test, label_train, label_test = train_test_split(data_gv.iloc[:, 1:],
                                                                    data_gv['ham_or_spam'],
                                                                    test_size=0.2)
    # bow_rv_train, bow_rv_test, label_rv_train, label_rv_test = train_test_split(data_rv.iloc[:, 1:],
    #                                            data_rv['ham_or_spam'],
    #                                            test_size=0.2)
    cgv = MultinomialNB(alpha=0).fit(bow_train, label_train)
    cgv_predictions = cgv.predict(bow_test)
    cgv_l = MultinomialNB().fit(bow_train, label_train)
    cgv_l_predictions = cgv_l.predict(bow_test)
    # crv = MultinomialNB(alpha=0).fit(bow_rv_train, label_rv_train)
    # cgv_predictions = cgv.predict(bow_rv_test)
    # crv = MultinomialNB().fit(bow_rv_train, label_rv_train)
    # cgv_predictions = cgv.predict(bow_rv_test)
    report_model("Classifier Using General Vocabulary",
                 label_test, cgv_predictions)
    report_model("Classifier With Laplace Smoothing Using General Vocabulary",
                 label_test, cgv_predictions)
    # report_model("Classifier Using Reduced Vocabulary", label_test, cgv_predictions)
    # report_model("Classifier With Laplace Smoothing Using Reduced Vocabulary", label_test, cgv_predictions)
