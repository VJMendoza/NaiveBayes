import pandas as pd
import numpy as np
from os import path
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

base_dir = 'data/'
csv_gv_name = 'ham_or_spam_preprocessed.csv'


def report_model(model_name, label_test, model_predictions):
    print("Model: {}".format(model_name))
    print(classification_report(label_test, model_predictions))
    print(confusion_matrix(label_test, model_predictions))


# https: // stackabuse.com/the-naive-bayes-algorithm-in-python-with-scikit-learn/
def generate_bow(frame):
    count_vect = CountVectorizer()
    counts = count_vect.fit_transform(frame['text'])
    # TF-IDF code in case we need to
    # transformer = TfidfTransformer().fit(counts)
    # counts = transformer.transform(counts)
    return counts


def learn_evaluate_models(file_path):
    data_gv = pd.read_csv(file_path, sep=",")
    counts_gv = generate_bow(frame)
    bow_train, bow_test, label_train, label_test = train_test_split(
        counts_gv, data_gv['ham_or_spam'], test_size=0.2)
    """bow_rv_train and bow_rv_test will just be subsets of bow_train and
    bow_test, respectively. This subset approach makes sure that we have
    exactly the same entries in both the training and test sets"""
    # bow_rv_train =
    # bow_rv_test =
    # Classifier with general vocabulary
    cgv = MultinomialNB(alpha=0).fit(bow_train, label_train)
    cgv_predictions = cgv.predict(bow_test)

    # Classifier with Laplace smoothing on general vocabulary
    cgv_l = MultinomialNB().fit(bow_train, label_train)
    cgv_l_predictions = cgv_l.predict(bow_test)

    # Classifier with reduced vocabulary
    crv = MultinomialNB(alpha=0).fit(bow_rv_train, label_train)
    crv_predictions = crv.predict(bow_test)

    # Classifier with Laplace smoothing on reduced vocabulary
    crv_l = MultinomialNB().fit(bow_rv_train, label_train)
    crv_l_predictions = crv_l.predict(bow_test)

    report_model("Classifier Using General Vocabulary",
                 label_test, cgv_predictions)
    report_model("Classifier With Laplace Smoothing Using General Vocabulary",
                 label_test, cgv_l_predictions)
    report_model("Classifier Using Reduced Vocabulary",
                 label_test, crv_predictions)
    report_model("Classifier With Laplace Smoothing Using Reduced Vocabulary",
                 label_test, crv_l_predictions)


if __name__ == "__main__":
    learn_evaluate_models(path.join(base_dir, csv_gv_name))
