import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

import globals
from utils import *


def get_predictions():
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    clf = joblib.load('comment_classifier.pkl')
    filename = "YouTube06-JohnCena.csv"
    data = read_data_from_file(filename)
    data = [a['comment'] for a in data[1:]]
    data = preprocess_texts(data, remove_stopwords=False)
    vectors = vectorizer.transform(data)
    predictions = clf.predict_proba(vectors)
    for index, elem in enumerate(data):
        if index == 0:
            continue
        max_prediction = np.max(predictions[index])
        index_max_prediction = np.argmax(predictions[index])
        if index_max_prediction == 1 or max_prediction < 0.9:
            print("{}: {}".format(elem, max_prediction))


if __name__ == '__main__':
    comments_data = get_data_from_files(globals.PATH_TO_CSV_FILES)

    comments = [c['comment'] for c in comments_data]
    classes = [c['class'] for c in comments_data]

    comments = preprocess_texts(comments)
    get_class_count(classes)
    # draw_histogram_word_counts(comments)

    X_train, X_test, y_train, y_test = train_test_split(comments, classes, test_size=0.2)

    # TODO: Create sklearn pipeline here
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    X_train_vectors = vectorizer.fit_transform(X_train)

    clf = RandomForestClassifier(n_estimators=200)
    clf.fit(X_train_vectors, y_train)

    X_test_vectors = vectorizer.transform(X_test)
    predictions = clf.predict(X_test_vectors)

    print(classification_report(y_test, predictions))
    print_incorrect_predictions(X_test, y_test, predictions)

    get_feature_importances(vectorizer=vectorizer, model=clf)

    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    joblib.dump(clf, "comment_classifier.pkl")

    # get_predictions()
