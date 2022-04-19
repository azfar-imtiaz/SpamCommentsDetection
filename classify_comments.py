from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

import globals
from utils import *


if __name__ == '__main__':
    comments_data = get_data_from_files(globals.PATH_TO_CSV_FILES)

    comments = [c['comment'] for c in comments_data]
    classes = [c['class'] for c in comments_data]

    comments = preprocess_texts(comments)
    get_class_count(classes)
    draw_histogram_word_counts(comments)

    X_train, X_test, y_train, y_test = train_test_split(comments, classes, test_size=0.2)

    # TODO: Create sklearn pipeline here
    vectorizer = CountVectorizer(ngram_range=(1, 3))
    X_train_vectors = vectorizer.fit_transform(X_train)

    clf = RandomForestClassifier(n_estimators=200)
    clf.fit(X_train_vectors, y_train)

    X_test_vectors = vectorizer.transform(X_test)
    predictions = clf.predict(X_test_vectors)

    print(classification_report(y_test, predictions))
    print_incorrect_predictions(X_test, y_test, predictions)

    # TODO: Get word feature importances. The word 'subscribe' should have a very high value...
