import re
import csv
import glob
import numpy as np
from columnar import columnar
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from urlextract import URLExtract

from globals import PUNCTUATION


def read_data_from_file(filename):
    data = []
    with open(filename, 'r') as rfile:
        reader = csv.DictReader(rfile, fieldnames=['COMMENT_ID', 'AUTHOR', 'DATE', 'CONTENT', 'CLASS'], )
        for index, row in enumerate(reader):
            if index == 0:
                continue
            data.append({
                'author': row['AUTHOR'],
                'date': row['DATE'],
                'comment': row['CONTENT'],
                'class': int(row['CLASS'])
            })
    return data


def get_data_from_files(path_to_files):
    data = []
    path_to_files += '/*.csv'
    filenames = glob.glob(path_to_files)
    for filename in filenames:
        data.extend(read_data_from_file(filename))
    return data


def preprocess_texts(texts, remove_stopwords=False):
    # TODO: Consider using lemmatization here?
    # TODO: Can identify the URL of the video itself (when scraping) versus URL of other video
    extractor = URLExtract()
    preprocessed_texts = []
    for text in texts:
        urls_in_text = extractor.find_urls(text)
        for url in urls_in_text:
            text = text.replace(url, 'URL')
        # try:
        #     text = re.sub(globals.URL_REGEX, 'URL', text)
        # except re.error:
        #     continue
        text = ''.join([c for c in text if c not in PUNCTUATION])
        text = re.sub('\\s+', ' ', text)
        text = text.lower()
        if remove_stopwords:
            text = ' '.join([w for w in text.split() if w not in stopwords.words('english')])
        preprocessed_texts.append(text)
    return preprocessed_texts


def get_class_count(classes):
    num_spam_comments = classes.count(1)
    num_ham_comments = classes.count(0)
    print("Number of spam comments: {}/{}".format(num_spam_comments, len(classes)))
    print("Number of ham comments: {}/{}".format(num_ham_comments, len(classes)))


def draw_histogram_word_counts(comments, num_bins=10):
    word_counts = []
    min_word_count = 500
    max_word_count = -1
    for comment in comments:
        count = len(comment.split())
        if count < min_word_count:
            min_word_count = count
        elif count > max_word_count:
            max_word_count = count
        word_counts.append(len(comment.split()))
    plt.hist(word_counts, bins=num_bins)
    plt.xlabel('Word counts')
    plt.ylabel('Frequency')
    plt.xticks(np.arange(min_word_count, max_word_count, num_bins))
    plt.xlim([min_word_count, 120])
    plt.show()


def print_incorrect_predictions(comments, y_test, predictions):
    header = ['True', 'Predicted', 'Comment']
    print_data = []
    for index in range(len(predictions)):
        if y_test[index] != predictions[index]:
            print_data.append([y_test[index], predictions[index], comments[index]])
    table = columnar(print_data, header)
    print(table)


def get_feature_importances(vectorizer, model):
    """
        We get feature importances here from the vectorizer and model.
        We observed better features using Tfidf instead of Count Vectorizer.
        The word 'subscribe' has the highest feature importance score, as we'd predicted.

    :param vectorizer: This is a text vectorizer
    :param model: This can be any model that has the feature_importances_ method
    :return: None
    """
    for feature, importance in sorted(
            zip(vectorizer.get_feature_names(), model.feature_importances_),
            key=lambda x: x[1],
            reverse=False):
        print("{}: {}".format(feature, importance))