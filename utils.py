import re
import csv
import glob
import numpy as np
from columnar import columnar
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

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
    # TODO: Perhaps identify and remove URLs beforehand? But they might aid in identifying spam... Replace URLs with
    #     some keyword like URL?
    #     Can also make it so that it identifies the URL of the video itself (for timestamp) versus URL of other video
    preprocessed_texts = []
    for text in texts:
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
