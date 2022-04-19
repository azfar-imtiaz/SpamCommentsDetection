import re
import csv
import glob
from string import punctuation


def read_data_from_file(filename):
    data = []
    with open(filename, 'r') as rfile:
        reader = csv.DictReader(rfile, fieldnames=['COMMENT_ID', 'AUTHOR', 'DATE', 'CONTENT', 'CLASS'])
        for row in reader:
            data.append({
                'author': row['AUTHOR'],
                'date': row['DATE'],
                'comment': row['CONTENT'],
                'class': row['CLASS']
            })
    return data


def get_data_from_files(path_to_files):
    data = []
    path_to_files += '/*.csv'
    filenames = glob.glob(path_to_files)
    for filename in filenames:
        data.extend(read_data_from_file(filename))


def preprocess_texts(texts):
    preprocessed_texts = []
    for text in texts:
        text = ''.join([c for c in text if c not in punctuation])
        text = re.sub('\\s+', ' ', text)
        text = text.lower()
        preprocessed_texts.append(text)
    return preprocessed_texts
