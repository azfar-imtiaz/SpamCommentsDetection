from sklearn.model_selection import train_test_split

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
