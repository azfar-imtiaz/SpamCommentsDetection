from sklearn.model_selection import train_test_split

import globals
from utils import get_data_from_files, preprocess_texts


if __name__ == '__main__':
    comments_data = get_data_from_files(globals.PATH_TO_CSV_FILES)
    comments_data = preprocess_texts(comments_data)
    comments = [c['comment'] for c in comments_data]
    classes = [c['class'] for c in comments_data]

    num_spam_comments = classes.count(1)
    num_ham_comments = classes.count(0)

    print("Number of spam comments: {}/{}".format(num_spam_comments, len(classes)))
    print("Number of ham comments: {}/{}".format(num_ham_comments, len(classes)))

    X_train, X_test, y_train, y_test = train_test_split(comments, classes, test_size=0.2)
