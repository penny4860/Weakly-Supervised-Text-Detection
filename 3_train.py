#-*- coding: utf-8 -*-

from sklearn.svm import SVC
from src.utils import save_model
import numpy as np

def create_xy(pos_features, neg_features):
    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle
    
    pos_ys = np.ones((len(pos_features)))
    neg_ys = np.zeros((len(neg_features)))
    xs = np.concatenate([pos_features, neg_features], axis=0)
    ys = np.concatenate([pos_ys, neg_ys], axis=0)
    xs, ys = shuffle(xs, ys, random_state=0)

    X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size=0.1, random_state=0)
    return X_train, X_test, y_train, y_test

def test(clf, X, y):
    from sklearn.metrics import classification_report
    y_true, y_pred = y, clf.predict(X)
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":

    # 1. load features
    pos_features = np.load("positive_features.npy")
    neg_features = np.load("negative_features.npy")
    print(pos_features.shape, neg_features.shape)
 
    # 2. create (X, y)
    X_train, X_test, y_train, y_test = create_xy(pos_features, neg_features)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print(y_train[:10])

    clf = SVC(C=1.0, kernel='linear')
    clf.fit(X_train, y_train)
    
    test(clf, X_train, y_train)
    test(clf, X_test, y_test)

    save_model(clf, "cls.pkl")
