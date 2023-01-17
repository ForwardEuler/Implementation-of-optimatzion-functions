import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype
from sklearn import linear_model
from numpy import ndarray
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

import torch
import torch.optim as optim

__author__ = "Chuan Tian"
__sID__ = "s4510890"


def printf(s, *args):
    print(s % args, end='')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    pr = np.exp(x - np.max(x, axis=1, keepdims=True))
    pr /= np.sum(pr, axis=1, keepdims=True)
    return pr


class LogisticRegression:
    def __init__(self):
        self.onehot = OneHotEncoder()

    def fit(self, X, y, lr=0.1, momentum=0.1, niter=800):
        """
        Train a multiclass logistic regression model on the given training set.

        Parameters
        ----------
        X: training examples, represented as an input array of shape (n_sample,
           n_features).
        y: labels of training examples, represented as an array of shape
           (n_sample,) containing the classes for the input examples
        lr: learning rate for gradient descent
        niter: number of gradient descent updates
        momentum: the momentum constant (see assignment task sheet for an explanation)

        Returns
        -------
        self: fitted model
        """
        self.classes_ = np.unique(y)
        self.class2int = dict((c, i) for i, c in enumerate(self.classes_))
        # y = np.array([self.class2int[c] for c in y])
        n_features = X.shape[1]
        n_classes = len(self.classes_)
        m = X.shape[0]
        # X = X.T
        self.intercept_ = np.zeros(n_classes)
        # self.coef_ = np.zeros((n_classes, n_features))
        self.coef_ = np.zeros((n_features, n_classes))
        # Implement your gradient descent training code here; uncomment the code below to do "random training"
        self.intercept_ = np.random.randn(*self.intercept_.shape)
        self.coef_ = np.random.randn(*self.coef_.shape)
        y = y.reshape(-1, 1)
        self.onehot.fit(y)
        y_true = self.onehot.transform(y).toarray()
        losses = np.zeros(niter)
        w_last = self.coef_
        for epoch in range(niter):
            z = X @ self.coef_ + self.intercept_
            y_hat = softmax(z)
            # Calculate the gradient
            dw = (1 / m) * X.T @ (y_hat - y_true)
            db = (1 / m) * np.sum(y_hat - y_true)

            # BP
            self.coef_ = self.coef_ - lr * dw + momentum * (self.coef_ - w_last)
            self.intercept_ = self.intercept_ - lr * db
            w_last = self.coef_

            loss = np.mean(-np.sum(y_true * np.log(y_hat + 0.001), axis=1))
            losses[epoch] = loss
            # Print out loss.
            if epoch % 20 == 0:
                printf("Epoch %d ==> Loss = %f\n", epoch, loss)
        return self

    def predict_proba(self, X):
        """
        Predict the class distributions for given input examples.

        Parameters
        ----------
        X: input examples, represented as an input array of shape (n_sample,
           n_features).

        Returns
        -------
        y: predicted class distributions, represented as an array of shape (n_sample,
           n_classes)
        """
        z = X @ self.coef_ + self.intercept_
        y_hat = softmax(z)
        return y_hat

    def predict(self, X):
        """
        Predict the classes for given input examples.

        Parameters
        ----------
        X: input examples, represented as an input array of shape (n_sample,
           n_features).

        Returns
        -------
        y: predicted class labels, represented as an array of shape (n_sample,)
        """
        z = X @ self.coef_ + self.intercept_
        y_hat = softmax(z)
        y = self.onehot.inverse_transform(y_hat)
        return y


if __name__ == '__main__':
    X, y = fetch_covtype(return_X_y=True)
    std_scaler = StandardScaler()
    X = std_scaler.fit_transform(X)
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.3, random_state=42)

    # clf = linear_model.LogisticRegression()
    clf = LogisticRegression()
    clf.fit(X_tr, y_tr)
    c = clf.predict_proba(X_tr)
    print(accuracy_score(y_tr, clf.predict(X_tr)))
    print(accuracy_score(y_ts, clf.predict(X_ts)))
