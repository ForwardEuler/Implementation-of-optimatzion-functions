# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:10:09 2020

@author: Yang Liu
"""

import numpy as np
import math


def unpickle_csv(file):
    with open(file, 'rb') as fo:
        data = np.loadtxt(fo, delimiter=",", skiprows=0)
    return data


def loadData():
    raw_data = unpickle_csv("./spam/spambase.data")
    train_raw = raw_data[:, 0:-1]
    labels_raw = raw_data[:, -1]
    n = labels_raw.size
    test_size = math.ceil(n / 5)
    test_index = np.random.choice(n, test_size, replace=False)
    train_index = np.setdiff1d(np.arange(len(labels_raw)), test_index)
    A_train = train_raw[train_index, :]
    b_train = labels_raw[train_index].reshape(-1, 1)
    A_test = train_raw[test_index, :]
    b_test = labels_raw[test_index].reshape(-1, 1)
    b_test = np.append(b_test, 1 - b_test, axis=1)
    return A_train, b_train, A_test, b_test


def main():
    A_train, b_train, A_test, b_test = loadData()
    print(A_train.shape, A_train.dtype)
    print(b_train.shape, b_train.dtype)
    print(A_test.shape, A_test.dtype)
    print(b_test.shape, b_test.dtype)


if __name__ == '__main__':
    main()
