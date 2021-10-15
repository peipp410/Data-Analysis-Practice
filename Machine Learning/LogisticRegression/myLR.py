# Reference: https://www.bioinfo-scrounger.com/archives/750/
import numpy as np
import pandas as pd
import random


def split(X, y, ratio=0.3):
    index = np.random.permutation(X.shape[0])
    test_size = int(X.shape[0]*ratio)
    test_index = index[:test_size]
    train_index = index[test_size:]
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    return X_train, y_train, X_test, y_test


def sigmoid(z):
    if z >= 0:
        return 1.0/(1+np.exp(-z))
    else:
        return np.exp(z)/(1+np.exp(z))


def gradient_descent_sto(X, y, iter=1000, eta=0.001):
    beta = np.zeros(X.shape[1])
    samplenum = X.shape[0]
    for i in range(iter):
        a = 0
        while a == 0:
            s = random.sample(range(0, samplenum-1), 1)[0]
            xx = X[s]
            xi = np.zeros(X.shape[1])
            for j in range(X.shape[1]):
                xi[j] = xx[0, j]
            yi = y[s]
            yi = yi[0, 0]
            if predict(xi, beta) != yi:
                dot = np.dot(xi, beta)
                beta += eta*(yi*xi - sigmoid(dot)*xi)
                a = 1
    return beta


def gradient_descent_batch(X, y, iter=1000, eta=0.001):
    beta = np.ones(X.shape[1])
    for i in range(iter):
        h = np.dot(X, beta)
        pred = np.ones((h.shape[1], 1))
        for j in range(h.shape[1]):
            pred[j, 0] = sigmoid(h[0, j])
        error = pred-y
        dot = np.dot(X.T, error)
        tmp = np.ones(dot.shape[0])
        for j in range(dot.shape[0]):
            tmp[j] = dot[j, 0]
        beta -= eta * tmp
    return beta


def predict(data, beta):
    prob = sigmoid(np.dot(data, beta))
    if prob >= 0.5:
        return 1
    else:
        return 0


def main():
    filename = 'heart_failure.csv'
    df = np.mat(pd.read_csv(filename))
    X = df[:, :-1]  # the features
    y = df[:, -1]  # the labels
    X = np.insert(X, X.shape[1], values=1, axis=1)
    X_train, y_train, X_test, y_test = split(X, y, 0.3)
    beta = gradient_descent_sto(X_train, y_train)
    error = 0
    for i in range(X_test.shape[0]):
        if predict(X_test[i], beta) != y_test[i]:
            error += 1
    print("stochastic gradient descent:")
    print(error)
    print(1-error/X_test.shape[0])
    beta = gradient_descent_batch(X_train, y_train)
    error = 0
    for i in range(X_test.shape[0]):
        if predict(X_test[i], beta) != y_test[i]:
            error += 1
    print("batch gradient descent:")
    print(error)
    print(1-error/X_test.shape[0])


if __name__ == '__main__':
    main()
