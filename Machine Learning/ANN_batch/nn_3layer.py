# Acknowledgement: I refer to Teacher Maoying Wu(School of Life Science and Biotechnology)'s teaching material
#                  for this algorithm. I add one more hidden layer hoping that the model can perform better.
# Reference: https://github.com/ricket-sjtu/bi390/blob/master/lec9-deeplearning.ipynb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def train_model(X, y):
    D = X.shape[1]
    K = 2
    h = 100  # size of the first hidden layer
    h2 = 50  # size of the second hidden layer
    W = 0.01 * np.random.randn(D, h)
    b = np.zeros((1, h))
    W2 = 0.01 * np.random.randn(h, h2)
    b2 = np.zeros((1, h2))
    W3 = 0.01 * np.random.randn(h2, K)
    b3 = np.zeros((1, K))
    step_size = 1e-0
    reg = 1e-3  # regularization strength

    num_examples = X.shape[0]
    for i in range(10000):
        hidden_layer = np.maximum(0, np.dot(X, W) + b)  # ReLU activation
        hidden_layer2 = np.maximum(0, np.dot(hidden_layer, W2) + b2)
        scores = np.dot(hidden_layer2, W3) + b3
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        corect_logprobs = -np.log(probs[range(num_examples), y])
        data_loss = np.sum(corect_logprobs) / num_examples
        reg_loss = 0.5 * reg * np.sum(W * W) + 0.5 * reg * np.sum(W2 * W2) + 0.5 * reg * np.sum(W3 * W3)
        loss = data_loss + reg_loss
        if i % 1000 == 0:
            print("iteration %d: loss %f" % (i, loss))
        dscores = probs
        dscores[range(num_examples), y] -= 1
        dscores /= num_examples

        dW3 = np.dot(hidden_layer2.T, dscores)
        db3 = np.sum(dscores, axis=0, keepdims=True)

        dhidden2 = np.dot(dscores, W3.T)
        dhidden2[hidden_layer2 <= 0] = 0

        dW2 = np.dot(hidden_layer.T, dhidden2)
        db2 = np.sum(dhidden2, axis=0, keepdims=True)
        dhidden = np.dot(dhidden2, W2.T)
        dhidden[hidden_layer <= 0] = 0

        dW = np.dot(X.T, dhidden)
        db = np.sum(dhidden, axis=0, keepdims=True)

        dW3 += reg * W3
        dW2 += reg * W2
        dW += reg * W

        W += -step_size * dW
        b += -step_size * db
        W2 += -step_size * dW2
        b2 += -step_size * db2
        W3 += -step_size * dW3
        b3 += -step_size * db3
    return W, W2, W3, b, b2, b3


def evaluate(X, y, W, W2, W3, b, b2, b3):
    hidden_layer = np.maximum(0, np.dot(X, W) + b)
    hidden_layer2 = np.maximum(0, np.dot(hidden_layer, W2) + b2)
    scores = np.dot(hidden_layer2, W3) + b3
    predicted_class = np.argmax(scores, axis=1)
    print('training accuracy: %.2f' % (np.mean(predicted_class == y)))


if __name__ == '__main__':
    df = pd.read_excel("heart_failure.xlsx")
    X = df.iloc[:, :-1]  # the features
    y = df.iloc[:, -1]  # the labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    W, W2, W3, b, b2, b3 = train_model(X_train, y_train)
    evaluate(X_test, y_test, W, W2, W3, b, b2, b3)
