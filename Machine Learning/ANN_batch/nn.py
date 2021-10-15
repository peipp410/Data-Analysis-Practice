import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def train_model(X, y):
    # initialize parameters randomly
    D = X.shape[1]
    K = 2
    h = 100  # size of hidden layer
    W = 0.01 * np.random.randn(D, h)
    b = np.zeros((1, h))
    W2 = 0.01 * np.random.randn(h, K)
    b2 = np.zeros((1, K))
    # some hyperparameters
    step_size = 1e-0
    reg = 1e-3  # regularization strength

    # gradient descent loop
    num_examples = X.shape[0]
    for i in range(10000):
        # evaluate class scores, [N x K]
        hidden_layer = np.maximum(0, np.dot(X, W) + b)  # note, ReLU activation
        scores = np.dot(hidden_layer, W2) + b2
        # compute the class probabilities/
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # [N x K]

        # compute the loss: average cross-entropy loss and regularization
        corect_logprobs = -np.log(probs[range(num_examples), y])
        data_loss = np.sum(corect_logprobs) / num_examples
        reg_loss = 0.5 * reg * np.sum(W * W) + 0.5 * reg * np.sum(W2 * W2)
        loss = data_loss + reg_loss
        if i % 1000 == 0:
            print("iteration %d: loss %f" % (i, loss))
        # compute the gradient on scores
        dscores = probs
        dscores[range(num_examples), y] -= 1
        dscores /= num_examples

        # backpropate the gradient to the parameters
        # first backprop into parameters W2 and b2
        dW2 = np.dot(hidden_layer.T, dscores)
        db2 = np.sum(dscores, axis=0, keepdims=True)
        # next backprop into hidden layer
        dhidden = np.dot(dscores, W2.T)
        # backprop the ReLU non-linearity
        dhidden[hidden_layer <= 0] = 0
        # finally into W,b
        dW = np.dot(X.T, dhidden)
        db = np.sum(dhidden, axis=0, keepdims=True)

        # add regularization gradient contribution
        dW2 += reg * W2
        dW += reg * W

        # perform a parameter update
        W += -step_size * dW
        b += -step_size * db
        W2 += -step_size * dW2
        b2 += -step_size * db2
    return W, W2, b, b2


def evaluate(X, y, W, W2, b, b2):
    hidden_layer = np.maximum(0, np.dot(X, W) + b)
    print(hidden_layer.shape)
    scores = np.dot(hidden_layer, W2) + b2
    predicted_class = np.argmax(scores, axis=1)
    print('training accuracy: %.2f' % (np.mean(predicted_class == y)))


if __name__ == '__main__':
    df = pd.read_excel("bankloan.xls")
    X = df.iloc[:, :-1]  # the features
    y = df.iloc[:, -1]  # the labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    W, W2, b, b2 = train_model(X_train, y_train)
    evaluate(X_test, y_test, W, W2, b, b2)
