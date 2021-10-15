import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split


def t1(X, y):
    train_accu = []
    test_accu = []
    training_ratio = [0.05, 0.15, 0.5, 0.75, 0.85]
    for i in training_ratio:
        test_tmp = []
        train_tmp = []
        for j in range(100):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-i)
            LR = lr(max_iter=1000)  # create the logistic regression model
            LR.fit(X_train, y_train)  # learn the parameters
            train_score = accuracy_score(y_train, LR.predict(X_train))
            test_accuracy = LR.score(X_test, y_test)
            test_tmp.append(test_accuracy)
            train_tmp.append(train_score)
        test_accu.append(test_tmp)
        train_accu.append(train_tmp)
    test_avg = []
    train_avg = []
    test_std_dev = []
    train_std_dev = []
    for i in range(5):
        test_avg.append(np.mean(test_accu[i]))
        test_std_dev.append(np.std(test_accu[i]))
        train_avg.append(np.mean(train_accu[i]))
        train_std_dev.append(np.std(train_accu[i]))
    print("The average accuracy on training data:")
    print(train_avg)
    print("The standard deviation of accuracy on training data:")
    print(train_std_dev)
    print("The average accuracy on test data:")
    print(test_avg)
    print("The standard deviation of accuracy on test data:")
    print(test_std_dev)
    plt.plot(training_ratio, train_avg, marker="x", label="training")
    plt.plot(training_ratio, test_avg, marker="+", label="test")
    plt.xlim([0.0, 1.0])
    plt.xlabel('training set size')
    plt.ylabel('average accuracy')
    plt.title('training set size vs avg.')
    plt.legend()
    plt.savefig("avg.png")
    plt.show()
    plt.clf()
    plt.plot(training_ratio, train_std_dev, marker="x", label="training")
    plt.plot(training_ratio, test_std_dev, marker="+", label="test")
    plt.xlim([0.0, 1.0])
    plt.xlabel('training set size')
    plt.ylabel('standard deviation')
    plt.title('training set size vs std. dev.')
    plt.legend()
    plt.savefig("std_dev.png")
    plt.show()
    plt.clf()


def t2(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
    LR = lr(max_iter=1000)  # create the logistic regression model
    LR.fit(X_train, y_train)  # learn the parameters
    probs_y = LR.predict_proba(X_test)
    precision, recall, thresholds = precision_recall_curve(y_test, probs_y[:, 1])
    plt.plot(thresholds, precision[: -1], label="Precision")
    plt.plot(thresholds, recall[: -1], label="Recall")
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Thresholds")
    plt.ylabel("Rate")
    plt.title("Recall and precision with different thresholds")
    plt.legend()
    plt.savefig("recall_precision.png")
    plt.show()
    plt.clf()
    plt.plot(recall[: -1], precision[: -1], label="pr")
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR curve")
    plt.legend()
    plt.savefig("pr_curve.png")
    plt.show()
    plt.clf()
    fpr, tpr, threshold = roc_curve(y_test, probs_y[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig("roc.png")
    plt.show()
    plt.clf()


if __name__ == "__main__":
    filename = 'bankloan.xls'
    df = pd.read_excel(filename)
    X = df.iloc[:, :-1]  # the features
    y = df.iloc[:, -1]  # the labels
    plt.style.use('ggplot')
    t1(X, y)
    t2(X, y)
