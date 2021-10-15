import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import precision_score, recall_score


def t1(df):
    w = [7, 17, 27, 37, 47, 57]
    train_accu = []
    test_accu = []
    for i in w:
        X = df.iloc[:, :i]  # the features
        y = df.iloc[:, -1]  # the labels
        test_tmp = []
        train_tmp = []
        for j in range(100):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            clf = svm.SVC(C=1, kernel='linear')
            clf.fit(X_train, y_train)
            train_tmp.append(clf.score(X_train, y_train))
            test_tmp.append(clf.score(X_test, y_test))
        test_accu.append(test_tmp)
        train_accu.append(train_tmp)
    test_avg = []
    train_avg = []
    test_std_dev = []
    train_std_dev = []
    for i in range(len(w)):
        test_avg.append(np.mean(test_accu[i]))
        test_std_dev.append(np.std(test_accu[i]))
        train_avg.append(np.mean(train_accu[i]))
        train_std_dev.append(np.std(train_accu[i]))
    print("-----TASK 1-----")
    print("The average accuracy on training data with different number of features:")
    print(train_avg)
    print("The standard deviation of accuracy on training data with different number of features:")
    print(train_std_dev)
    print("The average accuracy on test data with different number of features:")
    print(test_avg)
    print("The standard deviation of accuracy on test data with different number of features:")
    print(test_std_dev)
    plt.plot(w, train_avg, marker="x", label="training")
    plt.plot(w, test_avg, marker="+", label="test")
    plt.xlim([0, 60])
    plt.xlabel('first W features')
    plt.ylabel('average accuracy')
    plt.title('number of features vs avg.')
    plt.legend()
    plt.savefig("avg.png")
    plt.show()
    plt.clf()
    plt.plot(w, train_std_dev, marker="x", label="training")
    plt.plot(w, test_std_dev, marker="+", label="test")
    plt.xlim([0, 60])
    plt.xlabel('first W features')
    plt.ylabel('standard deviation')
    plt.title('number of features vs std. dev.')
    plt.legend()
    plt.savefig("std_dev.png")
    plt.show()
    plt.clf()


def t2_1(df):
    X = df.iloc[:, :-1].values  # the features
    y = df.iloc[:, -1].values  # the labels
    c = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 10]
    cc = ["0.001", "0.01", "0.05", "0.1", "0.5", "1", "10"]
    accu = []
    margin = []
    for i in c:
        accu_tmp = []
        margin_tmp = []
        kf = KFold(n_splits=10, shuffle=True)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = svm.SVC(C=i, kernel='linear')
            clf.fit(X_train, y_train)
            accu_tmp.append(clf.score(X_test, y_test))
            w = clf.coef_[0]
            margin_tmp.append(2/np.linalg.norm(w, ord=2))
        accu.append(accu_tmp)
        margin.append(margin_tmp)
    accu_avg = []
    accu_std_dev = []
    margin_avg = []
    for i in range(len(c)):
        accu_avg.append(np.mean(accu[i]))
        accu_std_dev.append(np.std(accu[i]))
        margin_avg.append(np.mean(margin[i]))
    print("-----TASK 2-----")
    print("The average accuracy with different penalty coefficients:")
    print(accu_avg)
    print("The std. dev. of accuracy with different penalty coefficients:")
    print(accu_std_dev)
    print("The average margin with different penalty coefficients:")
    print(margin_avg)
    plt.plot(accu_avg, marker="x", label="avg_accu")
    plt.xlabel("penalty coefficients")
    plt.xticks(range(7), cc)
    plt.ylabel('average accuracy')
    plt.title('avg. accuracy with different penalty coefficients')
    plt.legend()
    plt.savefig("avg_accu_pena.png")
    plt.show()
    plt.clf()
    plt.plot(accu_std_dev, marker="x", label="std_dev")
    plt.xlabel("penalty coefficients")
    plt.xticks(range(7), cc)
    plt.ylabel('standard deviation')
    plt.title('std. dev. with different penalty coefficients')
    plt.legend()
    plt.savefig("std_dev_pena.png")
    plt.show()
    plt.clf()
    plt.plot(margin_avg, marker="x", label="margin_avg")
    plt.xlabel("penalty coefficients")
    plt.xticks(range(7), cc)
    plt.ylabel('average margin')
    plt.title('avg. margin with different penalty coefficients')
    plt.legend()
    plt.savefig("avg_margin_pena.png")
    plt.show()
    plt.clf()


def t2_2(df):
    X = df.iloc[:, :-1]  # the features
    y = df.iloc[:, -1]  # the labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = svm.SVC(C=0.5, kernel='linear')
    clf.fit(X_train, y_train)
    print("The accuracy on test data with the chosen penalty coefficient is %s" % clf.score(X_test, y_test))


def t3(df):
    X = df.iloc[:, :-1].values  # the features
    y = df.iloc[:, -1].values  # the labels
    svm_accu = []
    lr_accu = []
    svm_prec = []
    lr_prec = []
    svm_recall = []
    lr_recall = []
    kf = KFold(n_splits=10, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = svm.SVC(C=0.5, kernel='linear')
        clf.fit(X_train, y_train)
        svm_accu.append(clf.score(X_test, y_test))
        y_pred = clf.predict(X_test)
        svm_prec.append(precision_score(y_test, y_pred))
        svm_recall.append(recall_score(y_test, y_pred))
        LR = lr(max_iter=1000)
        LR.fit(X_train, y_train)
        lr_accu.append(LR.score(X_test, y_test))
        probs_y = LR.predict(X_test)
        lr_prec.append(precision_score(y_test, probs_y))
        lr_recall.append(recall_score(y_test, probs_y))
    print("-----TASK 3-----")
    print("The average and the standard deviation of the accuracy of SVM:")
    print(np.mean(svm_accu), np.std(svm_accu))
    print("The average and the standard deviation of the accuracy of Logisitc Regression:")
    print(np.mean(lr_accu), np.std(lr_accu))
    print("The average precision and recall of SVM:")
    print(np.mean(svm_prec), np.mean(svm_recall))
    print("The average precision and recall of Logistic Regression:")
    print(np.mean(lr_prec), np.mean(lr_recall))
    mean_diff = np.mean(np.array(svm_accu)-np.array(lr_accu))
    std_diff = np.std(np.array(svm_accu)-np.array(lr_accu))
    t = mean_diff/(std_diff/np.sqrt(10))
    print("t-statistic value=", end='')
    print(t)


if __name__ == "__main__":
    filename = './spambase.csv'
    df = pd.read_csv(filename)
    plt.style.use('ggplot')
    t1(df)
    t2_1(df)
    t2_2(df)
    t3(df)
