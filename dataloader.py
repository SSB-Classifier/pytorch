from network import Classifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tools.arff import Arff
import numpy as np
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import torch
import torch.optim as optim
from sklearn.model_selection import KFold


def multi_factor_load(filename, encode_x=[], drop=[]):
    print(filename)
    arff = Arff(label_count=1)
    arff.load_arff(filename)
    Y = []
    X = []
    X_encode = []
    all = []
    for i in arff.data:
        all.append(list(i))
        Y.append([i[-1]])
        encode = []
        no_encode = []
        for j in range(len(i) - 1):
            if j in encode_x:
                encode.append(i[j])
            elif j not in drop:
                no_encode.append(i[j])
        X.append(no_encode)
        X_encode.append(encode)

    enc = OneHotEncoder()
    enc.fit(Y)
    Y = enc.transform(Y).toarray()
    if len(encode_x) > 0:
        enc.fit(X_encode)
        X_encode = enc.transform(X_encode).toarray()
        X = np.concatenate((X, X_encode), axis=1)

    return np.asarray(X), Y


def test(X_test, Y_test):
    temp_losses = []
    for x, y_truth in zip(X_test, Y_test):
        y_hat = model(x)
        temp_losses.append((int)(y_hat.argmax() == y_truth.argmax()))
    return np.mean(temp_losses)


if __name__ == "__main__":

    X, Y = multi_factor_load('smash-bros.arff', drop=[0, 2])
    X = torch.from_numpy(X).cuda().float()
    Y = torch.from_numpy(Y).cuda().float()
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.33, shuffle=True)
    model = Classifier(X, Y).cuda()
    objective = torch.nn.CrossEntropyLoss()
    losses = []
    test_accuracies = []
    kf = KFold(n_splits=5)
    kf.get_n_splits(X)
    test_maxes = []
    for train_index, test_index in kf.split(X):
        model = Classifier(X, Y).cuda()
        optimizer = optim.Adam(model.parameters(), lr=1e-6)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        loop = tqdm(range(100))
        for epoch in range(100):
            for x, y_truth in zip(X_train, y_train):
                optimizer.zero_grad()

                y_hat = model(x)

                loss = objective(y_hat.unsqueeze(0), y_truth.argmax().unsqueeze(0))
                losses.append(loss)

                loss.backward()
                optimizer.step()
                loop.set_description('loss:{:.4f}'.format(loss.item()))
            # loop.set_description('loss:{:.4f}'.format(test(X_test, y_test)))
            loop.update(1)
            test_accuracies.append(test(X_test, y_test))
        print(max(test_accuracies))
        test_maxes.append(max(test_accuracies))
        test_accuracies = []
        loop.close()
    print(np.mean(test_maxes))
