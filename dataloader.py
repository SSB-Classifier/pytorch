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
from queue import PriorityQueue
import itertools
import concurrent.futures
from multiprocessing import Pool
import multiprocessing


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


def run(X, Y, n_splits=5, lr=1e-6, n_epochs=25):
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()
    objective = torch.nn.CrossEntropyLoss()
    losses = []
    test_accuracies = []
    kf = KFold(n_splits=n_splits)
    kf.get_n_splits(X)
    test_maxes = []
    # loop = tqdm(range(n_epochs*n_splits))
    for train_index, test_index in kf.split(X):
        model = Classifier(X, Y)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        for epoch in range(n_epochs):
            for x, y_truth in zip(X_train, y_train):
                optimizer.zero_grad()

                y_hat = model(x)

                loss = objective(y_hat.unsqueeze(0), y_truth.argmax().unsqueeze(0))
                losses.append(loss)

                loss.backward()
                optimizer.step()
                # loop.set_description('loss:{:.4f}'.format(loss.item()))
            # loop.update(1)
            test_accuracies.append(test(X_test, y_test, model))
        test_maxes.append(max(test_accuracies))
        test_accuracies = []
    # loop.close()
    print(np.mean(test_maxes))
    return np.mean(test_maxes)


def filter_features(X, to_keep):
    to_return = []
    for row in X:
        newrow = []
        for index in range(len(row)):
            if index in to_keep:
                newrow.append(row[index])
        to_return.append(newrow)
    return np.asarray(to_return)

def thread_wrapper(combo, X, Y):
    filtered_x = filter_features(X, combo)
    return run(filtered_x, Y), combo


def get_best_combination(X, Y, indexes):
    num_features = len(indexes)-1
    max_accuracy = -1
    best_combo = None
    with Pool(processes=multiprocessing.cpu_count() // 2) as executor:
        futures = [executor.apply_async(thread_wrapper, (x, X, Y)) for x in itertools.combinations(indexes, num_features)]
        for future in futures:
            accuracy, combo = future.get(timeout=1000)
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                best_combo = combo
        # for combo in itertools.combinations(indexes, num_features):
        #     filtered_x = filter_features(X, combo)
        #     accuracy = run(filtered_x, Y)
        #     if accuracy >= max_accuracy:
        #         max_accuracy = accuracy
        #         best_combo = combo
    return max_accuracy, best_combo


def test(X_test, Y_test, model):
    temp_losses = []
    for x, y_truth in zip(X_test, Y_test):
        y_hat = model(x)
        temp_losses.append((int)(y_hat.argmax() == y_truth.argmax()))
    return np.mean(temp_losses)


if __name__ == "__main__":

    X, Y = multi_factor_load('smash-bros.arff', drop=[0, 2])
    max_accuracy = -1
    max_features = [x for x in range(X.shape[-1])]
    while True:
        new_accuracy, new_feature_set = get_best_combination(X, Y, max_features)
        if max_accuracy > new_accuracy:
            break
        max_accuracy = new_accuracy
        max_features = new_feature_set

    print(max_accuracy)
    print(max_features)

    run(X, Y, 5, 100)
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.33, shuffle=True)

