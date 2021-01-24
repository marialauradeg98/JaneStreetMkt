"""
In this module we explore the effect of max depth on prediction accuracy and
overfitting.
After chosing the other hyperparameters, 15 Random forests are fitted with 15
different values of max_depth.
For each RF the accuracy score on test and training set are computed and then
they are plotted over the value of max_depth.
"""
import time
import gc
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from initial_import import import_training_set
from splitting import split_data
import matplotlib.pyplot as plt


if __name__ == "__main__":
    #import data
    data = import_training_set()

    # skip first 85 days because of change JaneStreetMkt trading critieria
    #data = data[data["date"] > 85]
    #data["date"] = data["date"]-85

    # divide test and training set
    X, y, X_test, y_test = split_data(data)

    score_train = []  # empty list will contain accuracy score on training set of each iteration
    score_test = []  # empty list will contain accuracy score on test set of each iteration
    comp_time = []  # empty list will contain cmputational time of each iteration

    # create 15 models with 15 different values of max_depth and saves
    # computational time, accuracy score on training and test set
    for depth in range(5, 80, 5):
        start = time.time()
        # set hyperparameters
        parameters = {'n_estimators': 577,
                      'max_features': "auto",
                      'max_depth': depth,
                      'bootstrap': True,
                      'min_samples_leaf': 6,
                      "max_samples": 0.10,  # 0.265
                      'verbose': 1,
                      'n_jobs': -1,
                      'random_state': 18}

        # build and fit model
        forest = ExtraTreesClassifier(**parameters)
        forest.fit(X, y)

        # compute score on test and training set
        score_test.append(forest.score(X_test, y_test))
        score_train.append(forest.score(X, y))

        # get computational time
        finish = (time.time()-start)/60
        comp_time.append(finish)
        del forest
        gc.collect()

    # create array containing values of max_depth
    depth = range(5, 80, 5)

    # plot accuracy on training and test set over max_depth
    plt.plot(depth, score_train, label="score training", marker="o", c="b")
    plt.plot(depth, score_test, label="score test", marker="o", c="r")
    plt.title("Score over max depth", fontsize=18)
    plt.xlabel("Max depth")
    plt.ylabel("Accuracy score")
    plt.legend()
    plt.show()

    # plot computational time over max_depth
    plt.plot(depth, comp_time, label="score training", marker="o", c="g")
    plt.title("Computational time over max depth", fontsize=18)
    plt.xlabel("Max depth")
    plt.ylabel("Computational time [min]")
    plt.legend()
    plt.show()
