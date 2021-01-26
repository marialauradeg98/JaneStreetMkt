"""
In the first part of this module we implement a bayesian optimizer to find the
best hyperparameters for an RF algorithm.
After that we fit 3 models with the 3 best hyperparameters using a 5 fold
TimeSeriesSplit cross validation and we compute the accuracy on training and test set.
TimeSeriesSplit is used for 2 reasons:
1) We want to make sure that the models doesn't look into future data
2) We introduce a gap between test and train in each fold  to prevent
   information leakage
"""
import time
import pickle
import gc
from statistics import mean, stdev
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from initial_import import import_training_set
from splitting import split_data
import feature_selection
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


def time_cv(clf, n_folds, t_size=int(2e5), gap_size=int(2e5)):
    """
    This function train a model with TimeSeriesSplit cross validation.

    Parameters
    ----------
    clf: classifier
        The classifier we want to fit.
    n_folds: int
        number of CV splits
    t_size: int (default 2e5)
        Maximum test size for each CV split
    gap_size: int (default 2e5)
        Gap between training and test for each split

    Yields
    ------
    score_cv: list of float
        AUC score at each iteration

    """

    folds = TimeSeriesSplit(n_splits=n_folds,
                            test_size=t_size,
                            gap=gap_size)

    # create splits
    splits = folds.split(X, y)

    score_cv = []  # empty list will contain accuracy score of each split

    # start cross validation
    for fold_n, (train_index, val_index) in enumerate(splits):
        print('Fold: {}'.format(fold_n+1))

        # for each iteration define the boundaries of training and validation set
        X_train = X.iloc[train_index[0]:train_index[-1], :]
        X_val = X.iloc[val_index[0]:val_index[-1], :]
        y_train = y[train_index[0]:train_index[-1]]
        y_val = y[val_index[0]:val_index[-1]]

        # fit model and compute score
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_val)
        accuracy = roc_auc_score(y_val, predictions)
        score_cv.append(accuracy)
        print("AUC: {}".format(accuracy))

        # delete train and validation set to save memory
        del X_train, X_val, y_train, y_val
        gc.collect()

    return score_cv


def acc_model(params):
    """
    This function computes the AUC score of the model on the validation set

    Parameters
    ----------
    params: **kwargs
        paramters for the RF model

    Yields
    ------
    accuracy: float
        AUC score
    """

    clf = ExtraTreesClassifier(**params)
    list_auc = time_cv(clf, 5)
    accuracy = mean(list_auc)
    std = stdev(list_auc)

    return accuracy, std


def func(params):
    """
    This is the functon we want to minimize during the tuning process

    Parameters
    ----------
    params: **kwargs
        paramters for the RF model

    Yields
    ------
    loss: float
        the parameter we want to minimize (-AUC score)
    status:
        this parameter reports if an error occured while the model is fitting
    train_time:
        time fit the model and compute the AUC score
    """
    begin = time.time()
    print(params)
    acc, std = acc_model(params)
    print("Mean AUC {:.5f} \nStd {:.5f}".format(acc, std))
    end = (time.time() - begin) / 60
    return {'loss': -acc, 'status': STATUS_OK, 'train_time': end, 'std': std}


if __name__ == "__main__":

    CORRELATION = False
    NEW_START = False
    SKIP_85_DAYS = True
    NO_0_WEIGHT = False
    RF_IMPORTANCE = False
    HYPERPARAMETER_SEARCH = True

    data = import_training_set()  # import training set

    # skip first 85 days because of change of JaneStreetMkt trading critieria
    if SKIP_85_DAYS is True:
        data = data[data["date"] > 85]
        data["date"] = data["date"]-85

    # skip 0 weight transaction
    if NO_0_WEIGHT is True:
        data = data[data["weight"] != 0]

    # Remove feature based on correlation
    if CORRELATION is True:
        useless = feature_selection.main(0.93)
        data = data.drop(useless, axis=1)

    # remove features based on MSI feature importance
    if RF_IMPORTANCE is True:
        redunant_feat = np.loadtxt("Results/deleted_feat_skip85.csv", dtype="str")
        data = data.drop(redunant_feat, axis=1)

    if HYPERPARAMETER_SEARCH is True:
        # divide dataset into test training and validation set
        X, y, X_test, y_test = split_data(data)

        # define search_space
        search_space = {
            "n_estimators": hp.choice("n_estimators", range(200, 750)),
            "max_features": hp.choice("max_features", ["auto", "log2"]),
            "max_depth": hp.choice("max_depth", range(10, 45)),
            "min_samples_leaf": hp.choice("min_samples_leaf", range(1, 20)),
            "max_samples": hp.uniform("max_samples", 0.1, 0.25),
            "bootstrap": True,
            "criterion": hp.choice("criterion", ["gini", "entropy"]),
            "random_state": 18,
            "n_jobs": -1,
        }

        # if newstart is true we start optimization from scratch
        if NEW_START is True:
            SEARCHES = 5  # number of searches at the end of a cyle
            start = time.time()  # used to get computational time

            # this will contains all the informations about the search process
            trials = Trials()

            # search hyperparameters
            best = fmin(func, search_space, algo=tpe.suggest, max_evals=5, trials=trials)

            # nice prints
            print('best:')
            print(best)
            finish = (time.time()-start)/60  # time to fit model in minutes
            print("Time to search hyperparameters {:.2f} min".format(finish))

            # The trials database now contains 10 entries, it can be saved/reloaded with pickle

            pickle.dump(trials, open("Hyperopt/myfile3.p", "wb"))
            # write on a txt file number of trials done
            search_file = open("Hyperopt/searches3.txt", "w")
            search_file.write("{}".format(SEARCHES))
            search_file.close()

        # now we can continue the optimization process from when we stopped
        CONTINUE = True

        while CONTINUE is True:

            # load previous results
            trials = pickle.load(open("Hyperopt/myfile3.p", "rb"))
            start = time.time()  # used to get computational time

            # load total numer of evaluation already done
            textfile = open("Hyperopt/searches3.txt", "r")
            SEARCHES = int(textfile.read())
            textfile.close()

            SEARCHES = SEARCHES+5  # at the end of the cycle we will have ten more evaluations

            # searh hyperparameters
            best = fmin(func, search_space, algo=tpe.suggest, max_evals=SEARCHES, trials=trials)
            finish = (time.time()-start)/60  # time to fit model in minutes

            # nice prints
            print("Time to search hyperparameters {} min".format(finish))
            print('best:')
            print(best)

            # The trials database now contains 10 +searches entries
            # it can be saved/reloaded with pickle
            pickle.dump(trials, open("Hyperopt/myfile3.p", "wb"))

            # save new number of total evaluations
            search_file = open("Hyperopt/searches3.txt", "w")
            search_file.write("{}".format(SEARCHES))
            search_file.close()

            # continue the cycle until we press n
            save = input("Done :) \nDo you want to continue?\ny/n\n")
            if save == "n":
                CONTINUE = False

    else:
        # split data
        X, y, X_test, y_test = split_data(data)

        # best 3 hyperparameter from search

        parameters1 = {'criterion': "gini",
                       'n_estimators': 398,
                       'max_features': "auto",
                       'max_depth': 35,
                       'bootstrap': True,
                       'min_samples_leaf': 14,
                       "max_samples": 0.1660,
                       'verbose': 1,
                       'n_jobs': -1,
                       'random_state': 18}

        parameters2 = {'criterion': "gini",
                       'n_estimators': 672,
                       'max_features': "auto",
                       'max_depth': 37,
                       'bootstrap': True,
                       'min_samples_leaf': 14,
                       "max_samples": 0.1546,
                       'verbose': 1,
                       'n_jobs': -1,
                       'random_state': 18}

        parameters3 = {'criterion': "gini",
                       'n_estimators': 577,
                       'max_features': "auto",
                       'max_depth': 22,
                       'bootstrap': True,
                       'min_samples_leaf': 6,
                       "max_samples": 0.1259,
                       'verbose': 1,
                       'n_jobs': -1,
                       'random_state': 18}

        # create 3 different models
        forest1 = ExtraTreesClassifier(**parameters1)
        forest2 = ExtraTreesClassifier(**parameters2)
        forest3 = ExtraTreesClassifier(**parameters3)

        forests = (forest1, forest2, forest3)

        # set number of fold for CV
        N_FOLDS = 5

        # train eeach model with 5k CV
        for i, forest in enumerate(forests):
            print("Fitting forest with hyperparameters:")
            print(forest)
            start = time.time()
            time_cv(forest, N_FOLDS)
            finish = (time.time()-start)/60  # time to fit model in minutes
            print("Time to fit the RF{} with {}k cross validation {:.2f} min".format(
                i, N_FOLDS, finish))

            # save model
            filename = ("RF_{}.sav".format(i))
            pickle.dump(forest, open(filename, "wb"))

            # compute accuracy on test and training set
            score_test = forest.score(X_test, y_test)
            score_training = forest.score(X, y)
            print("Accuracy of RF{} on training set is: {} \nAccuracy on test set is: {}".format(
                i, score_training, score_test))

            # create a dataframe that sumarizes the results
            results = {"score test ": score_test, "score training": score_training,
                       "computational time (min)": finish, "splits": N_FOLDS}
            end_results = pd.DataFrame(results, index=["values"])

            # nice print
            print("a little recap of the results:")
            print(end_results)

            # save results
            end_results.to_csv("Results/RF_{}.csv".format(i))

            # compute AUC score
            pred_train = forest.predict(X)
            pred_test = forest.predict(X_test)
            auc_train = roc_auc_score(y, pred_train)
            auc_test = roc_auc_score(y_test, pred_test)
            print("AUC of RF{} on training set is: {} \nAUC on test set is: {}".format(
                i, auc_train, auc_test))
