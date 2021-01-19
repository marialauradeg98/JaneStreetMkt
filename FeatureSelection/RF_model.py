import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
import feature_selection
from initial_import import import_training_set
from PurgedGroupTimeSeriesSplit import PurgedGroupTimeSeriesSplit
from sklearn.model_selection import TimeSeriesSplit
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score
import pickle
from sklearn.metrics import roc_auc_score
from splitting import test_train_val


def acc_model(params):
    clf = ExtraTreesClassifier(**params)
    clf.fit(X_train, y_train)  # fit RF
    # set the function to minimize
    return clf.score(X_val, y_val)


def f(params):
    global best
    acc = acc_model(params)
    if acc > best:
        best = acc
        print("New best")
    print(params, acc)
    return {'loss': -acc, 'status': STATUS_OK}


if __name__ == "__main__":
    Correlation = False
    NewStart = True
    GridSearch = False
    Skip85Days = False
    No0Weight = False
    RFImportance = True
    iteration = 1

    data = import_training_set()  # import training set

    # skip first 85 days because of change of JaneStreetMkt trading critieria
    if Skip85Days == True:
        data = data[data["date"] > 85]
        data["date"] = data["date"]-85

    # skip 0 weight transaction
    if No0Weight == True:
        data = data[data["weight"] != 0]

    # Remove feature based on correlation
    if Correlation == True:
        useless = feature_selection.main(0.90)
        data = data.drop(useless, axis=1)

    # remove features based on MSI feature importance
    if RFImportance == True:
        redunant_feat = np.loadtxt("Results/deleted_feat1.csv", dtype="str")
        data = data.drop(redunant_feat, axis=1)

    # divide test and training set
    X_train, y_train, X_val, y_val, X_test, y_test = test_train_val(data)

    # define search_space
    search_space = {
        "n_estimators": hp.choice("n_estimators", range(2, 10)),
        "max_features": hp.choice("max_features", ["auto", "log2"]),
        "max_depth": hp.choice("max_depth", range(10, 100)),
        "min_samples_leaf": hp.choice("min_samples_leaf", range(1, 20)),
        "bootstrap": True,
        # "min_impurity_decrease": hp.uniform("min_impurity_decrease", 0.0, 0.25),
        "n_jobs": -1,
    }

    # if newstart is true we start optimization from scratch
    if NewStart is True:
        eval = 10  # number of evaluation at the end of a cyle
        best = 0

        start = time.time()  # used to get computational time
        trials = Trials()
        # start of the tuning process
        best = fmin(f, search_space, algo=tpe.suggest, max_evals=10, trials=trials)

        # nice prints
        print('best:')
        print(best)
        finish = (time.time()-start)/60  # time to fit model in minutes
        print("Time to search hyperparameters {} min".format(finish))

        # The trials database now contains 10 entries, it can be saved/reloaded with pickle
        pickle.dump(trials, open("Hyperopt/myfile.p", "wb"))
        # write on a txt file number of trials done
        eval_file = open("Hyperopt/eval.txt", "w")
        eval_file.write("{}".format(eval))
        eval_file.close()

    # now we can continue the optimization process from when we stopped
    CONTINUE = True

    while CONTINUE is True:
        # load previous results
        trials = pickle.load(open("Hyperopt/myfile.p", "rb"))
        best = 0

        start = time.time()  # used to get computational time
        # load total numer of evaluation already done
        textfile = open("Hyperopt/eval.txt", "r")
        eval = int(textfile.read())
        textfile.close()

        eval = eval+10  # at the end of the cycle we will have ten more evaluations
        best = fmin(f, search_space, algo=tpe.suggest, max_evals=eval, trials=trials)
        finish = (time.time()-start)/60  # time to fit model in minutes
        print("Time to search hyperparameters {} min".format(finish))
        print('best:')
        print(best)
        # The trials database now contains 10 +eval entries, it can be saved/reloaded with pickle
        pickle.dump(trials, open("Hyperopt/myfile.p", "wb"))

        # save new number of total evaluations
        eval_file = open("Hyperopt/eval.txt", "w")
        eval_file.write("{}".format(eval))
        eval_file.close()

        # continue the cycle until we press n
        save = input("Done :) \nDo you want to continue?\ny/n\n")
        if save == "n":
            CONTINUE = False

    '''
    # fit a model
    params = {}
    start = time.time()
    forest = ExtraTreesClassifier(**params, verbose=2, n_jobs=1)
    forest.fit(X_train, y_train)
    finish = (time.time()-start1)/60  # time to fit model in minutes

    # compute accuracy on test and training set
    score_test = forest.score(X_test, y_test)
    score_training = forest.score(X_train, y_train)
    print("Accuracy on training set is: {} \nAccuracy on test set is: {}".format(
        score_training, score_test))

    # create a dataframe that sumarizes the results
    results = {"score test ": score_test, "score training": score_training, "computational time (min)": finish,
               "corr feat sel": Correlation, "skip 85 days": Skip85Days, "drop 0 weight": No0Weight}

    end_results = pd.DataFrame(results)

    print("a little recap of the results:")
    print(end_results)

    # save results and removed features as csv
    end_results.to_csv("Results/results{}.csv".format(iteration))
    '''
