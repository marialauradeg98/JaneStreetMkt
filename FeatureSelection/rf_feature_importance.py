"""
This feature selection model is based on a tecnique explained in the book "Advances
in financial ML" by De Prado
The idea behind the tecnique is to use a RF classifier to compute the MDI feature importance
of each feature. MDI is normalize to 1 and if each feature is equally
important each of one them should have an MDI score of 1/n (n=number of features).
This means we can remove the features with a (MDI score + standard deviation) minor
than 1/n.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from initial_import import import_training_set
from splitting import split_data
import feature_selection


def compute_feat_imp(model, columns_names):
    """
    This functions finds the most important features of the dataset
    through the feature importance module of a random forest and deletes the redundant one.

    Parameters
    ----------
    model: classifier
        the RF model
    column_names: list of strings
        the names of the features

    Yields
    ------
    deleted_feat: np.array of strings
        the names of the redunant features
    """
    # compute importance and std deviation
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_],
                 axis=0)

    # create a dataframe containig the feature names,value of importances and std dev
    dictionary = {"features": columns_names, "importance": importances, "standard_dev": std}
    imp_data = pd.DataFrame(dictionary)
    imp_data = imp_data.sort_values(by=["importance"], ascending=False)  # sort array

    # nice prints
    print("The feature importances are:")
    print(imp_data)

    most_imp = imp_data.iloc[0:30, :]  # select 30 most important features

    # plot barplot of most important features
    most_imp.plot(
        kind="bar",
        x="features",
        y="importance",
        yerr="standard_dev"
    )
    plt.title("Most important features")

    tresh = 1 / imp_data.shape[0]  # compute treshold
    # plot an horizontal line that rapresents the treshold
    plt.axhline(y=tresh, linestyle="--")
    plt.show()

    # if std deviation + importance is < treshold the feature is redundant
    imp_data["imp+std"] = imp_data["importance"]+imp_data["standard_dev"]
    redundant_feat = imp_data[imp_data["imp+std"] < tresh]  # find redundant features
    # cancel them from data
    imp_data = imp_data[imp_data["imp+std"] > tresh].drop(["imp+std"], axis=1)

    # plot redundant features to make sure no errors were made
    redundant_feat.plot(
        kind="bar",
        x="features",
        y="importance",
        yerr="standard_dev"
    )

    plt.axhline(y=tresh, linestyle="--")  # plot treshold as horizontal line
    plt.show()

    # save redunant features as numpy array
    deleted_feat = (np.array(redundant_feat["features"]))

    # nice prints
    print("Number of redundant features {}".format(redundant_feat.shape[0]))
    print("Deleted features:")
    print(deleted_feat)
    return deleted_feat


if __name__ == "__main__":
    CORRELATION = True
    SKIP_85_DAYS = True

    data = import_training_set()  # import training set

    # skip first 85 days because of change JaneStreetMkt trading critieria
    if SKIP_85_DAYS is True:
        data = data[data["date"] > 85]
        data["date"] = data["date"]-85

    # Remove feature based on correlation
    if CORRELATION is True:
        useless = feature_selection.main(0.93)
        data = data.drop(useless, axis=1)

    # divide test and training set
    X_train, y_train, X_test, y_test = split_data(data)

    # params of the RF model
    params = {'n_estimators': 250,
              'max_features': "auto",
              'max_depth': 50,
              'bootstrap': True,
              'min_samples_leaf': 5,
              "max_samples": 0.20,
              'verbose': 2,
              'n_jobs': -1,
              'random_state': 18}

    # build RF model
    start = time.time()
    forest = ExtraTreesClassifier(**params)
    forest.fit(X_train, y_train)
    finish = (time.time()-start)/60  # time to fit model in minutes

    # compute accuracy on test and training set
    score_test = forest.score(X_test, y_test)
    score_training = forest.score(X_train, y_train)
    print("Accuracy on training set is: {} \nAccuracy on test set is: {}".format(
        score_training, score_test))

    # create an array with the features' names
    columns_data = np.zeros(X_train.shape[1])
    columns_data = X_train.columns

    # delete redundant features from train and test set
    redunant_feat = compute_feat_imp(forest, columns_data)
    X_train = X_train.drop(redunant_feat, axis=1)
    X_test = X_test.drop(redunant_feat, axis=1)

    # save deleted features
    np.savetxt("Results/deleted_feat_skip85.csv",
               redunant_feat, fmt="%s", delimiter=',')

    # create a dataframe that sumarizes the results
    results = {"score test ": score_test,
               "score training": score_training,
               "computational time (min)": finish,
               "num_deleted_features": len(redunant_feat),
               "Correlation": CORRELATION,
               "Skip 85 Days": SKIP_85_DAYS}
    end_results = pd.DataFrame(results, index=["values"])

    # nice print
    print("a little recap of the results:")
    print(end_results)

    # save results features as csv
    end_results.to_csv("Results/results_del_feat.csv")
