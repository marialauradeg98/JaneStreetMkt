""" Module docstring """
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import feature_selection
from initial_import import import_training_set


def evaluate_extra_tree(model, X_test, y_test):
    score = model.score(X_test, y_test)
    print(score)
    return score


def compute_feat_imp(model, columns_names):
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
    fig = most_imp.plot(
        kind="bar",
        x="features",
        y="importance",
        yerr="standard_dev"
    )
    plt.title("Most important features")

    tresh = 1 / imp_data.shape[0]  # compute treshold
    # plot an horizontal line that rapresents the treshold
    fig1 = plt.axhline(y=tresh, linestyle="--")
    plt.show()

    # if std deviation + importance is < treshold the feature is redundant
    imp_data["imp+std"] = imp_data["importance"]+imp_data["standard_dev"]
    redundant_feat = imp_data[imp_data["imp+std"] < tresh]  # find redundant features
    # cancel them from data
    imp_data = imp_data[imp_data["imp+std"] > tresh].drop(["imp+std"], axis=1)

    fig3 = redundant_feat.plot(
        kind="bar",
        x="features",
        y="importance",
        yerr="standard_dev"
    )

    fig4 = plt.axhline(y=tresh, linestyle="--")
    plt.show()

    # nice prints
    print("Number of redundant features {}".format(redundant_feat.shape[0]))
    print("Deleted features:")
    deleted_feat = (np.array(redundant_feat["features"]))
    print(deleted_feat)
    return deleted_feat


def test_train(data):
    data = data.fillna(0)  # fill missing values
    # split training and test set
    data_test = data[data["date"] > 450]
    data = data[data["date"] < 400]

    # remove the labels from the train features
    X_train = data.drop(["action"], axis=1)
    # create an array the rapresent the class label
    y_train = np.ravel(data.loc[:, ["action"]])
    y_train = np.array(y_train)

    # remove the labels from the test features
    X_test = data_test.drop(["action"], axis=1)
    # create an array the rapresent the class label
    y_test = np.ravel(data_test.loc[:, ["action"]])
    y_test = np.array(y_test)
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":

    # Build a classification task using 3 informative features

    data = import_training_set()  # import training set

    # prepare the data for modelling
    X_train, y_train, X_test, y_test = test_train(data)

    # Build a model and compute the impurity-based feature importances
    forest = ExtraTreesClassifier(
        n_estimators=200,
        class_weight='balanced_subsample',
        criterion='entropy',
        max_depth=10,
        verbose=2
    )

    # Fit classifier
    forest.fit(X_train, y_train)  # fit model

    # get accuracy of model with no feature selection
    test_acc = evaluate_extra_tree(forest, X_test, y_test)
    train_acc = evaluate_extra_tree(forest, X_train, y_train)
    print("accuracy on trainig set is {}\n accuracy on test set is {}\n" .format(train_acc, test_acc))

    useless = feature_selection.main(0.92)
    data = data.drop(useless, axis=1)

    X_train, y_train, X_test, y_test = test_train(data)

    forest1 = ExtraTreesClassifier(
        n_estimators=200,
        class_weight='balanced_subsample',
        criterion='entropy',
        max_depth=10,
        verbose=2
    )

    # Fit classifier
    forest1.fit(X_train, y_train)  # fit model

    # get accuracy of model with some feature selection
    test_acc = evaluate_extra_tree(forest1, X_test, y_test)
    train_acc = evaluate_extra_tree(forest1, X_train, y_train)
    print("accuracy on trainig set is {}\n accuracy on test set is {}\n" .format(train_acc, test_acc))

    # create an array with the features' name
    columns_names = np.zeros(X_train.shape[1])
    columns_names = X_train.columns
    print(forest.get_params())

    redundant_feat = compute_feat_imp(forest1, columns_names)

    data.drop(redundant_feat, axis=1)

    X_train, y_train, X_test, y_test = test_train(data)

    forest2 = ExtraTreesClassifier(
        n_estimators=200,
        class_weight='balanced_subsample',
        criterion='entropy',
        max_depth=10,
        verbose=2
    )

    # Fit classifier
    forest2.fit(X_train, y_train)  # fit model

    # get accuracy of model with major feature selection
    test_acc = evaluate_extra_tree(forest2, X_test, y_test)
    train_acc = evaluate_extra_tree(forest2, X_train, y_train)
    print("accuracy on trainig set is {}\n accuracy on test set is {}\n" .format(train_acc, test_acc))
