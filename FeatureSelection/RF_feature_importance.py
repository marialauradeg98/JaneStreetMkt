"""
This feature selection model is based on a tecnique explained in the book Advances
in financial ML by
The idea behind the tecnique is to use a RF classifier to compute the MSI feature importance
of each feature. MSI has the property that it sum is 1 and if each feature is equally
important each of them should have an MSI score of 1/n (n=number of features).
This means we can remove the features which (MSI score + standard deviation) is minor
than 1/n
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV
import feature_selection
from initial_import import import_training_set
from PurgedGroupTimeSeriesSplit import PurgedGroupTimeSeriesSplit
from sklearn.model_selection import TimeSeriesSplit


def compute_feat_imp(model, columns_names):
    """
    This functions finds the most important features of the dataset
    through the feature importance module of a random forest and deletes the redundant one.

    Parameters
    ----------
    model: float
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

    # plot redundant features to make sure no errors were made
    fig3 = redundant_feat.plot(
        kind="bar",
        x="features",
        y="importance",
        yerr="standard_dev"
    )

    fig4 = plt.axhline(y=tresh, linestyle="--")  # plot treshold as horizontal line
    plt.show()

    # save redunant features as numpy array
    deleted_feat = (np.array(redundant_feat["features"]))

    # nice prints
    print("Number of redundant features {}".format(redundant_feat.shape[0]))
    print("Deleted features:")
    print(deleted_feat)
    return deleted_feat


def test_train(data):
    """
    This function splits test and training set leaving a 50 days gap between them to prevent
    information leakage
    Parameters
    ----------
    data: DataFrame
        data we want to split

    Yields
    ------
    X_train: DataFrame
        training dataset without class label
    y_train: np.array
        class label for training set
    X_test: DataFrame
        test dataset without class label
    y_test: np.array
        class label for test set
     """
    data = data.fillna(0)  # fill missing values

    # split training and test set and leave a 50 days gap between them
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
    # prepare date for modeling
    useless = feature_selection.main(0.90)
    data = data.drop(useless, axis=1)
    # divide test and training set
    X_train, y_train, X_test, y_test = test_train(data)

    # only do random search if we want to know the best hyperparameters of the RF
    RandomSearch = True
    if RandomSearch == True:
        start = time.time()

        # Number of trees in random forest
        n_estimators = [250, 500,  750, 1000]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [10, 25, 50, 75]
        # bramch will stop splitting after this number of smaple in leaf
        min_samples_leaf = [2, 5, 10, 20]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]

        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}

        # Define the cv methond used
        cv = TimeSeriesSplit(
            n_splits=5,
            gap=int(3.0e5),
            max_train_size=int(1.25e6),
            test_size=int(2.5e5))

        rf = ExtraTreesClassifier()  # classifier

        rf_random = RandomizedSearchCV(
            estimator=rf,
            param_distributions=random_grid,
            n_iter=25,
            verbose=2,
            random_state=42,
            cv=cv,
            n_jobs=-1,
            scoring="f1")

        # Fit the random search model
        rf_random.fit(X_train, y_train)
        # print best_params_
        params = (rf_random.best_params_)
        print(params)
        # print accuracy score
        print(rf_random.score(X_test, y_test))

        # compute execution time
        mins = (time.time()-start)//60
        sec = (time.time()-start) % 60
        print('Time to perform RandomizedSearchCV is: {} min {:.2f} sec\n'.format(mins, sec))

    else:
        params = {'n_estimators': 1000,
                  'max_features': "auto",
                  'max_depth': 50,
                  'min_samples_leaf': 5,
                  'bootstrap': True}

    # build RF with best params
    forest = ExtraTreesClassifier(**params, verbose=2, n_jobs=-1)
    forest.fit(X_train, y_train)

    # create an array with the features' names
    columns_names = np.zeros(X_train.shape[1])
    columns_names = X_train.columns

    # delete redundant features from train and test set
    redunant_feat = compute_feat_imp(forest, columns_names)
    X_train = X_train.drop(redunant_feat, axis=1)
    X_test = X_test.drop(redunant_feat, axis=1)

    # fit a model over new dataset
    forest1 = ExtraTreesClassifier(**params, verbose=2, n_jobs=-1)
    forest1.fit(X_train, y_train)
    print(forest1.score(X_test, y_test))  # compute accuracy
