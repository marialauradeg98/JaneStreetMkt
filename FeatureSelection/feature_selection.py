"""
The main goal of this module is to make a first attempt at feature selection
through a correlation analysis.
The idea is that we want to have features highly correleted with the class
and with a low correlation between them.
To achieve this goal we eliminate feature highly correlated with multiple
features and select between the remainings the ones with an high correlation
with the class.
"""
import time
import pandas as pd
from initial_import import import_training_set
from data_visualization_main import corr_filter


def remove_features(data, list_duplicates):
    """
    This function removes rows from the higly correlated DataFrame
    containing the names in list duplicates.

    Parameters
    -----------
    data: DataFrame
        original dataframe from which we want to remove features.
    list_duplicates: list of str
        a list of strings containig the names of the features we want to remove.
    Yields
    ------
    new_data: Dataframe
        a dataframe without the features in list duplicates
    """

    delete = []  # this list will contain the names of the features we want to remove
    max_index = len(data)
    # if an object from the list is in a row of the dataframe that row is going to be deleted
    for item in list_duplicates:
        for i in range(max_index):
            if data.iloc[i, 0] == item:
                delete.append(i)
            elif data.iloc[i, 1] == item:
                delete.append(i)
    new_data = data.drop(delete)

    # change the indexes' labels to avoid bugs later
    new_max = len(new_data)
    index_names = []
    for i in range(new_max):
        index_names.append(i)
    new_data.index = index_names
    return new_data


def remove_duplicates(data, treshold):
    """
    This function finds features highly correlated with multiple features.
    They will be the first features we are going to remove.

    Parameters
    -----------
    data: DataFrame
        original dataframe from wich we want to remove features.
    treshold: float
        the correlation treshold we use to consider feature pairings.
    Yields
    ------
    purged_data: Dataframe
        a dataframe without the features in list duplicates
    new_list: list of str
        list containig the features with multiple correlation pairings.
    """
    # transform the pandas dataframe to a list containing the names of the features
    old_list = []
    max_index = len(data)
    for index in range(max_index):
        if data.iloc[index, 2] > treshold or data.iloc[index, 2] < -treshold:
            old_list.append(data.iloc[index, 0])
            old_list.append(data.iloc[index, 1])

    # create two lists
    # this list will contain only 1 copy of each different element from the original list
    final_list = []
    # this list will contain the duplicate elements
    list_duplicates = []
    for item in old_list:
        if item not in final_list:
            final_list.append(item)
        elif item not in list_duplicates:
            list_duplicates.append(item)

    # create a list with the duplicate elements removed
    for item in list_duplicates:
        final_list.remove(item)
    # a few aestetichally pleasing prints
    num_duplicates = len(list_duplicates)
    print("There are {} features that have a correlation greater than {} \
            with more than one other feature\n"
          .format(num_duplicates, treshold))
    print("Features we are going to remove:\n")
    for item in list_duplicates:
        print(item)
    print("\n")

    # remove duplicate elements from dataframe
    purged_data = remove_features(data, list_duplicates)
    return purged_data, list_duplicates


def remove_redundat_feat(data, series, treshold):
    """
    This function select from each correlation pairing the one with less class correlation
    in order to remove it.

    Parameters
    -----------
    data: DataFrame
        original dataframe from wich we want to remove features.
    series: Series
        series containig feature-class correlation
    treshold: float
        the correlation treshold we use to consider feature pairings.
    Yields
    ------
    purged_data: Dataframe
        a dataframe without the feature pairings containing the randomly selected features.
    new_list: list of str
        list containig features with low class correlation.
    """
    new_list = []  # this list will contain the names of features we want to remove
    max_index = len(data)

    # for each row of the dataset delete the most redundant if a correlation treshold is met

    for index in range(max_index):
        feat1 = data.iloc[index, 0]
        feat2 = data.iloc[index, 1]
        if data.iloc[index, 2] > treshold or data.iloc[index, 2] < -treshold:
            # remove feature with least class correlation
            if series.loc[feat1] < series.loc[feat2]:
                new_list.append(data.iloc[index, 0])
            else:
                new_list.append(data.iloc[index, 1])

    # a few aestetichally pleasing prints
    num_feat = len(new_list)
    print("There are {} features pairings with a correlation greater than {}.\n"
          .format(num_feat, treshold))
    print("Features we are going to remove:\n")
    for item in new_list:
        print(item)

    # remove features from dataset
    purged_data = remove_features(data, new_list)

    return purged_data, new_list


def compute_correlation(filepath=None):
    """
    This function computes and saves the correlation matrix of the test set
    and the matrix with feature pairings with correlation greater than 0.9.

    Parameters
    -----------
    filepath: str (default=None)
        filepath where the two matrices wll be saved
    Yields
    ------
    2 csv files containing the two matrices
    """
    data = import_training_set()
    # delete first 85 days from dataset
    data = data[data["date"] > 85]
    # delete test set from dataset
    data = data[data["date"] < 400]
    # rescale days
    data["date"] = data["date"] - 85
    print("Working on computing highly correlated features\n")
    # compute higly correlated features and correlation matrix
    corr_matrix_90, corr_matrix = corr_filter(data, 0.90)
    # sort highly correlated features
    sorted_matrix = corr_matrix_90.sort_values(ascending=False)
    print(sorted_matrix)
    # save results to csv
    if filepath is None:
        sorted_matrix.to_csv("../JaneStreetMkt/Matrices/features_to_remove.csv")
        corr_matrix.to_csv("../JaneStreetMkt/Matrices/correlation_matrix.csv")
    else:
        sorted_matrix.to_csv()
        corr_matrix.to_csv()

    print("Matrices saved")


def main(treshold):
    """
    This is the main of the module.
    It combines all the previously defined functions to do a feature selections.

    Parameters
    -----------
    treshold: float
        the correlation treshold we use to consider feature pairings.
    Yields
    ------
    new_list: list of str
        list containig the features to remove.
    """

    start = time.time()  # useful to compute execution time
    # load correlated features matrix
    correlated_features = pd.read_csv(
        "../JaneStreetMkt/Matrices/features_to_remove.csv", header=None)
    action_corr = pd.read_csv("correlation_matrix.csv", index_col=0, header=0)
    action_corr = action_corr["action"].abs()

    # remove main features from the ones we want to eliminate
    no_main_features = ["date"]
    correlated_features = remove_features(correlated_features, no_main_features)

    # remove duplicates
    no_duplicates, feature_removed = remove_duplicates(correlated_features, treshold)
    # remove redundant features
    final_matrix, redund_feat = remove_redundat_feat(no_duplicates, action_corr, treshold)
    # create list with the feature to eliminate
    for item in redund_feat:
        feature_removed.append(item)
    # print total number of deleted features
    num_removed = len(feature_removed)
    print("Total number of feature removed:\n{}".format(num_removed))
    # compute execution time
    mins = (time.time()-start)//60
    sec = (time.time()-start) % 60
    print('Time to execute the feature selection model is time is: {} min {:.2f} sec\n'
          .format(mins, sec))
    return feature_removed
