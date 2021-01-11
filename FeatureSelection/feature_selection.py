import time
from numpy import random
import pandas as pd


def remove_features(data, list_duplicates):
    delete = []
    max_index = len(data)
    for item in list_duplicates:
        for i in range(max_index):
            if (data.iloc[i, 0] or data.iloc[i, 1]) == item:
                delete.append(i)

    new_data = data.drop(delete)
    new_max = len(new_data)
    index_names = []
    for i in range(new_max):
        index_names.append(i)
    new_data.index = index_names
    return new_data


def remove_duplicates(data, treshold):
    duplicate = []
    max_index = len(data)
    for index in range(max_index):
        if data.iloc[index, 2] > treshold:
            duplicate.append(data.iloc[index, 0])
            duplicate.append(data.iloc[index, 1])

    final_list = []
    list_duplicates = []

    for item in duplicate:
        if item not in final_list:
            final_list.append(item)
        elif item not in list_duplicates:
            list_duplicates.append(item)

    num_duplicates = len(list_duplicates)+1
    print("There are {} features that have a correlation greater than {} \
     with more than one other feature\n"
          .format(num_duplicates, treshold))

    for item in list_duplicates:
        final_list.remove(item)

    print("Features we are going to remove:\n")
    for item in list_duplicates:
        print(item)

    print("\n")
    purged_data = remove_features(data, list_duplicates)
    return purged_data, list_duplicates


def random_features(data, treshold, number=None):
    random.seed(5)
    new_list = []  # this list will contain the names of features we want to keep
    max_index = len(data)
    if number is None:
        for index in range(max_index):
            if data.iloc[index, 2] > treshold:
                coin_toss = random.randint(2)
                new_list.append(data.iloc[index, coin_toss])
    else:
        for index in range(number):
            if data.iloc[index, 2] > treshold:
                coin_toss = random.randint(2)
                new_list.append(data.iloc[index, coin_toss])

    num_feat = len(new_list)+1
    print("There are {} features pairings with a correlation greater than {}.\n"
          .format(num_feat, treshold))

    print("Features we are going to remove:\n")
    for item in new_list:
        print(item)

    purged_data = remove_features(data, new_list)

    return purged_data, new_list


def main(treshold):
    start = time.time()
    correlated_features = pd.read_csv(
        "../JaneStreetMkt/Matrices/features_to_remove.csv", header=None)
    no_duplicates, feature_removed = remove_duplicates(correlated_features, treshold)
    final_matrix, random_feat = random_features(no_duplicates, treshold)
    for item in random_feat:
        feature_removed.append(item)

    num_removed = len(feature_removed)+1
    print("Total number of feature removed:\n{}".format(num_removed))

    mins = (time.time()-start)//60
    sec = (time.time()-start) % 60
    print('Time to execute the feature selection model is time is: {} min {:.2f} sec\n'
          .format(mins, sec))
    return feature_removed
