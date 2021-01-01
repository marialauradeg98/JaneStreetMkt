""" Module used to create a matrix containing useful statistical informations"""


from initial_import import import_dataset
from col_histogram import col_histogram


def statistical_matrix(df):
    """ creating matrix using the function df.describe () """
    df = df.drop(df.columns[[0, -1]], axis=1)  # delete date and date id from columns
    new_matrix = df.describe()  # create a new matrix with useful informations like mean, max etc
    return new_matrix


def daily_avarage(df):
    """ compute the daily avarage of each feature in the dataset"""
    day_mean = df.groupby("date").mean()
    return day_mean


if __name__ == '__main__':
    data = import_dataset()  # import competion dataset
    stats = statistical_matrix(data)  # creating the new matrix
    # stats.to_csv("stats.csv")  # save new matrix as csv
    print(stats)
    '''
    i need to finish this part

    mean_matrix = daily_avarage(data)  # compute daily mean of each feature
    print(mean_matrix["resp"])
    col_histogram(data["resp"], 60)
    
    '''
