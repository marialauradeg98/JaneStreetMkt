""" Module used to create a matrix containing useful statistical informations"""

import matplotlib.pyplot as plt
from initial_import import import_dataset
from col_histogram import col_histogram
import numpy as np
import pandas as pd


def statistical_matrix(df):
    """ creating matrix using the function df.describe () """
    df = df.drop(df.columns[[0, -1]], axis=1)  # delete date and date id from columns
    new_matrix = df.describe()  # create a new matrix with useful informations like mean, max etc
    return new_matrix


def daily_avarage(df):
    """ compute the daily avarage of each feature in the dataset"""
    day_mean = df.groupby("date").mean()
    return day_mean


def compute_profit(df, days):
    """ return profit u and t  """
    p_i = df.loc[:, ["weighted_resp", "date"]]  # create new matrix with only date and w_resp
    p_i = p_i.groupby("date").sum()  # sum for each day
    # now we compute t
    t = p_i.loc[0:days, ].sum()/np.sqrt((p_i.loc[0:days, ]**2).sum())*np.sqrt(250/days)
    t_fl = t.loc["weighted_resp"]  # from pandas Serie to float
    u = min(max(t_fl, 0), 6)*p_i.loc[0:days, ].sum()  # we compute t
    u_fl = u.loc["weighted_resp"]  # from pandas Serie to float
    return (t_fl, u_fl)


if __name__ == '__main__':
    data = import_dataset()  # import competion dataset
    data["date"] = data["date"]+1  # days would start with 0 otherwise
    data["weighted_resp"] = data["resp"]*data["weight"]  # add weighted resp to matrix
    stats = statistical_matrix(data)  # creating the stats matrix
    stats.to_csv("stats_complete.csv")  # save new matrix as csv
    mean_matrix = daily_avarage(data)  # compute daily mean of each feature

    '''
    maybe i can do this later
    stats.to_csv("stats.csv")  # save new matrix as csv
    print(stats)
    print(mean_matrix["resp"])
    col_histogram(data["resp"], 60)
    '''

    # plotting daily avarage cumulativesums of resps
    plt.title("Cumulative sum of resps")
    plt.xlabel("Days")
    plt.ylabel("Resp")
    plt.plot(mean_matrix["resp"].cumsum(), lw=4, label="resp")
    plt.plot(mean_matrix["resp_1"].cumsum(), lw=4, label="resp_1")
    plt.plot(mean_matrix["resp_2"].cumsum(), lw=4, label="resp_2")
    plt.plot(mean_matrix["resp_3"].cumsum(), lw=4, label="resp_3")
    plt.plot(mean_matrix["resp_4"].cumsum(), lw=4, label="resp_4")
    plt.plot(mean_matrix["weighted_resp"].sum(), lw=4, label="weighted_resp")
    plt.legend()
    plt.show()

    days = 15
    t_val, u_val = compute_profit(data, days)
    print("We get a value of t {} after {} days of trading. \n the expected return is {}".format(
        t_val, days, u_val))
