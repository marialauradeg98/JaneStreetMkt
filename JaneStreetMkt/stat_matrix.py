""" Module used to create a matrix containing useful statistical informations"""

import matplotlib.pyplot as plt
from initial_import import import_dataset
from col_histogram import col_histogram
import numpy as np
import pandas as pd


def statistical_matrix(df):
    """ creating matrix using the function df.describe () """
    df = df.drop(["date", "ts_id"], axis=1)  # delete date and date id from columns
    new_matrix = df.describe()  # create a new matrix with useful informations like mean, max etc
    return new_matrix


def daily_avarage(df):
    """ compute the daily avarage of each feature in the dataset"""
    day_mean = df.groupby("date").mean()
    return day_mean


def compute_profit(df):
    """ return number of days of trading profit u and t  """
    days = data.loc[:, "date"].iat[-1]  # find the last day of trading

    # compute a Pandas serie p_i from the original dataset
    p_i = df.loc[:, ["weighted_resp", "date", "action"]]
    p_i["weighted_resp"] = p_i["action"]*p_i["weighted_resp"]
    p_i = p_i.groupby("date").sum()  # sum for each day
    p_i = p_i.loc[:, "weighted_resp"]  # discard other colums

    # now we compute t and u
    t = p_i.sum()/np.sqrt((p_i**2).sum())*np.sqrt(250/days)
    u = min(max(t, 0), 6)*p_i.sum()
    return (days, u, t)


if __name__ == '__main__':
    data = import_dataset()  # import competion dataset
    data["date"] = data["date"]+1  # days would start with 0 otherwise
    data["weighted_resp"] = data["resp"]*data["weight"]  # add weighted resp to matr[ix
    data["action"] = ((data["resp"]) > 0)*1  # compute action
    actions = data["action"].value_counts()  # count number of actions
    print("the number of 1 is {} the number of 0 is {}".format(
        actions.iat[0], actions.iat[1]))
    stats = statistical_matrix(data)  # creating the stats matrix
    # stats.to_csv("stats_complete.csv")  # save new matrix as csv
    mean_matrix = daily_avarage(data)  # compute daily mean of each feature

    '''
    maybe i can do this later
    stats.to_csv("stats.csv")  # save new matrix as csv
    print(stats)
    print(mean_matrix["resp"])
    col_histogram(data["resp"], 60)


    # plotting daily avarage cumulative sums of resps
    plt.title("Cumulative sum of resps")
    plt.xlabel("Days")
    plt.ylabel("Resp")
    plt.plot(mean_matrix["resp"].cumsum(), lw=3, label="resp")
    plt.plot(mean_matrix["resp_1"].cumsum(), lw=3, label="resp_1")
    plt.plot(mean_matrix["resp_2"].cumsum(), lw=3, label="resp_2")
    plt.plot(mean_matrix["resp_3"].cumsum(), lw=3, label="resp_3")
    plt.plot(mean_matrix["resp_4"].cumsum(), lw=3, label="resp_4")
    plt.plot(mean_matrix["weighted_resp"].sum(), lw=3, label="weighted_resp")
    plt.legend()
    plt.show()
    '''

    days, u_val, t_val = compute_profit(data)
    print("We get a value of t {:.3f} after {} days of trading. \n the expected return is {:.3f}".format(
        t_val, days, u_val))

    # compute the correlation matrix between two days
    data.fillna(0)  # fill missing data with 0
    day1_data = data.loc[data["date"] == 1]
    day10_data = data.loc[data["date"] == 10]
    day1_and_10 = pd.concat([day1_data, day10_data])
    # corr_matrix = day1_and_10.corr(method='pearson').style.background_gradient(
    # cmap='coolwarm', axis=None).set_precision(2)
    corr_matrix = day1_and_10.corr(method='pearson')
    corr_matrix = corr_matrix[(corr_matrix > 0.5) | (corr_matrix < -0.5)]
    corr_matrix.to_csv("corr_matrix_day_1_and_10.csv")
