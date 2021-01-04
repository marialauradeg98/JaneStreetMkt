"""
Main module of the data visualization stage, the main goal of this module is to
plot different graphs and matrixes wich can be useful to better understand the data.
The focal points of this visual analysis are:
1) Correlation analysis
2) Missing values analysis
3) Behaviour's analysis of the features over time
4) Analysis of the distribution of the dataset

This is just a preliminary stage more in depth analysis will be carried later.
"""

import matplotlib.pyplot as plt
from initial_import import import_dataset
import numpy as np
import pandas as pd
import time
import progressbar


def statistical_matrix(df: pd.DataFrame):
    """
    This function returns a Pandas dataframe containing useful statistical informations
    about the dataset like mean, median, max , min etc
    """
    df = df.drop(["date", "ts_id"], axis=1)  # delete date and date id from columns
    new_matrix = df.describe()  # create a new matrix with useful informations like mean, max etc
    return new_matrix


def daily_avarage(df: pd.DataFrame):
    """
    This function computes the daily avarage of each feature in a Pandas dataframe
    """
    day_mean = df.groupby("date").mean()
    return day_mean


def compute_profit(df: pd.DataFrame):
    """
    This function compute the maximum possible profit u, the maximum value of t and the number of days elapsed.
    To compute this value we take an action everytime we have resp > 0, then we
    apply the formulas provided in the competition evaluation page.
    """
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


def corrFilter(x: pd.DataFrame, bound: float):
    """
    This function
    """
    xCorr = x.corr()
    xFiltered = xCorr[((xCorr >= bound) | (xCorr <= -bound)) & (xCorr != 1.000)]
    xFlattened = xFiltered.unstack().sort_values().drop_duplicates()
    return xFlattened


if __name__ == '__main__':
    start = time.time()  # useful to compute time to execute module
    '''
    big failure
    for i in progressbar.progressbar(range(100)):
        time.sleep(0.2)
        data = import_dataset()  # import competion dataset
    '''
    data = import_dataset()  # import competion dataset
    #data = import_sampled_dataset(100)
    # compute useful matrixs used later in the code
    stats = statistical_matrix(data)  # creating the stats matrix
    stats.round(3).to_csv("Matrices/stats_complete.csv")  # save new matrix as csv
    mean_matrix = daily_avarage(data)  # compute daily mean of each feature

    # plotting daily avarage cumulative sums of resps
    print("Working on plot of main features over time...\n")
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
    plt.savefig("Figures/cumsum_resps", dpi=300)
    plt.clf()
    print("Done :)\n")

    # compute the maximum value of u possible
    days, u_val, t_val = compute_profit(data)
    print("If we take an action every time we have resp > 0 \nwe get a value of t {: .3f} after {} days of trading. \nThe expected maximum return is {: .3f}.\n".format(
        t_val, days, u_val))

    # divide dataset in two parts one with the most relevant features (resps,weiights etc)
    # and the other with the anonymus features
    data_main = data.loc[:, ["resp", "resp_1", "resp_2",
                             "resp_3", "resp_4", "weight", "weighted_resp", "action", "date"]]
    data_anon = data.drop(["resp", "resp_1", "resp_2", "resp_3", "resp_4",
                           "weight", "weighted_resp", "action", "ts_id"], axis=1)

    # compute the correlation matrix between two hopefully uncorrelated days
    day1_data = data.loc[data["date"] == 1]
    day10_data = data.loc[data["date"] == 10]
    day1_and_10 = pd.concat([day1_data, day10_data])
    '''
    corr_matrix = day1_and_10.corr(method='pearson').style.background_gradient(
        cmap='coolwarm', axis=None).set_precision(2)
    '''
    # compute wich pairings has a correlation > 0.99, > 0.95 and save to csv
    corr_matrix_99 = corrFilter(day1_and_10, 0.99)
    corr_matrix_95 = corrFilter(day1_and_10, 0.95)
    (corr_matrix_99.round(3)).to_csv("Matrices/corr_99.csv")
    (corr_matrix_95.round(3)).to_csv("Matrices/corr_95.csv")

    # print number of pairings with correlation > 0.99, > 0.95
    features_99 = (corr_matrix_99.count()).sum()
    features_95 = (corr_matrix_95.count()).sum()
    print("The number of pairings with correlation > 0.99 is {}. \nThe number of pairings with correlation > 0.95 is {}.\n" .format(
        features_99, features_95))

    # print scatter plot correlated pairings
    print("Working on scatter plot most correlated features...\n")
    fig0, axes = plt.subplots(2, 2, figsize=(3, 3))
    axes[0, 0].scatter(x=data.loc[:, 'feature_60'],
                       y=data.loc[:, 'feature_61'], c="blue", marker=".")
    axes[0, 1].scatter(x=data.loc[:, 'feature_62'],
                       y=data.loc[:, 'feature_63'], c="red", marker=".")
    axes[1, 0].scatter(x=data.loc[:, 'feature_65'],
                       y=data.loc[:, 'feature_66'], c="yellow", marker=".")
    axes[1, 1].scatter(x=data.loc[:, 'feature_67'],
                       y=data.loc[:, 'feature_68'], c="green", marker=".")
    plt.title("Scatter plot higly correlated features")
    plt.savefig('Figures/scatter_correlation.png', dpi=300)
    plt.clf()
    print("Done :)\n")

    # missing data analysis
    miss_values = data.shape[0]-data.count()
    # select features with most missing values and plot a barplot
    miss_values = miss_values[(miss_values > data.count()*.005)]
    fig2 = miss_values.plot.bar(title='Features and missing values', fontsize=12, figsize=(3, 2))
    plt.savefig('Figures/missing_values_and_features.png', dpi=300)
    plt.clf()

    # plot the cumulative sum of anonymus features over time to see if there are patterns
    print("Working on plot of features over time...\n")
    mean_matrix_anon = daily_avarage(data_anon)
    fig3 = mean_matrix_anon.plot(subplots=True, layout=(
        68, 2), figsize=(3., 30.))
    plt.savefig('Figures/anonimous_features_over_time.png', dpi=300, bbox_inches="tight")
    plt.clf()

    # plot the cumulative sum of main features over time to see if there are patterns
    mean_matrix_anon = daily_avarage(data_main)
    fig4 = mean_matrix_anon.plot(subplots=True, layout=(
        4, 2), figsize=(3., 5.))
    plt.savefig('Figures/main_features_over_time.png', dpi=300, bbox_inches="tight")
    plt.clf()
    print("Done :)\n")

    # plot histogram of actions and features_0 the only categorical data
    print("Working on histograms of features...\n")
    fig5 = data.loc[:, ["action"]].plot.hist(legend=True, fontsize=18, figsize=(
        3, 2), bins=2, title="Histogram of actions")
    plt.savefig('Figures/histogram_actions.png', dpi=300, bbox_inches="tight")
    fig6 = data.loc[:, ["feature_0"]].plot.hist(legend=True, fontsize=18, figsize=(
        15, 10), bins=2, title="Histogram of features_0")
    plt.savefig('Figures/histogram_feature_0.png', dpi=300, bbox_inches="tight")
    plt.clf()

    # plot histogram of anonimous features
    figs, axs = plt.subplots(43, 3, figsize=(3, 30))
    column_names = data_anon.columns.values.tolist()  # create a list with the name of all the columns
    for i in range(43):
        for j in range(3):
            # +1 because we don't want to include date
            axs[i, j].hist(data_anon.iloc[:, 3*i+j+1].dropna(), bins=100)
            axs[i, j].legend(["{}".format(column_names[3*i+j+1])])  # label of each subplot
    plt.title("Histogram anonimous features", fontsize=20)
    plt.savefig('Figures/histogram_anon.png', dpi=300, bbox_inches="tight")

    # plot histograms main features
    fig8, axs = plt.subplots(3, 3, figsize=(3, 3))
    column_names = data_main.columns.values.tolist()  # create a list with the name of all the columns
    for i in range(2):
        for j in range(2):
            axs[i, j].hist(data_main.iloc[:, 3*i+j], bins=100)
            axs[i, j].legend(["{}".format(column_names[3*i+j])])
    plt.title("Histogram main features", fontsize=20)
    plt.savefig('Figures/histogram_main.png', dpi=300, bbox_inches="tight")
    print("Done :)\n")

    # plot boxplot of main features
    print("Working on boxplot of features...\n")
    fig9 = data_main.drop(["weight", "action", "date"], axis=1).plot(
        kind="box", grid=False, whis=(1, 99), meanline=True, vert=False, title="Boxplot main features", figsize=(3, 3))
    plt.savefig("Figures/boxplot_main", dpi=300)
    print("Done:)\n")

    print("The time to execute the whole module was {:.2f}".format(time.time()-start))
