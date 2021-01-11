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

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import initial_import


def statistical_matrix(d_frame: pd.DataFrame):
    """
    This function returns a Pandas dataframe containing useful statistical informations
    about the dataset like mean, median, max , min etc
    """
    d_frame = d_frame.drop(["date", "ts_id"], axis=1)  # delete date and date id from columns
    # create a new matrix with useful informations like mean, max et
    new_matrix = d_frame.describe()
    return new_matrix


def daily_avarage(d_frame: pd.DataFrame):
    """
    This function computes the daily avarage of each feature in a Pandas dataframe
    """
    day_mean = d_frame.groupby("date").mean()
    return day_mean


def compute_profit(d_frame: pd.DataFrame):
    """
    This function compute the maximum possible profit u, the maximum value
    of t and the number of days elapsed.
    To compute this value we take an action everytime we have resp > 0, then we
    apply the formulas provided in the competition evaluation page.
    """
    days = d_frame.loc[:, "date"].iat[-1]  # find the last day of trading

    # compute a Pandas serie p_i from the original dataset
    p_i = d_frame.loc[:, ["weighted_resp", "date", "action"]]
    p_i["weighted_resp"] = p_i["action"]*p_i["weighted_resp"]
    p_i = p_i.groupby("date").sum()  # sum for each day
    p_i = p_i.loc[:, "weighted_resp"]  # discard other colums

    # now we compute t and u
    val_t = p_i.sum()/np.sqrt((p_i**2).sum())*np.sqrt(250/days)
    val_u = min(max(val_t, 0), 6)*p_i.sum()
    return (days, val_u, val_t)


def corr_filter(d_frame: pd.DataFrame, bound: float):
    """
    This function returns the feature pairings with a correlation >bound or <-bound
    """
    data_corr = d_frame.corr()
    data_filtered = data_corr[((data_corr >= bound) | (
        data_corr <= -bound)) & (data_corr != 1.000)]
    # discard other values
    data_flattened = data_filtered.unstack().sort_values().drop_duplicates()
    return data_flattened


if __name__ == '__main__':
    start = time.time()  # useful to compute time to execute module
    data = initial_import.main()

    # compute the maximum value of u possible
    day, u_val, t_val = compute_profit(data)
    print("If we take an action every time we have resp > 0 \nwe get a value of t {: .3f}\
          after {} days of trading. \nThe expected maximum return is {: .3f}.\n".format(
        t_val, day, u_val))

    # compute matrices useful for later

    mean_matrix = daily_avarage(data)  # compute daily mean of each feature

    # matrix containing only the most relevant features (resps,weights etc)
    data_main = data.loc[:, ["resp", "resp_1", "resp_2",
                             "resp_3", "resp_4", "weight", "weighted_resp", "action", "date"]]

    # matrix containing the anonimous features
    data_anon = data.drop(["resp", "resp_1", "resp_2", "resp_3", "resp_4",
                           "weight", "weighted_resp", "action", "ts_id"], axis=1)

    # main part
    FLAG = False  # used to make sure to go back once an invalid string is entered
    while FLAG is False:
        value = input("What do you want to do? \n1)Compute statistical matrix \
                    \n2)Plotting main features over time \n3)Correlation analysis \
                    \n4)Plotting anonimous features over time \n5)Missing data analysis \
                    \n6)histogram main features \n7)histogram anonimous features \
                    \n 8)boxplot main features\n9)Exit programm\n")

        if value == "1":
            # compute matrix containig useful statistical informations like mean,
            # median,max, min, etc of each feauture of the dataset
            stats = statistical_matrix(data)  # creating the stats matrix
            SAVEFLAG = False
            while SAVEFLAG is False:
                save = input("Done :) \nDo you want to save it?\ny/n\n")
                if save == "y":
                    # save new matrix as csv
                    stats.round(3).to_csv("Matrices/stats_complete.csv")
                    print("Done :)\n")
                    SAVEFLAG = True
                elif save == "n":
                    SAVEFLAG = True
                else:
                    print("Please enter valid key\n")

        elif value == "2":
            # plotting daily avarage cumulative sums of resps
            print("Working on plot of main features over time...\n")
            plt.figure(figsize=(10, 5))
            plt.title("Cumulative sum of resps", fontsize=20)
            plt.xlabel("Days")
            plt.ylabel("Resp")
            plt.plot(mean_matrix["resp"].cumsum(), lw=3, label="resp")
            plt.plot(mean_matrix["resp_1"].cumsum(), lw=3, label="resp_1")
            plt.plot(mean_matrix["resp_2"].cumsum(), lw=3, label="resp_2")
            plt.plot(mean_matrix["resp_3"].cumsum(), lw=3, label="resp_3")
            plt.plot(mean_matrix["resp_4"].cumsum(), lw=3, label="resp_4")
            plt.plot(mean_matrix["weighted_resp"].sum(), lw=3, label="weighted_resp")
            plt.legend()
            SAVEFLAG = False
            while SAVEFLAG is False:
                # reads from input if we need to save the plot
                save = input("Done :) \nDo you want to save it?\ny/n\n")
                if save == "y":
                    plt.savefig("Figures/cumsum_resps.png", dpi=300)
                    print("Done :)\n")
                    SAVEFLAG = True
                elif save == "n":
                    SAVEFLAG = True
                else:
                    print("Please enter valid key\n")
                plt.show()
                plt.close("all")

        elif value == "3":

            # select and delete features with correlation > 0.9 or <-0.9
            print("Working on computing highly correlated features\n")
            corr_matrix_90 = corr_filter(data, 0.90)
            features_90 = (corr_matrix_90.count()).sum()
            print("The number of pairings with correlation > 0.90 is {}.\n" .format(features_90))
            sorted_matrix = corr_matrix_90.sort_values(ascending=False)
            print(sorted_matrix)
            sorted_matrix.to_csv("Matrices/features_to_remove2.csv")

            # print scatter plot correlated pairings
            print("Working on scatter plot most correlated features...\n")
            fig0, axes = plt.subplots(2, 2, figsize=(6, 6))
            fig0.suptitle("Scatter plot higly correlated features", fontsize=20)
            fig0.subplots_adjust(wspace=0.5, hspace=0.3)
            # each element of the list is a column of the dataframe containig higly
            # correlated features
            x = [data.loc[:, 'feature_60'], data.loc[:, 'feature_62'],
                 data.loc[:, 'feature_65'], data.loc[:, 'feature_67']]
            y = [data.loc[:, 'feature_61'], data.loc[:, 'feature_63'],
                 data.loc[:, 'feature_66'], data.loc[:, 'feature_68']]
            # plotting
            axes[0, 0].scatter(x=x[0], y=y[0], marker=",", s=5)
            axes[0, 0].set_xlabel('Feature 60')
            axes[0, 0].set_ylabel('Feature 61')
            axes[0, 1].scatter(x=x[1], y=y[1], marker=",", s=5)
            axes[0, 1].set_xlabel('Feature 62')
            axes[0, 1].set_ylabel('Feature 63')
            axes[1, 0].scatter(x=x[2], y=y[2], marker=",", s=5)
            axes[1, 0].set_xlabel('Feature 65')
            axes[1, 0].set_ylabel('Feature 66')
            axes[1, 1].scatter(x=x[3], y=y[3], marker=",", s=5)
            axes[1, 1].set_xlabel('Feature 67')
            axes[1, 1].set_ylabel('Feature 68')
            SAVEFLAG = False
            while SAVEFLAG is False:
                save = input("Done :) \nDo you want to save it?\ny/n\n")
                if save == "y":
                    plt.savefig("Figures/scatter_correlation", dpi=300)
                    print("Done :)\n")
                    SAVEFLAG = True
                elif save == "n":
                    SAVEFLAG = True
                else:
                    print("Please enter valid key\n")
                plt.show()
                plt.close("all")

        elif value == "5":
            # missing data analysis
            miss_values = data.shape[0]-data.count()  # counts number of missing data
            # select features with missing values over a certain treshold and plot a barplot
            miss_values = miss_values[(miss_values > data.count()*.005)]
            fig2 = miss_values.plot(kind="bar", fontsize=10, figsize=(10, 6))
            plt.title("Features with most missing values", fontsize=20)
            plt.subplots_adjust(bottom=0.2)
            SAVEFLAG = False
            while SAVEFLAG is False:
                save = input("Done :) \nDo you want to save it?\ny/n\n")
                if save == "y":
                    plt.savefig("Figures/missing_data", dpi=300)

                    print("Done :)\n")
                    SAVEFLAG = True
                elif save == "n":
                    SAVEFLAG = True
                else:
                    print("Please enter valid key\n")
                plt.show()
                plt.close("all")

        elif value == "4":
            # plot the cumulative sum of anonymus features over time to see if there are patterns
            names = []  # empty list, it will store the names of the png files
            print("Working on plot of features over time...\n")
            mean_matrix_anon = daily_avarage(data_anon)
            # create 14 images 3x3 containig plot of each anonimous features
            for i in range(14):
                mean_matrix_anon.iloc[:, (9*i):(9*i+9)].plot(subplots=True, layout=(
                    3, 3), figsize=(7., 7.))
                plt.subplots_adjust(wspace=0.4)
                plt.suptitle("Features over time", fontsize=20)
                # storing names of png files in case we want to save them
                names.append('Figures/anonimous_features_over_time{}.png'.format(i))
            SAVEFLAG = False
            while SAVEFLAG is False:
                save = input("Done :) \nDo you want to save them?\ny/n\n")
                if save == "y":
                    for name in names:
                        plt.savefig(name, dpi=300)
                    print("Done :)\n")
                    SAVEFLAG = True
                elif save == "n":
                    SAVEFLAG = True
                else:
                    print("Please enter valid key\n")
                plt.show()
                plt.close("all")

        elif value == "6":
            # plot histogram of actions and features_0 the only categorical data
            print("Working on histograms of action...\n")
            fig5 = data.loc[:, ["action"]].plot(kind="hist", legend=True, fontsize=18, figsize=(
                6, 6), bins=2, rwidth=0.8, range=[-.5, 1.5])
            plt.xticks([0, 1])
            plt.subplots_adjust(left=0.2, right=.9)
            plt.title("Histogram of action", fontsize=20)
            SAVEFLAG = False
            while SAVEFLAG is False:
                save = input("Done :) \nDo you want to save it?\ny/n\n")
                if save == "y":
                    plt.savefig("Figures/histogram_action", dpi=300)
                    print("Done :)\n")
                    SAVEFLAG = True
                elif save == "n":
                    SAVEFLAG = True
                else:
                    print("Please enter valid key\n")
                plt.show()
                plt.close("all")

            print("Working on histograms of feature 0...\n")
            fig6 = data.loc[:, ["feature_0"]].plot(kind="hist", legend=True, fontsize=18, figsize=(
                6, 6), bins=2, rwidth=0.8, range=[-2, 2])
            plt.subplots_adjust(left=0.2, right=.9)
            plt.title("Histogram of feature 0", fontsize=20)
            plt.xticks([-1, 1])
            SAVEFLAG = False
            while SAVEFLAG is False:
                save = input("Done :) \nDo you want to save it?\ny/n\n")
                if save == "y":
                    plt.savefig("Figures/histogram_feature_=0", dpi=300)
                    print("Done :)\n")
                    SAVEFLAG = True
                elif save == "n":
                    SAVEFLAG = True
                else:
                    print("Please enter valid key\n")
                plt.show()
                plt.close("all")

            # plot histograms main features
            print("Working on histogram main features")
            # to better visualize the results we set the range of the histograms
            # equal to the mean variance between the features excluding weight
            # because it has a very high variance
            mean_var = data_main.drop(["weight"], axis=1).var().mean()
            fig8 = data_main.drop(["date", "action"], axis=1).plot(subplots=True, layout=(
                4, 2), figsize=(6., 6.), kind="hist", bins=100, yticks=[],
                range=([-mean_var, mean_var]))
            plt.suptitle("Histogram main features", fontsize=20)
            SAVEFLAG = False
            while SAVEFLAG is False:
                save = input("Done :) \nDo you want to save it?\ny/n\n")
                if save == "y":
                    plt.savefig("Figures/histogram_main.png", dpi=300)
                    print("Done :)\n")
                    SAVEFLAG = True
                elif save == "n":
                    SAVEFLAG = True
                else:
                    print("Please enter valid key\n")
                plt.show()
                plt.close("all")

        elif value == "7":
            # plot histogram of anonimous features
            names = []  # empty list, it will store the names of the png files
            figs, axs = plt.subplots(3, 3, figsize=(6, 6))
            print("Working on plot the histogram of the anonimous features...\n")
            # create 14 images 3x3 containig plot of each anonimous features
            for i in range(14):
                data_hist = data_anon.iloc[:, (9*i):(9*i+9)]
                # to better visualize the results we set the range of each 3x3 histogram
                # equal to the max variance between the features
                max_var = data_hist.var().max()
                data_hist.plot(subplots=True, layout=(
                    3, 3), figsize=(6., 6.), kind="hist", bins=100, yticks=[],
                    range=([-max_var, max_var]))
                plt.subplots_adjust(wspace=0.4)
                plt.suptitle("distribution", fontsize=20)
                # storing names of png files in case we want to save them
                names.append('Figures/histogram_anonimous_features{}.png'.format(i))
            SAVEFLAG = False
            while SAVEFLAG is False:
                save = input("Done :) \nDo you want to save them?\ny/n\n")
                if save == "y":
                    for name in names:
                        plt.savefig(name, dpi=300)
                    print("Done :)\n")
                    SAVEFLAG = True
                elif save == "n":
                    SAVEFLAG = True
                else:
                    print("Please enter valid key\n")
                plt.show()
                plt.close("all")

        elif value == "8":
            # plot boxplot of main features
            print("Working on boxplot of features...\n")
            # some consideraations about the boxplot:
            # the wiskers represent the I10 and I90
            # outliers aren't shown because they will clutter the graphs and
            # make it less readable
            fig9 = data_main.drop(["weight", "action", "date"], axis=1).plot(
                kind="box", grid=False, whis=(10, 90), meanline=True, vert=False,
                figsize=(7, 6), sym="", label="wiskers rapresent I$_{10}$ and I$_{90}$")
            plt.subplots_adjust(left=0.2)
            plt.suptitle("Boxplot main features")
            plt.legend()
            SAVEFLAG = False
            while SAVEFLAG is False:
                save = input("Done :) \nDo you want to save it?\ny/n\n")
                if save == "y":
                    plt.savefig("Figures/boxplot_main.png", dpi=300)
                    print("Done :)\n")
                    SAVEFLAG = True
                elif save == "n":
                    SAVEFLAG = True
                else:
                    print("Please enter valid key\n")
                plt.show()
                plt.close("all")

        elif value == "9":
            print("THX :)\n")
            FLAG = True
        else:
            print("Please enter valid key\n")

    # compute execution time
    mins = (time.time()-start)//60
    sec = (time.time()-start) % 60
    print('Execution time is: {} min {:.2f} sec\n'.format(mins, sec))
