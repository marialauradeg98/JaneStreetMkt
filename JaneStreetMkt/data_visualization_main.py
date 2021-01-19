import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import initial_import


def statistical_matrix(d_frame: pd.DataFrame):
    """
    This function returns a Pandas dataframe containing useful statistical informations
    about the dataset (mean, median, max , min etc...).
    Parameters
    ----------
    d_frame: Pandas dataframe
        The dataset we want to describe.
    Yields
    ------
    new_matrix: Pandas dataframe
        The dataframe in which we save all the information about our dataset.
    """
    # delete date and date id from columns
    d_frame = d_frame.drop(["date", "ts_id"], axis=1)
    # create a new matrix with all usefull information
    new_matrix = d_frame.describe()
    return new_matrix


def daily_avarage(d_frame: pd.DataFrame):
    """
    This function computes the daily avarage values of each feature.
    Parameters
    ----------
    d_frame: Pandas dataframe
        The dataset on which we compute our daily average quantities.
    Yields
    ------
    day_mean: Pandas dataframe
        The dataframe composed by the daily average values.
    """
    day_mean = d_frame.groupby("date").mean()
    return day_mean


def compute_profit(d_frame: pd.DataFrame):
    """
    This function compute the maximum possible profit in terms of the variables
    recommended in the competition evaluation page.
    Parameters
    ----------
    d_frame: Pandas dataframe
        The dataset in which we want to find the maximum profit
    Yields
    ------
    days: int
        Time range considered in terms of trading days
    val_u:
        Value of the maximum possible utility
    """
    #find the last day of trading
    days = int(d_frame.loc[:, "date"].iat[-1])
    # compute a Pandas series p_i from the original dataset
    p_i = d_frame.loc[:, ["weighted_resp", "date", "action"]]
    p_i["weighted_resp"] = p_i["action"]*p_i["weighted_resp"]
    p_i = p_i.groupby("date").sum()  # sum for each day
    p_i = p_i.loc[:, "weighted_resp"]  # discard other colums
    # compute t and u
    val_t = p_i.sum()/np.sqrt((p_i**2).sum())*np.sqrt(250/days)
    val_u = min(max(val_t, 0), 6)*p_i.sum()
    print("If we take an action every time we have resp > 0 .\n The expected utility is {: .3f} after {} days of traiding .\n".format(val_u,days))
    return (val_u,days)


def corr_filter(d_frame: pd.DataFrame, bound: float):
    """
    This function analyzes the correlation between the features and identifies
    the highly correlated features.
    Parameters
    ----------
    d_frame: Pandas dataframe
        The dateset on which we evaluate the features correlation.
    bound: float
        This bound identifies our correlation range.
    Yields
    ------
    data_flattened: Pandas series
        The series which contains feature pairings with a correlation >bound or <-bound.
    """
    #compute our correlations
    data_corr = d_frame.corr()
    #selct the correlation in the choosen range
    data_filtered = data_corr[((data_corr >= bound) | (
        data_corr <= -bound)) & (data_corr != 1.000)]
    # discard the other values
    data_flattened = data_filtered.unstack().sort_values().drop_duplicates()
    return data_flattened

def activity_choice():
    """
    This fuction prints a list of activity on the dataset and return a value
    corresponding to the choosen activity.
    """
    return input("What do you want to do? \n1)Compute statistical matrix \
                \n2)Plot main features over time \n3)Correlation analysis \
                \n4)Plot anonymous features over time \n5)Missing data analysis \
                \n6)Usefull histograms for main features \n7)Usefull histograms for anonymous features \
                \n 8)Boxplot for main features\n9)Exit programm\n")


def plot_main_features(data):
    """
    This function is used to plot main features over time in terms of cumultaive
    sum of resp.
    The user can decide if save or not the figures obtained.
    Parameters
    ----------
    data: Pandas dataframe
        The dataset we are working on.
    """
    # compute daily mean of each feature
    mean_features = daily_avarage(data)
    print("Plotting...\n")
    # plot daily avarage cumulative sums of resps
    plt.figure(figsize=(10, 5))
    plt.title("Cumulative sum of resps", fontsize=20)
    plt.xlabel("Days")
    plt.ylabel("Resp")
    plt.plot(mean_features["resp"].cumsum(), lw=3, label="resp")
    plt.plot(mean_features["resp_1"].cumsum(), lw=3, label="resp_1")
    plt.plot(mean_features["resp_2"].cumsum(), lw=3, label="resp_2")
    plt.plot(mean_features["resp_3"].cumsum(), lw=3, label="resp_3")
    plt.plot(mean_features["resp_4"].cumsum(), lw=3, label="resp_4")
    plt.plot(mean_features["weighted_resp"].sum(), lw=3, label="weighted_resp")
    plt.legend()
    save_oneplot_options("Figures/cumsum_resps.png")

def correlation_analysis(data):
    """
    This function is used to evaluate the correlation between the features and to
    build the scatter plots for the most correlated features.
    The user can decide if save or not the figures obtained.
    Parameters
    ----------
    data: Pandas dataframe
        The dataset we are working on.
    """
    # select highly correlated features (correlation> 0.9 or <-0.9)
    corr_matrix_90 = corr_filter(data, 0.90)
    #couting the number of higly correlated features
    features_90 = (corr_matrix_90.count()).sum()
    print("The number of pairings with correlation > 0.90 is {}.\n" .format(features_90))
    #sort the correlation matrix obtained before
    sorted_matrix = corr_matrix_90.sort_values(ascending=False)
    print(sorted_matrix)
    sorted_matrix.to_csv("Matrices/features_to_remove2.csv")
    #scatter plot most correlated features pairings
    print("Working on scatter plot most correlated features...\n")
    fig0, axes = plt.subplots(2, 2, figsize=(6, 6))
    fig0.suptitle("Scatter plot higly correlated features", fontsize=20)
    fig0.subplots_adjust(wspace=0.5, hspace=0.3)
    x = [data.loc[:, 'feature_60'], data.loc[:, 'feature_62'],
         data.loc[:, 'feature_65'], data.loc[:, 'feature_67']]
    y = [data.loc[:, 'feature_61'], data.loc[:, 'feature_63'],
         data.loc[:, 'feature_66'], data.loc[:, 'feature_68']]
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
    save_oneplot_options("Figures/scatter_correlation.png")

def plot_anon_features(data):
    """
    This function is used to plot the anonymous features over time (to see if
    there are some patterns). The user can decide if save or not the figures obtained.
    Parameters
    ----------
    data: Pandas dataframe
        The dataset we are working on.
    """
    # built matrix containing the anonymous features
    data_anon = data.drop(["resp", "resp_1", "resp_2", "resp_3", "resp_4",
                           "weight", "weighted_resp", "action", "ts_id"], axis=1)
    names = []  # empty list, it will store the names of the png files
    print("Working on plots of features over time...\n")
    mean_matrix_anon = daily_avarage(data_anon)
    # create 14 images 3x3 containig plot of each anonimous features
    for i in range(14):
        mean_matrix_anon.iloc[:, (9*i):(9*i+9)].plot(subplots=True, layout=(
            3, 3), figsize=(7., 7.))
        plt.subplots_adjust(wspace=0.4)
        plt.suptitle("Anonymous features over time", fontsize=20)
        # storing names of png files in case we want to save them
        names.append('Figures/anonimous_features_over_time{}.png'.format(i))
    save_plots_options(names)


def missing_data_analysis(data,threshold):
    """
    This function is used to identify features with a number of missing values
    over a threshold and build the relative bar plots.
    The user can decide if save or not the figure obtained.
    Parameters
    ----------
    data: Pandas dataframe
        The dataset we are working on.
    threshold: float
        Threshold choosen for the selection of features with most missing values.
    """
    # count number of missing data
    miss_values = data.shape[0]-data.count()
    # select features with missing values over a choosen threshold
    miss_values = miss_values[(miss_values > data.count()*threshold)]
    #plot the relative bar plot
    fig2 = miss_values.plot(kind="bar", fontsize=10, figsize=(10, 6))
    plt.title("Features with most missing values", fontsize=20)
    plt.subplots_adjust(bottom=0.2)
    save_oneplot_options("Figures/missing_data.png")

def hist_main_features(data):
    """
    This function is used to plot histograms about main features distributions,
    particularly the categorical features: action and feature 0.
    The user can decide if save or not the figures obtained.
    Parameters
    ----------
    data: Pandas dataframe
        The dataset we are working on.
    """
    # matrix containing only the most relevant features
    data_main = data.loc[:, ["resp", "resp_1", "resp_2",
                             "resp_3", "resp_4", "weight", "weighted_resp", "action", "date"]]
    # plot histogram of actions and features_0
    print("Working on histograms of action...\n")
    fig5 = data.loc[:, ["action"]].plot(kind="hist", legend=True, fontsize=18, figsize=(
        6, 6), bins=2, rwidth=0.8, range=[-.5, 1.5])
    plt.xticks([0, 1])
    plt.subplots_adjust(left=0.2, right=.9)
    plt.title("Histogram for action", fontsize=20)
    save_oneplot_options("Figures/histogram_action")
    print("Working on histograms of feature 0...\n")
    fig6 = data.loc[:, ["feature_0"]].plot(kind="hist", legend=True, fontsize=18, figsize=(
        6, 6), bins=2, rwidth=0.8, range=[-2, 2])
    plt.subplots_adjust(left=0.2, right=.9)
    plt.title("Histogram for feature 0", fontsize=20)
    plt.xticks([-1, 1])
    save_oneplot_options("Figures/histogram_feature_=0")
    # plot histograms main features
    print("Working on histograms main features...")
    # to better visualize the results we set the range of the histograms
    # equal to the mean variance between the features excluding weight
    # (very high variance)
    mean_var = data_main.drop(["weight"], axis=1).var().mean()
    fig8 = data_main.drop(["date", "action"], axis=1).plot(subplots=True, layout=(
        4, 2), figsize=(6., 6.), kind="hist", bins=100, yticks=[],
        range=([-mean_var, mean_var]))
    plt.suptitle("Histogram main features", fontsize=20)
    save_oneplot_options("Figures/histogram_main.png")

def hist_anon_features(data):
    """
    This function is used to plot histograms about anonymous features distributions.
    The user can decide if save or not the figures obtained.
    Parameters
    ----------
    data: Pandas dataframe
        The dataset we are working on.
    """
    # built matrix containing the anonymous features
    data_anon = data.drop(["resp", "resp_1", "resp_2", "resp_3", "resp_4",
                           "weight", "weighted_resp", "action", "ts_id"], axis=1)
    names = []  # empty list, it will store the names of the png files
    figs, axs = plt.subplots(3, 3, figsize=(6, 6))
    print("Working on plot the histograms of the anonymous features...\n")
    # create 14 images 3x3 containig plot of each anonymous features
    for i in range(14):
        data_hist = data_anon.iloc[:, (9*i):(9*i+9)]
        # to better visualize the results we set the range of each 3x3 histogram
        # equal to the max variance between the features
        max_var = data_hist.var().max()
        data_hist.plot(subplots=True, layout=(
            3, 3), figsize=(6., 6.), kind="hist", bins=100, yticks=[],
            range=([-max_var, max_var]))
        plt.subplots_adjust(wspace=0.4)
        plt.suptitle("Distributions for anonymous features", fontsize=20)
        # storing names of png files in case we want to save them
        names.append('Figures/histogram_anonimous_features{}.png'.format(i))
    save_plots_options(names)

def boxplot_main(data):
    """
    This function is used to build the boxplot of main features. The user
    can decide if save or not the figure obtained.
    Parameters
    ----------
    data: Pandas dataframe
        The dataset we are working on.
    """
    #matrix containing only the most relevant features
    data_main = data.loc[:, ["resp", "resp_1", "resp_2",
                             "resp_3", "resp_4", "weight", "weighted_resp", "action", "date"]]
    print("Working on boxplot of features...\n")
    #Plot th boxplot (outliers aren't shown)
    fig9 = data_main.drop(["weight", "action", "date"], axis=1).plot(
        kind="box", grid=False, whis=(10, 90), meanline=True, vert=False,
        figsize=(7, 6), sym="", label="wiskers rapresent I$_{10}$ and I$_{90}$")
    plt.subplots_adjust(left=0.2)
    plt.suptitle("Boxplot for main features")
    plt.legend()
    save_oneplot_options("Figures/boxplot_main.png")

def save_data_options(object,name_save):
    """
    This fuction is used for the save options of a dataframe as .cvs.
    Parameters
    ----------
    object: Pandas dataframe
        Dataframe we want to save or not.
    name_save: string
        Figure's name.
    """
    SAVEFLAG = False
    while SAVEFLAG is False:
        save = input("Done. \nDo you want to save it?\ny/n\n")
        if save == "y":
            # save new matrix as csv
            object.round(3).to_csv(name_save)
            print("Saved successfully as {}\n".format(name_save))
            SAVEFLAG = True
        elif save == "n":
            SAVEFLAG = True
        else:
            print("Please enter valid key.\n")


def save_oneplot_options(name_save):
    """
    This fuction is used for the save options and visualization of a single plot.
    Parameters
    ----------
    name_save: string
        Figure's name.
    """
    SAVEFLAG = False
    while SAVEFLAG is False:
        # reads from input if we need to save the plot
        save = input("Done. \nDo you want to save it?\ny/n\n")
        if save == "y":
            plt.savefig(name_save, dpi=300)
            print("Saved successfully as {}\n".format(name_save))
            SAVEFLAG = True
        elif save == "n":
            SAVEFLAG = True
        else:
            print("Please enter valid key.\n")
        plt.show()
        plt.close("all")

def save_plots_options(names_save):
    """
    This fuction is used for the save options and visualization of some plots.
    Parameters
    ----------
    names_save: list
        List in which we stored figures' names.
    """
    SAVEFLAG = False
    while SAVEFLAG is False:
        save = input("Done. \nDo you want to save them?\ny/n\n")
        if save == "y":
            for name in names_save:
                plt.savefig(name, dpi=300)
            print("Saved successfully.\n")
            SAVEFLAG = True
        elif save == "n":
            SAVEFLAG = True
        else:
            print("Please enter valid key.\n")
        plt.show()
        plt.close("all")


if __name__ == '__main__':
    # start time for the exection of this main
    start = time.time()
    # import the choosen dataset
    data = initial_import.main()
    # compute the maximum value of u possible
    u_val, days = compute_profit(data)
    #user window
    FLAG = False  # used to make sure to go back once an invalid string is entered
    while FLAG is False:
        value = activity_choice()
        if value == "1":
            # compute matrix containig useful statistical informations
            stats = statistical_matrix(data)
            save_data_options(stats,"Matrices/stats_complete.csv")
        elif value == "2":
            plot_main_features(data)
        elif value == "3":
            correlation_analysis(data)
        elif value == "4":
            plot_anon_features(data)
        elif value == "5":
            missing_data_analysis(data,.005)
        elif value == "6":
            hist_main_features(data)
        elif value == "7":
            hist_anon_features(data)
        elif value == "8":
            boxplot_main(data)
        elif value == "9":
            print("End session.\n")
            FLAG = True
        else:
            print("Please enter valid key\n")

    # compute execution time
    mins = (time.time()-start)//60
    sec = (time.time()-start) % 60
    print('Execution time is: {} min {:.2f} sec\n'.format(mins, sec))
