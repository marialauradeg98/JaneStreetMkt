"""
In this module are implemented many function to import the competition dataset
as a pandas dataframe.
Most of those functions are only used in the visualization part of the code.
"""
import time
import pandas as pd
import datatable as dt
import numpy as np


def compute_action(d_frame):
    """
    This functions add the action and the weighted resp to the dataset.
    Action is equal to 1 when resp is > 0 and 0 otherwise.
    Weighted resp is the product between resp and weights.

    Parameters
    ----------
    d_frame: DataFrame
        the competition DataFrame

    Yields
    ------
    d_frame: DataFrame
        the competition datafreme with action and weighted_resp added
    """
    # add action to the dataframe
    d_frame["action"] = ((d_frame["resp"]) > 0) * 1
    # add weighted_resp
    d_frame["weighted_resp"] = d_frame["resp"]*d_frame["weight"]
    # we add 1 to each day so we don't start from day 0
    d_frame["date"] = d_frame["date"]+1
    # nice prints
    values = d_frame["action"].value_counts()
    print("Values of action are so distributed:\n{}\n".format(values))
    return d_frame


def import_dataset(rows=None, filepath=None):
    """
    This fuction imports the Jane Market dataset as a pandas dataframe.
    Each value in the dataframe is imported as to float32 to reduce memory usage.
    To import the datset the pandas function read_csv is used.
    It gets the job done but it is a little bit slow.

    Parameters
    ----------
    rows: int (default=None)
        number of rows we want to import.
    filepath: str (default=the filepath in my pc :) )
        filepath where the file train.csv is located.


    Yields
    ------
    new_data: DataFrame
        the entire dataset ready to use
    """
    start = time.time()
    print("Importing  dataset...\n")
    if filepath is None:
        if rows is None:
            data = pd.read_csv("../../jane-street-market-prediction/train.csv",
                               dtype=np.float32)
        else:
            data = pd.read_csv("../../jane-street-market-prediction/train.csv",
                               nrows=rows, dtype=np.float32)
    else:
        if rows is None:
            data = pd.read_csv(filepath, dtype=np.float32)
        else:
            data = pd.read_csv(filepath, nrows=rows, dtype=np.float32)

    print("Train size: {}".format(data.shape))  # print number of rows and columns
    new_data = compute_action(data)  # add action and weighted resp
    # compute execution time
    mins = (time.time()-start)//60
    sec = (time.time()-start) % 60
    print('Time to import the dataset is : {} min {:.2f} sec\n'.format(mins, sec))
    return new_data


def import_dataset_faster(rows=None, filepath=None):
    """
    This fuction imports the Jane Market dataset as a pandas dataframe.
    To import the datset the datatable function fread is used.
    Then the data is converted into a dataframe.
    This approach is significantly faster.

    Parameters
    ----------
    rows: int (default=None)
        number of rows we want to import.
    filepath: str (default=the filepath in my pc :) )
        filepath where the file train.csv is located.

    Yields
    ------
    new_data: DataFrame
        the entire dataset ready to use
    """
    start = time.time()  # get starttime
    print("Importing dataset...\n")
    if filepath is None:
        if rows is None:
            data_dt = dt.fread("../../jane-street-market-prediction/train.csv")
        else:
            data_dt = dt.fread("../../jane-street-market-prediction/train.csv", max_nrows=rows)
    else:
        if rows is None:
            data_dt = dt.fread(filepath)
        else:
            data_dt = dt.fread(filepath, max_nrows=rows)

    data = data_dt.to_pandas()  # converting to pandas dataframe
    print("Train size: {}".format(data.shape))  # print number of rows and columns
    new_data = compute_action(data)  # add action and weighted resp
    # compute execution time
    mins = (time.time()-start)//60
    sec = (time.time()-start) % 60
    print('Time to import the dataset is : {} min {:.2f} sec \n'.format(mins, sec))
    return new_data


def logic(index: int, num: int):
    """
    Used for slicing in import_sampled_dataset
    """
    if index % num != 0:
        return True
    return False


def import_sampled_dataset(skip, rows=None, filepath=None):
    """
    This function load a sampleed version of the original dataset.
    We sample a value every n*skip rows.
    This function is used only in the visualization module since linear sampling
    in noisy dataset is not advised.

    Parameters
    ----------
    skip: int
        sample a row for each multiple of skip.
    rows: int (default=None all rows will be imported)
        number of rows to import.
    filepath: str (default=the filepath in my pc :) )
        filepath where the file train.csv is located.

    Yields
    ------
    new_data: DataFrame
        sampled dataset
    """
    start = time.time()
    print("Importing sampled dataset...\n")
    if filepath is None:
        if rows is None:
            data = pd.read_csv("../../jane-street-market-prediction/train.csv",
                               skiprows=lambda x: logic(x, skip), dtype=np.float32)
        else:
            data = pd.read_csv("../../jane-street-market-prediction/train.csv",
                               skiprows=lambda x: logic(x, skip), nrows=rows, dtype=np.float32)
    else:
        if rows is None:
            data = pd.read_csv(filepath,
                               skiprows=lambda x: logic(x, skip), dtype=np.float32)
        else:
            data = pd.read_csv(filepath,
                               skiprows=lambda x: logic(x, skip), nrows=rows, dtype=np.float32)

    print("Train size: {}".format(data.shape))  # print number of rows and columns
    new_data = compute_action(data)  # add action and weighted resp
    # compute execution time
    mins = (time.time()-start)//60
    sec = (time.time()-start) % 60
    print('Time to import sampled dataset: {} min {:.2f} sec\n'.format(mins, sec))
    return new_data


def import_training_set(fast_pc=False, rows=None):
    """
    This is the import function we will call the most in the rest of the code.
    It imports the Jane Market dataset as a pandas dataframe and removes the
    6 resps features from the dataset since the competition test set
    will not have those features.

    Parameters
    ----------
    fast_pc: bool (default=False)
        False use read_csv to import data
        True use fred to import data

    Yields
    ------
    training_data: DataFrame
        dataset without resps
    """
    # load the first 400 days of data the last days will be used as a test set
    # let the user decide which import to use

    if fast_pc is True:
        if rows is None:
            data = import_dataset_faster()
        else:
            data = import_dataset_faster(rows)
    else:
        if rows is None:
            data = import_dataset()
        else:
            data = import_dataset(rows)

    # Delete the resps' values from training set
    training_data = data.drop(["resp", "resp_1", "resp_2", "resp_3",
                               "resp_4", "weighted_resp"], axis=1)

    return training_data


def main():
    """
    This function implements a interactive way to import the import the dataset.
    It reads the keabord inputs of the user to decide wich function to use to
    import the dataset.
    Parameters
    ----------
    None

    Yields
    ------
    data: DataFrame
        competition dataset
    """

    flag = False  # used to make sure to go back once an invalid string is entered
    while flag is False:
        # reads the input from keyboard to select what to do
        value = input(
            "Hello what dataset do you want to import? \n1)Entire dataset \
            \n2)Sampled dataset\n3)Small dataset\n4)Training set\n")
        if (value) == "1":
            pcflag = False
            while pcflag is False:
                fast_pc = input("Do you have a good computer?\ny/n\n")
                if fast_pc == "y":
                    data = import_dataset_faster()
                    pcflag = True
                elif fast_pc == "n":
                    data = import_dataset()
                    pcflag = True
                else:
                    print("Please enter valid key\n")
            flag = True
        elif (value) == "2":
            data = import_sampled_dataset(20)
            flag = True
        elif (value) == "3":
            rows = input("How many rows do you want to import?\n")
            data = import_dataset(int(rows))
            flag = True
        elif (value) == "4":
            data = import_training_set()
            flag = True
        else:
            print("Please enter valid key\n \n")
    return data
