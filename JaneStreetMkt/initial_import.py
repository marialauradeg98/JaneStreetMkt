"Two functions used to import the competition dataset"
import time
import pandas as pd
import datatable as dt
import numpy as np


def compute_action(d_frame: pd.DataFrame):
    """
    This functions add the action and the weighted resp to the dataset
    Action is equal to 1 when resp is > 0 and 0 otherwise
    Weighted resp is the product between resp and weights
    """
    d_frame["action"] = ((d_frame["resp"]) > 0) * 1  # action
    d_frame["weighted_resp"] = d_frame["resp"]*d_frame["weight"]
    d_frame["date"] = d_frame["date"]+1  # we add 1 to each day so we don't start from day 0
    values = d_frame["action"].value_counts()
    print("Values of action are so distributed:\n{}\n".format(values))
    return d_frame


def import_dataset(rows=None):
    """
    This fuction imports the Jane Market dataset as a pandas dataframe
    Inputs: rows(int) number of rows to import (default=None all roes will be imported)
    """
    start = time.time()
    if rows is None:
        data = pd.read_csv("../../jane-street-market-prediction/train.csv",
                           dtype=np.float32)  # load dataset
    data = pd.read_csv("../../jane-street-market-prediction/train.csv",
                       nrows=rows, dtype=np.float32)
    print("Train size: {}".format(data.shape))  # print number of rows and columns
    new_data = compute_action(data)  # add action and weighted resp
    # compute execution time
    mins = (time.time()-start)//60
    sec = (time.time()-start) % 60
    print('Execution time is: {} min {:.2f} sec\n'.format(mins, sec))
    return new_data


def import_dataset_faster():
    """
    This fuction imports the Jane Market dataset Jane Street dataset using datatable
    and then converts it into a pandas dataframe.
    This way should be faster
    """
    start = time.time()  # get starttime
    data_dt = dt.fread("../../jane-street-market-prediction/train.csv",
                       dtype=np.float32)  # load the dataset
    data = data_dt.to_pandas()  # converting to pandas dataframe
    print("Train size: {}".format(data.shape))  # print number of rows and columns
    new_data = compute_action(data)  # add action and weighted resp
    # compute execution time
    mins = (time.time()-start)//60
    sec = (time.time()-start) % 60
    print('Execution time is: {} min {:.2f} sec\n'.format(mins, sec))
    return new_data


def logic(index: int, num: int):
    """
    Used for slicing in import_sampled_dataset
    """
    if index % num != 0:
        return True
    return False


def import_sampled_dataset(skip: int, rows=None):
    """
    This function load a sampleed version of the original dataset.
    We sample a value every n*skip rows.
    Inputs: rows(int) = number of rows to import (default=None all rows will be imported)
            skip(int) =number of rows between each chosen sample
    """
    start = time.time()
    # load dataset
    if rows is None:
        data = pd.read_csv("../../jane-street-market-prediction/train.csv",
                           skiprows=lambda x: logic(x, skip), dtype=np.float32)
    data = pd.read_csv("../../jane-street-market-prediction/train.csv",
                       skiprows=lambda x: logic(x, slice), nrows=rows, dtype=np.float32)
    print("Train size: {}".format(data.shape))  # print number of rows and columns
    new_data = compute_action(data)  # add action and weighted resp
    # compute execution time
    mins = (time.time()-start)//60
    sec = (time.time()-start) % 60
    print('Execution time is: {} min {:.2f} sec\n'.format(mins, sec))
    return new_data


def import_training_set():
    """
    This fuction imports the Jane Market dataset as a pandas dataframe
    Inputs: rows(int) number of rows to import (default=None all roes will be imported)
    """
    # load the first 400 days of data the last days will be used as a test set
    # let the user decide which import to use
    PCFLAG = False
    while PCFLAG is False:
        pc = input("Done :) \nDo you have a good computer?\ny/n\n")
        if pc == "y":
            data = import_dataset_faster()
            PCFLAG = True
        elif pc == "n":
            data = import_dataset(10000)
            PCFLAG = True
        else:
            print("Please enter valid key\n")
    data = data[data["date"] < 400]
    # Delete the resps' values from training set
    training_data = data.drop(["resp", "resp_1", "resp_2", "resp_3",
                               "resp_4", "weighted_resp"], axis=1)
    # compute execution time
    return training_data


def import_test_set():
    """
    This fuction imports the Jane Market dataset as a pandas dataframe
    Inputs: rows(int) number of rows to import (default=None all roes will be imported)
    """
    # load the last days of data as test set there is a gap of 25 days between test
    # and training
    PCFLAG = False
    while PCFLAG is False:
        pc = input("Done :) \nDo you have a good computer?\ny/n\n")
        if pc == "y":
            data = import_dataset_faster()
            PCFLAG = True
        elif pc == "n":
            data = import_dataset()
            PCFLAG = True
        else:
            print("Please enter valid key\n")
    data = data[data["date"] > 425]
    # Delete the resps' values from training set
    test_data = data.drop(["resp", "resp_1", "resp_2", "resp_3", "resp_4", "weighted_resp"])
    # compute execution time
    mins = (time.time()-start)//60
    sec = (time.time()-start) % 60
    print('Execution time is: {} min {:.2f} sec\n'.format(mins, sec))
    return test_data


def main():
    FLAG = False  # used to make sure to go back once an invalid string is entered
    while FLAG is False:
        # reads the input from keyboard to select what to do
        value = input(
            "Hello what dataset do you want to import? \n1)Entire dataset \
            \n2)Sampled dataset\n3)Small dataset\n4)Training set\n")
        if (value) == "1":
            PCFLAG = False
            while PCFLAG is False:
                pc = input("Done :) \nDo you have a good computer?\ny/n\n")
                if pc == "y":
                    data = import_dataset_faster()
                    PCFLAG = True
                elif pc == "n":
                    data = import_dataset()
                    PCFLAG = True
                else:
                    print("Please enter valid key\n")
            print("Importing entire dataset...\n")
            FLAG = True
        elif (value) == "2":
            print("Importing sampled dataset...\n")
            data = import_sampled_dataset(20)
            FLAG = True
        elif (value) == "3":
            rows = input("How many rows do you want to import?\n")
            print("Importing small dataset...\n")
            data = import_dataset(int(rows))
            FLAG = True
        elif (value) == "4":
            print("Importing training set...\n")
            data = import_training_set()
            FLAG = True
        else:
            print("Please enter valid key\n \n")
    return(data)
