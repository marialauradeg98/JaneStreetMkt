"Two functions used to import the competition dataset"
import time
import pandas as pd
import datatable as dt


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
        data = pd.read_csv("../../jane-street-market-prediction/train.csv")  # load dataset
    data = pd.read_csv("../../jane-street-market-prediction/train.csv", nrows=rows)
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
    data_dt = dt.fread("../../jane-street-market-prediction/train.csv")  # load the dataset
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
                           skiprows=lambda x: logic(x, skip))
    data = pd.read_csv("../../jane-street-market-prediction/train.csv",
                       skiprows=lambda x: logic(x, slice), nrows=rows)
    print("Train size: {}".format(data.shape))  # print number of rows and columns
    new_data = compute_action(data)  # add action and weighted resp
    # compute execution time
    mins = (time.time()-start)//60
    sec = (time.time()-start) % 60
    print('Execution time is: {} min {:.2f} sec\n'.format(mins, sec))
    return new_data
