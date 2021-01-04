"Two functions used to import the competition dataset"
import time
import pandas as pd
import datatable as dt


def compute_action(df):
    df["action"] = (df["resp"] > 0) * 1
    df["weighted_resp"] = df["resp"]*df["weight"]
    print("number of acton = 1 is {} \n number of acton = 0 is {} ".format(
        (df["action"] == 1).count(), (df["action"] == 0).count()))
    return(df)


def import_dataset():
    """importing the Jane Market dataset as a pandas dataframe, quite slow"""
    start = time.time()
    data = pd.read_csv("../../jane-street-market-prediction/train.csv",
                       nrows=100000)  # load dataset
    print("Train size: {}".format(data.shape))
    print('Execution time is: {} s'.format(time.time()-start))  # compute execution time
    new_data = compute_action(data)
    # data.info()
    return new_data


def import_dataset_faster():
    """ import Jane Street dataset using datatable and then converts it into a pandas dataframe """
    start = time.time()  # get starttime
    data_dt = dt.fread("../../jane-street-market-prediction/train.csv")  # load the dataset
    data = data_dt.to_pandas()  # converting to pandas dataframe
    print("Train size: {}".format(data.shape))
    print('Execution time is: {} s'.format(time.time()-start))  # compute execution time
    return data
