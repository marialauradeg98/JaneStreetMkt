import numpy as np


def split(data, no_fill=False,val=False):
    """
    This function splits the dataset in test set and training set or in test set,
    training set and validation set.
    In each case we leave a gap between training set anda test set to prevent
    information leakage.
    Parameters
    ----------
    data: DataFrame
        The dataset we want to split
    no_fill: bool, default=False
        If False replace missing data with outliers, if True not.
    gap: float, default=10
        Gap in terms of percentage of total trading days considered.
    val: bool, default= False
        If False we divide the dataset in test set and training set, if True
        we also built a validation set.
    Yields
    ------
    X_train: DataFrame
        Training set without class label
    y_train: np.array
        Class label for training set
    X_test: DataFrame
        Test set without class label
    y_test: np.array
        Class label for test set
    X_val: DataFrame
        Validation set without class label
    y_test: np.array
        Class label for validation set
     """
    if no_fill is False:
        # fill missing values with outliers
        data = data.fillna(-999)
    #find the number of trading day
    num_days = data["date"].iloc[-1]
    #Compute the gap between the training and test set
    gap = num_days*5//100

    if val is False:
        # split training and test set with our choosen percentage
        days_train = num_days*4//5
        days_test = days_train+gap
        print("Trainig set begins at day 0 and ends at day {}".format(days_train))
        print("Test set begins at day {} and ends at day {}".format(days_test, num_days))
        data_test = data[data["date"] >= days_test]
        data_train = data[data["date"] <= days_train]
    else:
        #split training,test and validation set with our choosen percentage
        days_train = num_days*65//100
        days_val = (days_train + gap, days_train + gap+num_days//10)
        days_test = days_val[1]+gap
        print("trainig set begins at day 0 and ends at day {}".format(days_train))
        print("validation set begins at day {} and ends at day {}".format(days_val[0], days_val[1]))
        print("test set begins at day {} and ends at day {}".format(days_test, num_days))
        data_test = data[data["date"] >= days_test]
        data_train = data[data["date"] <= days_train]
        data_val = data[data["date"] >= days_val[0]]
        data_val = data_val[data_val["date"] <= days_val[1]]
        # remove the labels from val features
        X_val = data_val.drop(["action"], axis=1)
        # create an array which rapresent the class label
        y_val = np.ravel(data_val.loc[:, ["action"]])
        y_val = np.array(y_val)

    # remove the labels from the train features
    X_train = data_train.drop(["action"], axis=1)
    # create an array the rapresent the class label
    y_train = np.ravel(data_train.loc[:, ["action"]])
    y_train = np.array(y_train)
    # remove the labels from the test features
    X_test = data_test.drop(["action"], axis=1)
    # create an array the rapresent the class label
    y_test = np.ravel(data_test.loc[:, ["action"]])
    y_test = np.array(y_test)

    if val is False:
        return X_train, y_train, X_test, y_test
    else:
        return X_train, y_train,X_test, y_test, X_val, y_val
