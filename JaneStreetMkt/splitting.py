import numpy as np


def test_train(data, no_fill=False):
    """
    This function splits test and training set leaving a gap between them to prevent
    information leakage
    Parameters
    ----------
    data: DataFrame
        data we want to split
    no_fill: bool default=False
        if False replace missing data with outliers
        if True doesn't replace them


    Yields
    ------
    X_train: DataFrame
        training dataset without class label
    y_train: np.array
        class label for training set
    X_test: DataFrame
        test dataset without class label
    y_test: np.array
        class label for test set
     """
    if no_fill is False:
        data = data.fillna(-999)  # fill missing values with outliers

    # access last day of trading
    num_days = data["date"].iloc[-1]

    # Divide the dataset in training and test set with a gap between them to prevent information leakage
    gap = num_days//10
    days_train = num_days*4//5
    days_test = days_train+gap
    print("trainig set begins at day 0 and ends at day {}".format(days_train))
    print("test set begins at day {} and ends at day {}".format(days_test, num_days))

    # split training and test set and leave a 50 days gap between them
    data_test = data[data["date"] >= days_test]
    data_train = data[data["date"] <= days_train]

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
    return X_train, y_train, X_test, y_test


def test_train_val(data):
    """
    This function splits test training and validation set leaving a gap between them to prevent
    information leakage
    Parameters
    ----------
    data: DataFrame
        data we want to split
    no_fill: bool default=False
        if False replace missing data with outliers
        if True doesn't replace them


    Yields
    ------
    X_train: DataFrame
        training dataset without class label
    y_train: np.array
        class label for training set
    X_val: DataFrame
        validation dataset without class label
    y_val: np.array
        class label for validation set

    X_test: DataFrame
        test dataset without class label
    y_test: np.array
        class label for test set
     """
    if no_fill is False:
        data = data.fillna(-999)  # fill missing values with outliers

    # access last day of trading
    num_days = data["date"].iloc[-1]

    # Divide the dataset in training, validation and test set with a gap between them
    # to prevent information leakage
    gap = num_days//10
    days_train = num_days//2
    days_val = (days_train + gap, days_train + gap+num_days//4)
    days_test = days_val[1]+gap
    print("trainig set begins at day 0 and ends at day {}".format(days_train))
    print("validation set begins at day {} and ends at day {}".format(days_val[0], days_val[1]))
    print("test set begins at day {} and ends at day {}".format(days_test, num_days))

    # remove useless info
    data_test = data[data["date"] >= days_test]
    data_train = data[data["date"] <= days_train]
    data_val = data[data["date"] >= days_val[0]]
    data_val = data_val[data_val["date"] <= days_val[1]]

    # remove the labels from train features
    X_train = data_train.drop(["action"], axis=1)
    # create an array which rapresent the class label
    y_train = np.ravel(data_train.loc[:, ["action"]])
    y_train = np.array(y_train)

    # remove the labels from val features
    X_val = data_val.drop(["action"], axis=1)
    # create an array which rapresent the class label
    y_val = np.ravel(data_val.loc[:, ["action"]])
    y_val = np.array(y_val)

    # remove the labels from  test features
    X_test = data_test.drop(["action"], axis=1)
    # create an array which rapresent the class label
    y_test = np.ravel(data_test.loc[:, ["action"]])
    y_test = np.array(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test
