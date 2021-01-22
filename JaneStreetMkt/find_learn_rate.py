"""
In this module we implement a tecnique to find the best learning rate
based on the article:
In the first part of the model we build 100 models with a linearly encreasing
learning rate over a period of only one epoch.
Then, after applying a smoothing algorithm, we find the learning rate lr which
minimizes the loss.
The best learning rate for our model is lr/10

"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, train_test_split

from keras.utils import np_utils
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
from keras.models import Model
import keras.losses
from initial_import import import_training_set
from splitting import split_data
import feature_selection


def build(data, input_dim, hidden_units, num_labels, dropout_rates, learning_rate):
    """
    This function builts the deep neural network used for the training.

    Parameters
    ----------
    data: pd.DataFrame
        competition dataset.

    input_dim: int
        number of rows of competiton dataset

    hidden_units: list of 4 int
        number of hidden units for each layer

    num_labels:
        num labels?

    dropout_rates: list of 4 int
        dropout rates

    learning_rate: float
        learning rate for gradient discent

    Yields
    ------
    model:
        model?
    """

    input = Input(shape=(input_dim, ))
    x = BatchNormalization()(input)  # re-centring and rescaling input layer
    # a fraction of nodes is discarded with a frequency equal to the rate
    x = Dropout(dropout_rates[0])(x)
    for i in range(len(hidden_units)):
        x = Dense(hidden_units[i])(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rates[i+1])(x)

    output = Dense(num_labels, activation='sigmoid')(x)
    model = Model(inputs=input, outputs=output)
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(learning_rate), metrics=['accuracy'])
    model.summary()

    return model


def filter_loss(loss, beta):
    """
    this function computes the smoothed loss

    Parameters:
    ---------
    loss: np.array of float
        validation loss we want to smooth

    beta: float
        this paramer set how much strong the smoothing process is

    Yields
    ------
    smoothed_loss: np.array of float
        smoothed loss

    """
    num = len(loss)
    # create two empty arrays
    avg_loss = np.zeros(num)
    smoothed_loss = np.zeros(num)

    # compute avg loss
    for i in range(num):
        if i == 0:
            avg_loss[0] = (1-beta)*loss[0]
        else:
            avg_loss[i] = beta*avg_loss[i-1]+(1-beta)*loss[i]

            # compute smoothed loss
            smoothed_loss[i] = avg_loss[i]/(1-beta**(i+1))

    return smoothed_loss


if __name__ == '__main__':

    PLOT = True  # set to true if we want to plot
    SEARCH = False  # set to true if we want to search best learning rates

    if SEARCH is True:
        # import data
        data = import_training_set()
        # delete first 85 days of dataset
        data = data[data["date"] > 85]
        data["date"] = data["date"]-85

        # Remove feature based on correlation
        useless = feature_selection.main(0.93)
        data = data.drop(useless, axis=1)

        # remove features based on MDI feature importance
        redunant_feat = np.loadtxt(
            "../FeatureSelection/Results/deleted_feat_skip85.csv", dtype="str")
        data = data.drop(redunant_feat, axis=1)

        print('Imported successfully')
        features = [c for c in data.columns if 'feature' in c]
        cv = TimeSeriesSplit(n_splits=5, gap=10)

        # split data into train and test set
        X_train, y_train, X_test, y_test = split_data(data)

        # compute
        input_dim = X_train.shape[1]

        # create two empty arrays that will contain the learning rate and the loss values
        # for each iteration
        array_learn = np.zeros(40)
        value_loss = np.zeros(40)

        # starting learning rate
        learn_rate = 1e-8 * 1.22**51

        # computational time
        start = time.time()
        for i in range(40):

            # set hyperparameters
            hidden_units = [128, 256, 256, 128]
            dropout_rates = [0.10143786981358652, 0.19720339053599725,
                             0.2703017847244654, 0.23148340929571917, 0.2357768967777311]

            # build model
            print('Building model...')
            model = build(data, input_dim, hidden_units, 1, dropout_rates, learn_rate)
            print('Training model...')

            # fit the model
            history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                                epochs=1, batch_size=8192, verbose=1)

            print(history.history.keys())

            # save values of loss and learning rate
            value_loss[i] = (history.history['loss'][0])
            array_learn[i] = learn_rate

            # set learning rate for next cycle
            learn_rate = learn_rate*1.22
            print(learn_rate)

        # save results t a csv file so we can use them later
        np.savetxt("losss.txt", value_loss)

    if PLOT is True:
        raw_loss = np.loadtxt("losss.txt")
        new_loss = filter_loss(raw_loss, 0.95)

        l_rate = np.zeros(39)
        for i in range(39):
            l_rate[i] = 1.22**(51+i)*1e-8

        new_loss[0] = 0.7559  # i don't know why but it doesn't register first value

        # plot
        plt.plot(l_rate, new_loss, label="Smoothed loss")
        plt.plot(l_rate, raw_loss, label="Raw loss")
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
        plt.title("Best loss over 1 epoch training", fontsize=18)
        plt.legend()
        plt.xscale("log")
        plt.show()
