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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, train_test_split
import matplotlib.pyplot as plt
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


if __name__ == '__main__':
    # import data
    data = import_training_set()
    # delete first 85 days of dataset
    data = data[data["date"] > 85]
    data["date"] = data["date"]-85

    # Remove feature based on correlation
    useless = feature_selection.main(0.93)
    data = data.drop(useless, axis=1)

    # remove features based on MDI feature importance
    redunant_feat = np.loadtxt("../FeatureSelection/Results/deleted_feat_skip85.csv", dtype="str")
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
    array_learn = np.zeros(100)
    value_loss = np.zeros(100)

    # starting learning rate
    learn_rate = 1e-8

    # computational time
    start = time.time()

    for i in range(100):

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
        value_loss[i] = (history.history['val_loss'][0])
        array_learn[i] = learn_rate

        # set learning rate for next cycle
        learn_rate = learn_rate*1.22

    # save results t a csv file so we can use them later
    np.savetxt("loss1.csv", value_loss, delimiter=',')
    np.savetxt("learning_rate1.csv", array_learn, delimiter=',')

    plt.plot(array_learn, value_loss)
    plt.xscale("log")
    plt.show()
