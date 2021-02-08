"""
In this module we implement a tecnique to find the best learning rate
based on the article: https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
In the first part of the model we build 100 models with a linearly encreasing
learning rate over a period of only one epoch.
Then, after applying a smoothing algorithm, we find the learning rate lr which
minimizes the loss.
The best learning rate for our model is lr/10.
The reason we use this tecnique rather than getting the learning rate through an
hyperparameter search, is that  we need to fit the model over only one epoch
thus reducing by far the time required to find the best learning rate.
"""
from splitting import split_data
from initial_import import import_training_set
import feature_selection
import keras.losses
from keras.models import Model
from keras.layers import Input, Dense, Activation, BatchNormalization
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import time


def build(input_dim, num_layers, hidden_units, learning_r):
    """
    This function builts the model for our deep neural network used the hyperparameter
    we found with an optimization process.
    Parameters
    ----------

    input_dim: int
        The tensor shape we send to the first hidden layer and must be
        the same shape as the used training set.
    num_layers: int
        The number of hidden layers.
    hidden_units: list of int
        The units for each hidden layer.
    learining_r: float
        The learning rate.

    Yields
    ------

    model: Keras Model
        The deep neural network model we built
    """
    # input layer
    input = Input(shape=(input_dim, ))
    # re-centring and rescaling input layer
    x = BatchNormalization()(input)
    # iterations on the number of layers
    for i in range(num_layers):
        # dense layers
        x = Dense(hidden_units[i], activation='relu')(x)
    # final dense layer with its activation function
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input, outputs=output)
    # compile our model choosing the type of loss, optimizer and metrics we want to use
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=learning_r), metrics=['AUC'])
    return model


def filter_loss(loss, beta):
    """
    this function computes the smoothed loss
    Parameters
    ----------
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

        # delete first 85 days of data
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

        # split data into train and test set
        X_train, y_train, X_test, y_test = split_data(data)

        # compute
        dimension = X_train.shape[1]

        # create two empty arrays that will contain the learning rate and the loss values
        # for each iteration
        array_learn = np.zeros(40)
        value_loss = np.zeros(40)

        # starting learning rate
        LEARN_RATE = 1e-8 * 1.22**51

        # computational time
        start = time.time()
        for j in range(40):

            # set hyperparameters
            hidden = [128, 256, 256, 128]

            # build model
            print('Building model...')
            nn_brain = build(dimension, 4, hidden, LEARN_RATE)
            print('Training model...')

            # fit the model
            HISTORY = nn_brain.fit(X_train, y_train, validation_data=(X_test, y_test),
                                   epochs=1, batch_size=8192, verbose=1)

            # save values of loss and learning rate
            value_loss[j] = (HISTORY.history['loss'][0])
            array_learn[j] = LEARN_RATE

            # set learning rate for next cycle
            LEARN_RATE = LEARN_RATE*1.22
            print(LEARN_RATE)

        # save results t a csv file so we can use them later
        np.savetxt("losss.txt", value_loss)

    if PLOT is True:
        # load loss values
        raw_loss = np.loadtxt("losss.txt")
        new_loss = filter_loss(raw_loss, 0.95)

        # create an array of learning rates
        l_rate = np.zeros(39)
        for k in range(39):
            l_rate[k] = 1.22**(51+k)*1e-8

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
