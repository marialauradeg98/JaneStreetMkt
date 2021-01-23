import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, train_test_split
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
from keras.models import Model
import keras.losses
from initial_import import import_training_set
from splitting import split_data
import feature_selection
import gc


def build(input_dim, hidden_units, num_labels, dropout_rates, learning_rate):
    """
    This function builts the deep neural network used for the training.
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
    data = import_training_set()
    #data = data[data["date"] > 85]
    #data["date"] = data["date"]-85

    # Remove feature based on correlation

    useless = feature_selection.main(0.93)
    data = data.drop(useless, axis=1)

    # remove features based on MSI feature importance
    redunant_feat = np.loadtxt("../FeatureSelection/Results/deleted_feat_skip85.csv", dtype="str")
    data = data.drop(redunant_feat, axis=1)

    # splitting dataset in training and test set
    X, y, X_test, y_test = split_data(data)
    # set parameters for the model
    dropout0 = 0.1808
    num_layers = 2
    hidden_units = [320, 512]
    dropout_rates = [0.1808, 0.1296, 0.2920]
    learning_r = 0.0081

    # define parameters of cross validation
    N_FOLDS = 5
    folds = TimeSeriesSplit(n_splits=N_FOLDS,
                            max_train_size=int(1e6),
                            test_size=int(2e5),
                            gap=int(2e5))

    splits = folds.split(X, y)
    score_cv = []  # empty list will contain accuracy score of each split
    start = time.time()
    for fold_n, (train_index, val_index) in enumerate(splits):
        print('Fold: {}'.format(fold_n+1))
        # for each iteration we define the boundaries of training and validation set
        X_train = X.iloc[train_index[0]:train_index[-1], :]
        X_val = X.iloc[val_index[0]:val_index[-1], :]
        y_train = y[train_index[0]:train_index[-1]]
        y_val = y[val_index[0]:val_index[-1]]
        dimension = X_train.shape[1]
        # build the deep neural network
        print('Building model...')
        model = build(dimension, hidden_units, 1, dropout_rates, learning_r)
        # define usefull callbacks
        # reduce learning rate when accuracy stops to increase
        reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=4, mode='max')
        # stop training when the accuracy stops to increase
        es = EarlyStopping(monitor='val_accuracy', patience=7, mode='max', min_delta=3e-4)
        # training step for out neural network: fit and score
        print('Training model...')
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1000,
                  batch_size=4096, verbose=1, callbacks=[es, reduce_lr])

        # delete train and validation set to save memory
        del X_train, X_val, y_train, y_val
        gc.collect()

    finish = (time.time()-start)/60
    print("The accuracy score of each iteration is:")

    # evaluate the model on test set
    print('Evaluating model...')
    history = model.fit(X, y, validation_data=(X_test, y_test), epochs=1000,
                        batch_size=4096, verbose=1, callbacks=[es, reduce_lr])

    acc_train = history.history["val_accuracy"]
    acc_test = history.history["accuracy"]

    print(max(acc_test), max(acc_train))
