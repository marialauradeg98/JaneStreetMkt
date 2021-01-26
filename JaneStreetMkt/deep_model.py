
import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from initial_import import import_training_set
from splitting import split_data
import gc
import feature_selection
from statistics import mean, stdev


def build_NOTHONG(input_dim,  num_layers, hidden_units, learning_r):
    """
    This function builts the deep neural network used for the training.
    """
    input = Input(shape=(input_dim, ))
    # a fraction of nodes is discarded with a frequency equal to the rate
    x = BatchNormalization()(input)  # re-centring and rescaling input layer
    for i in range(num_layers):
        x = Dense(hidden_units[i], activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(
        learning_rate=learning_r), metrics=['AUC'])
    return model


if __name__ == '__main__':

    data = import_training_set()

    data = data[data["date"] > 85]
    data["date"] = data["date"]-85

    # Remove feature based on correlation

    useless = feature_selection.main(0.93)
    data = data.drop(useless, axis=1)

    # remove features based on MSI feature importance
    redunant_feat = np.loadtxt("../FeatureSelection/Results/deleted_feat_skip85.csv", dtype="str")
    data = data.drop(redunant_feat, axis=1)

    X, y, X_test, y_test = split_data(data)
    # set parameters for the model

    num_layers = 5
    hidden_units = [320, 192, 256, 384, 128]
    learning_r = 0.0081
    # cross validation

    scores_cv = []
    N_FOLDS = 5

    start = time.time()
    fold = TimeSeriesSplit(n_splits=N_FOLDS, gap=int(2e5), test_size=int(2e5))
    for fold, (train_index, val_index) in enumerate(fold.split(X, y)):
        print('Fold: {}'.format(fold+1))
        # for each iteration we define the boundaries of training and validation set
        X_train = X.iloc[train_index[0]:train_index[-1], :]
        X_val = X.iloc[val_index[0]:val_index[-1], :]
        y_train = y[train_index[0]:train_index[-1]]
        y_val = y[val_index[0]:val_index[-1]]
        input_dim = X_train.shape[1]
        # define checkpoint path
        checkpoint_path = f'Model_{fold+1}.hdf5'
        # build the deep neural network
        print('Building model...')
        model = build_NOTHONG(input_dim, num_layers, hidden_units, learning_r)
        # define usefull callbacks
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_auc',
                                     verbose=1, save_best_only=True, save_weights_only=True, mode='max')

        # reduce learning rate when accuracy stops to increase
        reduce_lr = ReduceLROnPlateau(monitor='val_auc', factor=0.2, patience=4, mode='max')
        # stop training when the accuracy stops to increase
        es = EarlyStopping(monitor='val_auc', patience=7, mode='max', min_delta=3e-4)
        # training step for out neural network: fit,score and loss plot
        print('Fit the model...')
        history = model.fit(X_train, y_train, validation_data=(
            X_val, y_val), epochs=1000, batch_size=4096, verbose=1, callbacks=[es, reduce_lr, checkpoint])

        print("accuracy of predictions:")

        model.load_weights(checkpoint_path)
        model.save_weights(checkpoint_path)

        results = model.evaluate(X_val, y_val, batch_size=4096)
        scores_cv.append(results[1])
        print(scores_cv)

        del X_train, X_val, y_train, y_val
        gc.collect()

    print('The score for each fold is:\n')
    print(scores_cv)
    # score on the test set
    # define usefull callbacks

    finish = (time.time()-start)/60

    # evaluate the model on test set
    print('Evaluating model...')

    results_test = model.evaluate(X_test, y_test, batch_size=4096)
    score_test = results_test[1]
    print("AUC score on test set is : {}".format(score_test))

    results_train = model.evaluate(X, y, batch_size=4096)
    score_train = results_train[1]
    print("AUC score on training set is : {}".format(score_train))

    # nice prints
    print("accuracy score of each iteration is")
    print("Time to fit the RF with {}k cross validation {:.2f} min".format(N_FOLDS, finish))

    print("Accuracy on training set is: {} \nAccuracy on test set is: {}".format(
        score_train, score_test))

    # create a dataframe that sumarizes the results
    results = {"score test ": score_test, "score training": score_train,
               "computational time (min)": finish, "splits": N_FOLDS}
    end_results = pd.DataFrame(results, index=["values"])

    # nice print
    print("a little recap of the results:")
    print(end_results)

    # save results and removed features as csv
    end_results.to_csv("Results/results_NN_1.csv")
