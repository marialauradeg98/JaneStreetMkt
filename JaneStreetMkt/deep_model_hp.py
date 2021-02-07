"""
This module is used for the hyperparameters optimization in our deep
neural network. We implemented our search using Keras Tuner, in particular the
so called Random Search for finding hyperparameters which maximize the val_auc.
"""
import time
#import pandas as pd
import numpy as np
from kerastuner import Objective
from kerastuner.tuners import RandomSearch
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, BatchNormalization
from keras.optimizers import Adam
from keras.models import Sequential
from initial_import import import_training_set
from splitting import split_data
import feature_selection


def build_model(hp):
    """
    This function in used for build the neural network model, specifing the
    hyperparameters we want to optimize and their searching rate.

    Parameters
    ----------
    hp: Keras Tuner hyperparameter
        The argument from which you can sample hyperparameters.
    Yields
    ------
    model: Keras Model
        The deep neural network model we built
    """
    #defining the type of model we want to use
    model = Sequential()
    #re-centring and rescaling
    model.add(BatchNormalization())
    #iterations on the number of layers
    for i in range(hp.Int('num_layers', 2, 5)):
        #dense layer with a variable number of hidden units
        model.add(Dense(units=hp.Int('hidden_units'+str(i), min_value=64,
                                     max_value=512, step=64), activation='relu'))
    #final dense layer with its activation function
    model.add(Dense(1, activation='sigmoid'))
    #compile our model choosing the type of loss, optimizer and metrics we want to use
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.00810), metrics=['AUC'])
    return model


if __name__ == '__main__':
    start = time.time()
    # import training dataset
    print('Importing training set...')
    data = import_training_set()
    print('Training set imported successfully.')
    # Remove feature based on correlation
    useless = feature_selection.main(0.93)
    data = data.drop(useless, axis=1)
    # remove features based on MDI feature importance
    redunant_feat = np.loadtxt(
        "../FeatureSelection/Results/deleted_feat_skip85.csv", dtype="str")
    data = data.drop(redunant_feat, axis=1)

    #if the training set is too heavy to import you can directly import the reduced dataset we built
    #data = pd.read_csv("reduced_dataset.csv", index_col=0, dtype="float32")

    # splitting dataset in training,test and validation sets
    X_tr, y_tr, X_test, y_test, X_val, y_val = split_data(data, val=True)
    # set the Random search specifing the quantity we want to maximize (val_auc)
    tuner = RandomSearch(
        build_model,
        seed=18,
        objective=Objective('val_auc', 'max'),
        max_trials=20,#number of hyperparameter combinations that will be tested by the tuner
        executions_per_trial=1, #number of models that should be built and fit for each trial
        directory='randomsearch',
        project_name='keres_tuner')
    # define usefull callbacks
    # reduce learning rate when val_auc stops to increase
    reduce_lr = ReduceLROnPlateau(monitor='val_auc', factor=0.2,
                                  patience=5, mode='max')
    #stop training when val_auc stops to increase
    es = EarlyStopping(monitor='val_auc', patience=6, mode='max', min_delta=0.001)
    #start the search for our hyperparameters
    tuner.search(X_tr, y_tr, epochs=20000, callbacks=[
                 es, reduce_lr], batch_size=4096, validation_data=(X_val, y_val))
    # print a summary of out tuning actions
    tuner.results_summary()
    #evaluate best model on test set
    best_model = tuner.get_best_models(num_models=1)[0]
    loss_t, accuracy_t = best_model.evaluate(X_test, y_test)
    print('The loss on the test set is:\n')
    print(loss_t)
    print('The accuracy on the test set is:\n')
    print(accuracy_t)
    #print the best parameters obtained by the Random Search
    best_hp = tuner.get_best_hyperparameters()[0].values
    print('The best hyperparameters are:\n')
    print(best_hp)
    # compute execution time
    mins = (time.time()-start)//60
    sec = (time.time()-start) % 60
    print('Time to execute the random search is: {} min {:.2f} sec\n'
          .format(mins, sec))
