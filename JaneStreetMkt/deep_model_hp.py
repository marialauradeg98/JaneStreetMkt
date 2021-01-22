"""
This module is used for the hyperparameters optimization in our deep neural network
We implemented our search using Keras Tuner, in particular the so called Random Search
for finding hyperparameters which maximize the AUC.
"""
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from kerastuner.tuners import RandomSearch
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Input,Dense,Dropout,Activation,BatchNormalization
from keras.optimizers import Adam
from keras.models import Model,Sequential
from initial_import import import_training_set
from splitting import split_data


def build_model(hp):
    """
    This function in used for build the neural network model specifing the hyperparameters
    we want to optimize and their searching rate.
    Parameters
    ----------
    hp: Keras Tuner hyperparameter
        The argument from which you can sample hyperparameters.
    Yields
    ------
    model: Keras Model
        The deep neural network model we built
    """
    print(type(hp))
    model=Sequential()
    model.add(BatchNormalization()) #re-centring and rescaling input layer
    model.add(Dropout(rate = hp.Float('dropout0'+str(0), min_value=0.3,max_value=0.5))) #a fraction of nodes is discarded with a frequency equal to the rate
    for i in range(hp.Int('num_layers',2,5)):
        model.add(Dense(units=hp.Int('hidden_units'+str(i), min_value=64, max_value=512, step=64), activation='relu'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(rate=hp.Float('dropout_rates'+ str(i),min_value=0.3, max_value=0.5)))
    model.add(Dense(1, activation= 'sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.00810), metrics=['AUC'] )
    print(type(model))
    return model


if __name__ == '__main__':
    start = time.time()
    #importing the dataset
    data = pd.read_csv("reduced_dataset.csv", index_col=0, dtype="float32")
    #splitting dataset in training,test and validation set
    X_tr, y_tr, X_test, y_test,X_val,y_val=split_data(data,val=True)
    #set the Random search specifing the quantity we want to maximize: AUC
    tuner = RandomSearch(
        build_model,
        seed=10,
        objective='val_AUC',
        max_trials=10,
        executions_per_trial=1,
        directory='randomsearch',
        project_name='keres_tuner')
    #define usefull callbacks
    #reduce learning rate when AUC stops to increase
    reduce_lr=ReduceLROnPlateau(monitor='val_AUC',factor=0.2, patience=5,mode='max')
    #stop training when the AUC stops to increase
    es=EarlyStopping(monitor='val_AUC', patience=6,mode='max',min_delta=0.001)
    #start the search for our hyperparameters
    tuner.search(X_tr,y_tr,epochs=20000,callbacks=[es,reduce_lr], batch_size=4096,validation_data=(X_val,y_val))
    #print the best results
    tuner.results_summary()
    best_model=tuner.get_best_models(1)[0]
    best_hyperparameters=tuner.get_best_hyperparameters(1)[0]
    print('After the search, the best hyperparameters are:\n {}'.format(best_hyperparameters))
    # compute execution time
    mins = (time.time()-start)//60
    sec = (time.time()-start) % 60
    print('Time to execute the feature selection model is time is: {} min {:.2f} sec\n'
          .format(mins, sec))
