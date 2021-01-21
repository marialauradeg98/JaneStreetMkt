import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score,TimeSeriesSplit
import matplotlib.pyplot as plt
from kerastuner.tuners import RandomSearch
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import np_utils
from keras.layers import Input,Dense,Dropout,Activation,BatchNormalization
from keras.optimizers import Adam
from keras.models import Model,Sequential
from initial_import import import_training_set
from splitting import split


def build_model(hp):
    """

    """
    model=Sequential()
    model.add(BatchNormalization()) #re-centring and rescaling input layer
    model.add(Dropout(rate = hp.Float('dropout0'+str(0), min_value=0.0,max_value=0.5))) #a fraction of nodes is discarded with a frequency equal to the rate
    for i in range(hp.Int('num_layers',2,5)):
        model.add(Dense(units=hp.Int('hidden_units'+str(i), min_value=40, max_value=2000, step=50), activation='relu'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(rate=hp.Float('dropout_rates'+ str(i),min_value=0.0, max_value=0.5)))
    model.add(Dense(1, activation= 'sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(hp.Float('learning_rate',min_value=0.0001, max_value=0.01,sampling='LOG')), metrics=['AUC'] )
    return model



def main(data):
    start = time.time()
    #splitting dataset in training,test and validation set
    X_tr, y_tr, X_test, y_test,X_val,y_val=split(data,val=True)
    tuner = RandomSearch(
        build_model,
        seed=1,
        objective='val_AUC',
        max_trials=50,
        executions_per_trial=1,
        directory='randomsearch',
        project_name='keres_tuner')
    #define usefull callbacks
    #reduce learning rate when accuracy stops to increase
    reduce_lr=ReduceLROnPlateau(monitor='val_AUC',factor=0.2, patience=5,mode='max')
    #stop training when the accuracy stops to increase
    es=EarlyStopping(monitor='val_AUC', patience=10,mode='max')
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
