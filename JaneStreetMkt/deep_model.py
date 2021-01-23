import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from keras.layers import Input,Dense,Dropout,Activation,BatchNormalization
from keras.models import Model
from keras.callbacks import ReduceLROnPLatea, EarlyStopping
from keras.optimizers import Adam
from initial_import import import_training_set
from splitting import split



def build (input_dim,dropout0,num_layers,hidden_units,dropout_rates,learning_r):
    """
    This function builts the deep neural network used for the training.
    """
    input = Input(shape = (input_dim, ))
    x = BatchNormalization()(input) #re-centring and rescaling input layer
    x = Dropout(dropout_rates[0])(x) #a fraction of nodes is discarded with a frequency equal to the rate
    for i in range(num_layers)):
        x = Dense(hidden_units[i], actvation='relu')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rates[i+1])(x)
    output = Dense(1, activation= 'sigmoid')(x)
    model = Model(inputs = input, outputs = output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_r), metrics=['AUC'] )
    return model

def skip_days(data,days):
    data = data[data["date"] > 85]
    data["date"] = data["date"]-85
    return data

def skip_value(data,value):
    data = data[data["weight"] != value]

if __name__ == '__main__':
    start = time.time()
    #importing training set
    data = import_training_set()
    print('Training set imported successfully.')
    #user window
    #After the first 85 days Jane Street changed its trading criteria
    value1=input('Do you want to skip the first 85 trading days? y/n\n')
    if value1=="y":
        data =skip_days(data,85)
    if value1 != "n":
        print('Please,enter valid key.\n')
    #decide to consider or not transaction with 0 weight
    value2=input('Do you want to skip transaction with weight 0? y/n\n')
    if value2=="y":
        skip_value(data,0)
    if value2 !="n":
        print('Please,enter valid key.\n')


    #splitting dataset in training and test set
    X_tr y_tr, X_test, y_test=split_data(data)
    #set parameters for the model
    input_dim=X_tr.shape[1]
    dropout0=0.10143786981358652
    num_layers=5
    hidden_units = [384, 896, 896, 394]
    dropout_rates = [0.19720339053599725, 0.2703017847244654, 0.23148340929571917, 0.2357768967777311]
    learning_r=1e-3
    #cross validation
    folds = TimeSeriesSplit(n_splits = 5, gap=int(2000),max_train_size=int(5000),test_size=int(1000))
    splits = folds.split(X, y)
    score_cv = []  # empty list will contain accuracy score of each split
    for fold_n, (train_index, val_index) in enumerate(splits):
        print('Fold: {}'.format(fold_n+1))
        # for each iteration we define the boundaries of training and validation set
        X_train = X.iloc[train_index[0]:train_index[-1], :]
        X_val = X.iloc[val_index[0]:val_index[-1], :]
        y_train = y[train_index[0]:train_index[-1]]
        y_val = y[val_index[0]:val_index[-1]]
        #build the deep neural network
        print('Building model...')
        model=build(input_dim,dropout0,num_layers,hidden_units,dropout_rates,learining_r)
        #define usefull callbacks
        #reduce learning rate when accuracy stops to increase
        reduce_lr=ReduceLROnPlateau(monitor='val_auc',factor=0.2, patience=5,mode='max')
        #stop training when the accuracy stops to increase
        es=EarlyStopping(monitor='val_auc', patience=6,mode='max',min_delta=0.001)
        #training step for out neural network: fit and score
        print('Training model...')
        model.fit(X_tr,y_tr, validation_data=(X_val,y_val), epochs=10,batch_size=4096, verbose=1, callbacks=[es,reduce_lr])
        score = model.score(X_val, y_val)
        score_cv.append(score)
        # delete train and validation set to save memory
        del X_tr, X_val, y_tr, y_val
        gc.collect()

    print("The accuracy score of each iteration is:")
    print(score_cv)
    #evaluate the model on test set
    print('Evaluating model...')
    predictions = model.predict(X_test,verbose=1)
    acc=accuracy_score(y_test, predictions)
    print('The score on test is:\n')
    print(acc)
    model.summary()

    # compute execution time
    mins = (time.time()-start)//60
    sec = (time.time()-start) % 60
    print('Time to execute the feature selection model is time is: {} min {:.2f} sec\n'
          .format(mins, sec))
