"""
This module is used to compute the cross validation for our deep neural network
model. For the neural network parameters we choose the best hyperparameters identified
by the optimization with Keras Tuner and the learning rate we compute in the
module find_learn_rate. We compute a 5 fold cross validation and we save the
best model obtained for each fold in a specific file which we save on our directory.
"""
import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from keras.layers import Input,Dense,Dropout,Activation,BatchNormalization
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from initial_import import import_training_set
from splitting import split_data
import feature_selection
import gc



def build (input_dim,num_layers,hidden_units,learning_r):
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
    hidden_units:
        The units for each hidden layer.
    learining_r:
        The learning rate.
    Yields
    ------
    model: Keras Model
        The deep neural network model we built
    """
#GIRARE SENZA DROPOUT, AGGIUNGO EPOCHE SULLA VALIDAZIONE
    #input layer
    input = Input(shape = (input_dim, ))
    #re-centring and rescaling input layer
    x = BatchNormalization()(input)
    #iterations on the number of layers
    for i in range(num_layers):
        #dense layers
        x = Dense(hidden_units[i], activation='relu')(x)
        #new normalization step
        x = BatchNormalization()(x)
        #defining the activation function
        x = Activation('relu')(x)
    #final dense layer with its activation function
    output = Dense(1, activation= 'sigmoid')(x)
    model = Model(inputs = input, outputs = output)
    #compile our model choosing the type of loss, optimizer and metrics we want to use
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=learning_r), metrics=['AUC'] )
    return model


if __name__ == '__main__':
    start = time.time()
    """ import training dataset
    print('Importing training set...')
    data = import_training_set()
    print('Training set imported successfully.')
    # Remove feature based on correlation
    useless = feature_selection.main(0.93)
    data = data.drop(useless, axis=1)
    # remove features based on MDI feature importance
    redunant_feat = np.loadtxt(
        "../FeatureSelection/Results/deleted_feat_skip85.csv", dtype="str")
    data = data.drop(redunant_feat, axis=1)"""

    #if the training set is too heavy to import you can directly import the reduced dataset we built
    data = pd.read_csv("reduced_dataset.csv", index_col=0, dtype="float32")

    #splitting dataset in training and test set
    X, y, X_test, y_test=split_data(data)
    #set parameters for the model
    input_dim=X.shape[1]
    num_layers,learning_r=np.loadtxt('best_lr_nl.txt')
    hidden_units=np.loadtxt('best_hu.txt',comments='#',delimiter=',',unpack=False)
    num_layers=int(num_layers)
    hidden_units=np.array(hidden_units,dtype=int)
    #cross validation
    scores_cv=[]    #empty array for the scores on the folds
    #set the fold, considering the fact we have a timeline for our dataset
    fold = TimeSeriesSplit(n_splits = 5, gap=int(2e5),test_size=int(2e5))
    for fold, (train_index, val_index) in enumerate(fold.split(X,y)):
        print('Fold: {}'.format(fold+1))
        # for each iteration we define the boundaries of training and validation set
        X_train = X.iloc[train_index[0]:train_index[-1], :]
        X_val = X.iloc[val_index[0]:val_index[-1], :]
        y_train = y[train_index[0]:train_index[-1]]
        y_val = y[val_index[0]:val_index[-1]]
        #define checkpoint path for each fold
        checkpoint_path=f'Model_{fold+1}.hdf5'
        #build the deep neural network
        print('Building model...')
        model=build(input_dim,num_layers,hidden_units,learning_r)
        #define usefull callback
        #usefull for saving best model we obtained during the cross validation
        checkpoint=ModelCheckpoint(checkpoint_path,monitor='val_auc', verbose=1,
                                   save_best_only=True,save_weights_only=True,
                                   mode='max')
        #reduce learning rate when accuracy stops to increase
        reduce_lr=ReduceLROnPlateau(monitor='val_auc',factor=0.2,
                                    patience=5,mode='max')
        #stop training when the accuracy stops to increase
        es=EarlyStopping(monitor='val_auc', patience=6,mode='max'
                         ,min_delta=0.001)
        #training step for out neural network: fit and score
        print('Fit the model...')
        model.fit(X_train,y_train, validation_data=(X_val,y_val), epochs=1000,
                  batch_size=4096, verbose=1, callbacks=[es,reduce_lr,checkpoint])
        predictions=model.predict(X_val)
        predictions_classes=predictions.argmax(axis=1)
        acc=accuracy_score(y_val,predictions_classes)
        scores_cv.append(acc)
        #a few epochs on validation set with small learning rate
        model=build(input_dim,num_layers,hidden_units,learning_r/100)
        model.load_weights(checkpoint_path)
        model.fit(X_val,y_val, epochs=4,batch_size=4096, verbose=1)
        model.save_weights(checkpoint_path)
        del X_train, X_val, y_train, y_val
        gc.collect()

    print('The accuracy score for each fold is:\n')
    print(scores_cv)
    search_file = open("cv_score.txt", "w")
    search_file.write("{}".format(scores_cv))
    search_file.close()
    print('Resulting score saved.')
    # compute execution time
    mins = (time.time()-start)//60
    sec = (time.time()-start) % 60
    finish = (time.time()-start)/60
    print('Time to execute the cross validation is : {} min {:.2f} sec\n'
          .format(mins, sec))
