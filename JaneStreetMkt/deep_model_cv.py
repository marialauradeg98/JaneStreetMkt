"""
This module is used to compute the cross validation for our deep neural network
model. For the neural network parameters we choose the best hyperparameters identified
by the optimization with Keras Tuner and the learning rate we compute in the
module find_learn_rate. We compute a 5 fold cross validation and we save the
best model obtained for each fold in a specific file which we save on our directory.
"""
import time
import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from keras.layers import Input,Dense,BatchNormalization
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from initial_import import import_training_set
from splitting import split_data
import feature_selection


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
    #input layer
    input = Input(shape = (input_dim, ))
    #re-centring and rescaling input layer
    x = BatchNormalization()(input)
    #iterations on the number of layers
    for i in range(num_layers):
        #dense layers
        x = Dense(hidden_units[i], activation='relu')(x)
    #final dense layer with its activation function
    output = Dense(1, activation= 'sigmoid')(x)
    model = Model(inputs = input, outputs = output)
    #compile our model choosing the type of loss, optimizer and metrics we want to use
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=learning_r), metrics=['AUC'] )
    return model


if __name__ == '__main__':
    start = time.time()
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

    #splitting dataset in training and test set
    X, y, X_test, y_test=split_data(data)
    #set parameters for the model
    NUM_LAYERS = 5
    hidden_units = [320, 192, 256, 384, 128]
    LEARNING_RATE = 0.0081
    #cross validation
    results_cv=[]    #empty array for the results (loss and acc) on the folds
    #set the fold, considering the fact we have a timeline for our dataset
    fold = TimeSeriesSplit(n_splits = 5, gap=int(2e5),test_size=int(2e5))
    for fold, (train_index, val_index) in enumerate(fold.split(X,y)):
        print('Fold: {}'.format(fold+1))
        # for each iteration we define the boundaries of training and validation set
        X_train = X.iloc[train_index[0]:train_index[-1], :]
        X_val = X.iloc[val_index[0]:val_index[-1], :]
        y_train = y[train_index[0]:train_index[-1]]
        y_val = y[val_index[0]:val_index[-1]]
        dim=X_train.shape[1]
        # define checkpoint path
        checkpoint_path = f'Model_{fold+1}.hdf5'
        #build the deep neural network
        print('Building model...')
        nn_model=build(dim,NUM_LAYERS,hidden_units,LEARNING_RATE)
        #define usefull callback
        #usefull for saving best model we obtained during the cross validation
        checkpoint=ModelCheckpoint(checkpoint_path,monitor='val_auc', verbose=1,
                                   save_best_only=True,save_weights_only=True,
                                   mode='max')
        #reduce learning rate when accuracy stops to increase
        reduce_lr=ReduceLROnPlateau(monitor='val_auc',factor=0.2,
                                    patience=4,mode='max')
        #stop training when the accuracy stops to increase
        es=EarlyStopping(monitor='val_auc', patience=7,mode='max'
                         ,min_delta=3e-4)
        #training step for out neural network: fit and score
        print('Fit the model...')
        nn_model.fit(X_train,y_train, validation_data=(X_val,y_val), epochs=1000,
                  batch_size=4096, verbose=1, callbacks=[es,reduce_lr,checkpoint])
        #evaluate the model on each fold
        res_cv=nn_model.evaluate(X_val,y_val,batch_size=4096)
        results_cv.append(res_cv[1])
        #checkpoint path update
        nn_model.load_weights(checkpoint_path)
        nn_model.save_weights(checkpoint_path)
        del X_train, X_val, y_train, y_val
        gc.collect()

    #print results for the cross validation
    print('AUC score for each fold is:\n',results_cv)
    #evaluate the model on test set
    print('Evaluating model...')
    results_test = nn_model.evaluate(X_test, y_test, batch_size=4096)
    #evaluate the model on training set
    results_train = nn_model.evaluate(X, y, batch_size=4096)
    # compute execution time
    mins = (time.time()-start)//60
    sec = (time.time()-start) % 60
    finish = (time.time()-start)/60
    # create a datafram that sumarizes the results
    results = {"score test ": results_test[1], "score training": results_train[1],
               "computational time (min)": finish}
    end_results = pd.DataFrame(results, index=["values"])
    #print final results
    print("Recap of the results:")
    print(end_results)
    # save results and removed features as csv
    end_results.to_csv("Results/results_NN.csv")
