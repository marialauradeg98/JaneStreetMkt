import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from keras.layers import Input,Dense,Dropout,Activation,BatchNormalization
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from initial_import import import_training_set
from splitting import split_data
import gc



def build (input_dim,dropout0,num_layers,hidden_units,dropout_rates,learning_r):
    """
    This function builts the deep neural network used for the training.
    """
    input = Input(shape = (input_dim, ))
    x = BatchNormalization()(input) #re-centring and rescaling input layer
    x = Dropout(dropout_rates[0])(x) #a fraction of nodes is discarded with a frequency equal to the rate
    for i in range(num_layers):
        x = Dense(hidden_units[i], activation='relu')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rates[i])(x)
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
    data = pd.read_csv("reduced_dataset.csv", index_col=0, dtype="float32")
    print('Training set imported successfully.')
    #user window
    #After the first 85 days Jane Street changed its trading criteria
    #value1=input('Do you want to skip the first 85 trading days? y/n\n')
    #if value1=="y":
    #    data =skip_days(data,85)
    #if value1 != "n":
    #    print('Please,enter valid key.\n')
    #decide to consider or not transaction with 0 weight
    #value2=input('Do you want to skip transaction with weight 0? y/n\n')
    #if value2=="y":
    #    skip_value(data,0)
    #if value2 !="n":
    #    print('Please,enter valid key.\n')
    #splitting dataset in training and test set
    X, y, X_test, y_test=split_data(data)
    #set parameters for the model
    input_dim=X.shape[1]
    dropout0=0.10891515897566811
    num_layers=5
    hidden_units = [320, 192, 256, 384,128]
    dropout_rates = [0.30791117675740814,0.2649715309017116, 0.29364750243165694,  0.3302864832545722, 0.1773745339569533]
    learning_r=0.0081
    #cross validation
    scores_cv=[]
    fold = TimeSeriesSplit(n_splits = 5, gap=int(2e5),max_train_size=int(1e6),test_size=int(2e5))
    for fold, (train_index, val_index) in enumerate(fold.split(X,y)):
        print('Fold: {}'.format(fold+1))
        # for each iteration we define the boundaries of training and validation set
        X_train = X.iloc[train_index[0]:train_index[-1], :]
        X_val = X.iloc[val_index[0]:val_index[-1], :]
        y_train = y[train_index[0]:train_index[-1]]
        y_val = y[val_index[0]:val_index[-1]]
        #define checkpoint path
        checkpoint_path=f'Model_{fold+1}.hdf5'
        #build the deep neural network
        print('Building model...')
        model=build(input_dim,dropout0,num_layers,hidden_units,dropout_rates,learning_r)
        #define usefull callbacks
        checkpoint=ModelCheckpoint(checkpoint_path,monitor='val_auc', verbose=1,save_best_only=True,save_weights_only=True, mode='max')
        #reduce learning rate when accuracy stops to increase
        reduce_lr=ReduceLROnPlateau(monitor='val_auc',factor=0.2, patience=5,mode='max')
        #stop training when the accuracy stops to increase
        es=EarlyStopping(monitor='val_auc', patience=6,mode='max',min_delta=0.001)
        #training step for out neural network: fit,score and loss plot
        print('Fit the model...')
        history=model.fit(X_train,y_train, validation_data=(X_val,y_val), epochs=1000,batch_size=4096, verbose=1, callbacks=[es,reduce_lr,checkpoint])
        model.load_weights(checkpoint_path)
        model.save_weights(checkpoint_path)
        predictions=model.predict(X_val)
        predictions_classes=predictions.argmax(axis=1)
        acc=accuracy_score(y_val,predictions_classes)
        scores_cv.append(acc)
        del X_train, X_val, y_train, y_val
        gc.collect()

    # compute execution time
    mins = (time.time()-start)//60
    sec = (time.time()-start) % 60
    finish = (time.time()-start)/60
    print('Time to execute the cross validation is : {} min {:.2f} sec\n'
          .format(mins, sec))
    print('The score for each fold is:\n')
    print(scores_cv)
    #score on the test set
    test_predictions=model.predict(X_test)
    test_classes=test_predictions.argmax(axis=1)
    score_test=accuracy_score(y_test,test_classes)
    print('The score on the test set is:\n')
    print(score_test)
    #score on the entire training set
    train_predictions=model.predict(X)
    train_classes=train_predictions.argmax(axis=1)
    score_train=accuracy_score(y_train,train_classes)
    print('The score on the entire training set is:\n')
    print(score_train)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend(loc='lower right')
    plt.show()
    # compute execution time
    mins = (time.time()-start)//60
    sec = (time.time()-start) % 60
    print('Time to execute the feature selection model is time is: {} min {:.2f} sec\n'
          .format(mins, sec))
