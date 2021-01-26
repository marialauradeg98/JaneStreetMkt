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
import feature_selection
import gc
from deep_model_cv import build



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
    checkpoint_path='Model_final.hdf5'
    #call for building the model with the defined parameters
    model=build(input_dim,num_layers,hidden_units,learning_r)
    #redefine callbacks
    checkpoint=ModelCheckpoint(checkpoint_path,monitor='val_auc', verbose=1,save_best_only=True,save_weights_only=True, mode='max')
    model.load_weights('Model_5.hdf5')
    #reduce learning rate when accuracy stops to increase
    reduce_lr=ReduceLROnPlateau(monitor='val_auc',factor=0.2, patience=5,mode='max')
    #stop training when the accuracy stops to increase
    es=EarlyStopping(monitor='val_auc', patience=6,mode='max',min_delta=0.001)
    #new fit on the entire training set
    print('New fit on the entire training set:\n')
    model.fit(X,y,validation_data=(X_test,y_test), epochs=1000,batch_size=4096, verbose=1, callbacks=[es,reduce_lr,checkpoint])
    model.save_weights(checkpoint_path)
    model.load_weights(checkpoint_path)
    checkpoint=ModelCheckpoint(checkpoint_path,monitor='val_auc', verbose=1,save_best_only=True,save_weights_only=True, mode='max')
    #fast fit with smaller learning rate
    print('Faster training on the entire training set:\n')
    model=build(input_dim,num_layers,hidden_units,learning_r/100)
    history=model.fit(X,y,validation_data=(X_test,y_test), epochs=1000,batch_size=4096, verbose=1, callbacks=[es,reduce_lr,checkpoint])
    #evaluation on the test set
    print('Evaluate on test data... ')
    pred_test=model.predict(X_test)
    pred_test_classes=pred_test.argmax(axis=1)
    accs_test=accuracy_score(y_test,pred_test_classes)
    print('The accuracy score on test set is: ',accs_test)
    results_ts=model.evaluate(X_test,y_test,batch_size=4096)
    print('test loss, test acc: ', results_ts)
    #evaluation on the entire training set
    print('Evaluate on training data: ')
    pred_tr=model.predict(X)
    pred_tr_classes=pred_tr.argmax(axis=1)
    accs_tr=accuracy_score(y,pred_tr_classes)
    print('The accuracy score on training set is: ',accs_tr)
    results_tr=model.evaluate(X,y,batch_size=4096)
    print('training loss, training acc: ', results_tr)
    #plot loss and accuracy
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend(loc='lower right')
    plt.show()
    plt.plot(history.history['auc'], label='Training auc')
    plt.plot(history.history['val_auc'], label='Validation auc')
    plt.legend(loc='lower right')
    plt.show()
    #save the final model checkpoint
    model.save_weights(checkpoint_path)
    # compute execution time
    mins = (time.time()-start)//60
    sec = (time.time()-start) % 60
    print('Time to execute the feature selection model is time is: {} min {:.2f} sec\n'
          .format(mins, sec))
