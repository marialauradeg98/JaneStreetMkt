import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score,TimeSeriesSplit, train_test_split
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.layers import Input,Dense,Dropout,Activation,BatchNormalization
from keras.models import Model
from initial_import import import_training_set
from RF_feature_importance import test_train



def build (data, input_dim,hidden_units, num_labels,dropout_rates):
    """
    This function builts the deep neural network used for the training.
    """
    X_train, X_test, y_train, y_test = test_train(data)
    input = Input(shape = (input_dim, ))
    x = BatchNormalization()(input) #re-centring and rescaling input layer
    x = Dropout(dropout_rates[0])(x) #a fraction of nodes is discarded with a frequency equal to the rate
    for i in range(len(hidden_units)):
        x = Dense(hidden_units[i])(x)
        x = BatchNormalization()(x)
        x = Activation('swish')(x)
        x = Dropout(dropout_rates[i+1])(x)

    output = Dense(num_labels, activation= 'sigmoid')(x)
    model = Model(inputs = input, outputs = output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'] )

    return model


if __name__ == '__main__':
    start = time.time()
    data = import_training_set()
    print('Imported successfully')
    features = [c for c in data.columns if 'feature' in c]
    cv = TimeSeriesSplit(n_splits = 5, gap=10)
    for s, (tr, te) in enumerate(cv.split(data['action'].values, data['action'].values, data['date'].values)):
        X_tr, X_val = data.loc[tr, features].values, data.loc[te, features].values
        y_tr, y_val = data.loc[tr, 'action'].values, data.loc[te, 'action'].values
    input_dim=X_tr.shape[1]
    hidden_units = [384, 896, 896, 394]
    dropout_rates = [0.10143786981358652, 0.19720339053599725, 0.2703017847244654, 0.23148340929571917, 0.2357768967777311]
    print('Building model...')
    model=build(data,input_dim,hidden_units,1,dropout_rates)
    print('Training model...')
    history=model.fit(X_tr,y_tr, validation_data=(X_val,y_val), epochs=10,batch_size=4096, verbose=1)

    print(history.history.keys())
    print(history.history['loss'])
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.show()
    print('Evaluating model...')
    predictions = model.predict_proba(X_val,verbose=1)
    roc=roc_auc_score(y_val, predictions)
    scores = model.evaluate(X_val,y_val)
    print(scores)
    model.summary()
    # compute execution time
    mins = (time.time()-start)//60
    sec = (time.time()-start) % 60
    print('Time to execute the feature selection model is time is: {} min {:.2f} sec\n'
          .format(mins, sec))
