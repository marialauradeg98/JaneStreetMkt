import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, train_test_split
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
from keras.models import Model
import keras.losses
from initial_import import import_training_set
from splitting import split_data
import feature_selection


def build(data, input_dim, hidden_units, num_labels, dropout_rates, learning_rate):
    """
    This function builts the deep neural network used for the training.
    """
    input = Input(shape=(input_dim, ))
    x = BatchNormalization()(input)  # re-centring and rescaling input layer
    # a fraction of nodes is discarded with a frequency equal to the rate
    x = Dropout(dropout_rates[0])(x)
    for i in range(len(hidden_units)):
        x = Dense(hidden_units[i])(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rates[i+1])(x)

    output = Dense(num_labels, activation='sigmoid')(x)
    model = Model(inputs=input, outputs=output)
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(learning_rate), metrics=['accuracy'])
    model.summary()

    return model


if __name__ == '__main__':
    start = time.time()
    data = import_training_set()
    data = data[data["date"] > 85]
    data["date"] = data["date"]-85

    # Remove feature based on correlation

    useless = feature_selection.main(0.93)
    data = data.drop(useless, axis=1)

    # remove features based on MSI feature importance
    redunant_feat = np.loadtxt("../FeatureSelection/Results/deleted_feat_skip85.csv", dtype="str")
    data = data.drop(redunant_feat, axis=1)

    print('Imported successfully')
    features = [c for c in data.columns if 'feature' in c]
    cv = TimeSeriesSplit(n_splits=5, gap=10)
    X_train, y_train, X_test, y_test = split_data(data)
    input_dim = X_train.shape[1]
    lulz = np.zeros(50)
    lear = 1e-8
    fug = np.zeros(50)
    bungo = 0
    for i in range(50):
        hidden_units = [128, 256, 256, 128]
        dropout_rates = [0.10143786981358652, 0.19720339053599725,
                         0.2703017847244654, 0.23148340929571917, 0.2357768967777311]
        print('Building model...')
        model = build(data, input_dim, hidden_units, 1, dropout_rates, lear)
        print('Training model...')
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                            epochs=1, batch_size=8192, verbose=1)

        print(history.history.keys())
        lulz[i] = (history.history['val_loss'][0])
        lear = lear*1.22
        fug[i] = lear
        bungo = (history.history['val_loss'][0])

    np.savetxt("loss.csv", lulz, delimiter=',')
    np.savetxt("learning_rate.csv", fug, delimiter=',')

    plt.plot(fug, lulz)
    plt.xscale("log")
    plt.show()
    '''


    print('Evaluating model...')
    predictions = model.predict_proba(X_val, verbose=1)
    roc = roc_auc_score(y_val, predictions)
    scores = model.evaluate(X_val, y_val)
    print(scores)
    model.summary()
    # compute execution time
    mins = (time.time()-start)//60
    sec = (time.time()-start) % 60
    print('Time to execute the feature selection model is time is: {} min {:.2f} sec\n'
          .format(mins, sec))
    '''
