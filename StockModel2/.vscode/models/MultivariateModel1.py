import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime
import tensorflow
import tensorflow.keras

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam

from pylab import rcParams

def datetime_to_timestamp(x):
    return datetime.strptime(x.strftime('%Y%m%d'), '%Y%m%d')


if __name__ == "__main__":

    dataset_train=pd.read_csv("C:/Users/Jaime Kershaw Brown/Documents/Final year project/TSLA_TEMP.csv")

    del dataset_train['Adj Close']

    cols = list(dataset_train)[1:6]
    
    datelist_train = list(dataset_train['Date'])
    datelist_train = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in datelist_train]

    print('Training set shape == {}'.format(dataset_train.shape))
    print('All timesteps == {}'.format(len(datelist_train)))
    print('Featured selected: {}'.format(cols))

    ### Data pre-processing

    dataset_train = dataset_train[cols].astype(str)
    # for i in cols:
    #     for j in range(0, len(dataset_train)):
    #         dataset_train[i][j] = dataset_train[i][j].replace(',', '')
        
    dataset_train = dataset_train.astype(float)

    training_set = dataset_train.to_numpy()

    print("shape of training set == {}".format(training_set.shape))
    print(training_set)

    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    sc_predict = MinMaxScaler(feature_range = (0, 1))
    sc_predict.fit_transform(training_set[:, 0:1])

    x_train = []
    y_train = []

    n_future = 1
    n_past = 70

    for i in range(n_past, len(training_set_scaled) - n_future + 1):
        x_train.append(training_set_scaled[i - n_past:i, 0:dataset_train.shape[1] - 1])
        y_train.append(training_set_scaled[i + n_future - 1:i + n_future, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)

    print("x_train shape =={}".format(x_train.shape))
    print("y_train shape =={}".format(y_train.shape))

    ### Model

    model = Sequential()

    model.add(LSTM(units=64, return_sequences=True, input_shape=(n_past, dataset_train.shape[1]-1)))

    model.add(LSTM(units=10, return_sequences=False))

    model.add(Dropout(0.25))

    model.add(Dense(units=1, activation='linear'))

    model.compile(optimizer = Adam(learning_rate=0.01), loss='mean_squared_error')

    ### training

    es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
    mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)

    tb = TensorBoard('logs')

    history = model.fit(x_train, y_train, shuffle=True, epochs=30, callbacks=[es, rlr, mcp, tb], validation_split=0.2, verbose=1, batch_size = 256)

    ### make predictions

    datelist_future = pd.date_range(datelist_train[-1], periods=n_future, freq='1d').tolist()

    datelist_future_ = []
    for this_timestamp in datelist_future:
        datelist_future_.append(this_timestamp.date())

    predictions_future = model.predict(x_train[-n_future:])
    predictions_train = model.predict(x_train[n_past:])

    predictions_future = sc_predict.inverse_transform(predictions_future)
    predictions_train = sc_predict.inverse_transform(predictions_train)

    y_pred_future = sc_predict.inverse_transform(predictions_future)
    y_pred_train = sc_predict.inverse_transform(predictions_train)

    predictions_FUTURE = pd.DataFrame(y_pred_future, columns=['Close']).set_index(pd.Series(datelist_future))
    prediction_TRAIN = pd.DataFrame(y_pred_train, columns=['Close']).set_index(pd.Series(datelist_train[2 * n_past + n_future - 1:]))

    prediction_TRAIN.index = prediction_TRAIN.index.to_series().apply(datetime_to_timestamp)

    print(predictions_FUTURE.head(1))
    
    # rcParams['figure.figsize'] = 12, 5

    # START_DATE_FOR_PLOTTING = '11-01-2018'
    # plt.plot(predictions_FUTURE.index, predictions_FUTURE['Close'], color='r', label='Predicted Stock Price')
    # plt.plot(prediction_TRAIN.loc[START_DATE_FOR_PLOTTING:].index, prediction_TRAIN.loc[START_DATE_FOR_PLOTTING:]['Close'], color='orange', label='Training predictions')
    # plt.plot(dataset_train.loc[START_DATE_FOR_PLOTTING:].index, dataset_train.loc[START_DATE_FOR_PLOTTING:]['Close'], color='b', label='Actual Stock Price')

    # plt.axvline(x = min(predictions_FUTURE.index), color='green', linestyle='--')
    # plt.grid(which='major', color='#cccccc', alpha=0.5)

    # plt.legend(shadow=True)
    # plt.title('Predictions and actual stock prices', family='Arial', fontsize=12)
    # plt.xlabel('Timeline', family='Arial', fontsize=10)
    # plt.xticks(rotation=45, fontsize=8)
    # plt.show()