import math
from math import sqrt
import matplotlib.pyplot as plt
import tensorflow.keras
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
#from tensorflow.keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import os
from tensorflow.keras.optimizers import Adam
import holidays
import datetime

def ExtendDates(currentLastDate):
    ONE_DAY = datetime.timedelta(days=1)
    HOLIDAYS_US = holidays.US()

    next_day = currentLastDate + ONE_DAY
    while next_day.weekday() in holidays.WEEKEND or next_day in HOLIDAYS_US:
        next_day += ONE_DAY
    return next_day


def DisplayFullDataset(dataset):
    """

    """
    XPoints = np.array(dataset['Date'])
    YPoints = np.array(dataset['Close'])

    plt.plot(XPoints, YPoints, color = 'blue', label = 'All Stock Data')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

def EvaluateForecast(actual, predicted):

    mse = mean_squared_error(actual[:], predicted[:])
    rmse = sqrt(mse)

    print("mean squred error: " + str(mse / len(predicted)))
    print("root mean squared error: " + str(rmse / len(predicted)))

    increase = sum(actual[:]) - sum(predicted[:])
    increase = increase / sum(actual[:])
    print("Percentage change {}".format(abs(increase * 100)))

if __name__ == "__main__":

    dataset=pd.read_csv("C:/Users/Jaime Kershaw Brown/Documents/Final year project/TSLA.csv")

    del dataset['Adj Close']

    rowsToDrop = 150
    dataset.drop(dataset.tail(rowsToDrop).index,inplace=True) # drop last rowsToDrop rows


    dataset['Date'] = pd.to_datetime(dataset['Date'], format="%Y/%m/%d")

    
    

    print("dataset shape {}".format(dataset.shape))
    # DisplayFullDataset(dataset)

    ### Set up training data
    trainingDataLength = math.floor(len(dataset.iloc[:, 1:2])*0.7)

    trainingData = dataset.iloc[:trainingDataLength, 1:].values
    print("training data shape {}".format(trainingData.shape))

    scaler = MinMaxScaler(feature_range=(0,1))
    trainingData = scaler.fit_transform(trainingData)

    trainPredictionScaler = MinMaxScaler(feature_range=(0,1))
    trainPredictionDataRange = dataset.iloc[:trainingDataLength, 4].values
    trainPredictionDataRange = trainPredictionDataRange.reshape(-1, 1)
    trainPredictionScaler.fit_transform(trainPredictionDataRange)

    ### Set up data indexing for dependent and independent variables
    x_train = []
    y_train = []

    n_future = 1
    n_future_values = 0
    n_past = 50

    ### Use past 50 days of all 5 columns to predict the next day of the 4th column (close)
    for i in range(n_past, len(trainingData) - n_future - n_future_values):
        x_train.append(trainingData[i - n_past:i, 0:trainingData.shape[1]])
        y_train.append(trainingData[i + n_future - 1:i + n_future + n_future_values, 3])
        # print("x train {}".format(trainingData[i - n_past:i, 0:trainingData.shape[1]]))
        # print("y train {}".format(trainingData[i + n_future - 1:i + n_future + n_future_values, 3]))

    x_train, y_train = np.array(x_train), np.array(y_train)

    print("x_train shape =={}".format(x_train.shape))
    print("y_train shape =={}".format(y_train.shape))

# ### Model

    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(n_past, x_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation=None))
    model.compile(loss='mean_squared_error', optimizer = Adam(learning_rate=0.01))

    history = model.fit(x_train, y_train, epochs = 30, batch_size = 32)

    # plt.plot(history.history['loss'], label='train')
    # plt.legend()
    # plt.show()



    testingData =  dataset.iloc[trainingDataLength:, 1:].values
    testingData = scaler.fit_transform(testingData)

    scalerPredict = MinMaxScaler(feature_range = (0, 1))
    predictValues = dataset.iloc[trainingDataLength + n_past:, 4].values
    predictValues = predictValues.reshape(-1, 1)
    scalerPredict.fit_transform(predictValues)

    xTest = []
    yTest = []

    for i in range(n_past, len(testingData) - n_future - n_future_values):
        xTest.append(testingData[i - n_past:i, 0:testingData.shape[1]])
        yTest.append(testingData[i + n_future - 1:i + n_future + n_future_values, 3])

    xTest, yTest = np.array(xTest), np.array(yTest)

    print(x_train.shape)
    print(xTest.shape)

    trainPredict = model.predict(x_train)
    testPredict = model.predict(xTest)

    trainPredict = trainPredictionScaler.inverse_transform(trainPredict)
    trainActual = trainPredictionScaler.inverse_transform(y_train)

    EvaluateForecast(trainActual, trainPredict)

    testPredict = scalerPredict.inverse_transform(testPredict)
    testActual = scalerPredict.inverse_transform(yTest)

    EvaluateForecast(testActual, testPredict)

    ActualXPoints = np.array(dataset['Date'])
    ActualYPoints = np.array(dataset['Close'])

    trainDates = dataset['Date'][ n_past + n_future:trainingDataLength]
    trainDates = trainDates.iloc[0:]

    newDay = ExtendDates(trainDates.iloc[-1])
    trainDates.loc[trainingDataLength-1] = newDay
    

    # trainXPoints = np.array(dataset['Date'][ n_past + n_future:trainingDataLength])
    trainXPoints = np.array(trainDates)
    trainYPoints = np.array(trainPredict)

    print(len(trainPredict))
    print(len(dataset['Date'][ n_past + n_future:trainingDataLength]))

    testDates = dataset['Date'][trainingDataLength + n_past + n_future:]
    print("Initial dates length {}".format(len(testDates)))
    testDates = testDates.iloc[1:]
    print("dates length after remove first {}".format(len(testDates)))

    newDay = ExtendDates(testDates.iloc[-1])
    testDates.loc[len(testingData)-1] = newDay
    print("Final dates length {}".format(len(testDates)))

    # testXPoints = np.array(dataset['Date'][trainingDataLength + n_past + n_future:])
    testXPoints = np.array(testDates)
    testYPoints = np.array(testPredict)
    print(len(testPredict))
    print(len(dataset['Date'][trainingDataLength + n_past + n_future:]))

    plt.plot(ActualXPoints, ActualYPoints, color = 'blue', label = 'Actual Data')
    plt.plot(trainXPoints, trainYPoints, color = 'green', label = 'Training Data')
    plt.plot(testXPoints, testYPoints, color = 'red', label = 'Testing Data')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()