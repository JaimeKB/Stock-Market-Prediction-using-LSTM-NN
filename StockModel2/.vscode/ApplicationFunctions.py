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

def ShowGraph():
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

def PlotData(XValues, YValues, colour, dataTitle):
    plt.plot(XValues, YValues, marker = 'o', color = colour, label = dataTitle)
    # plt.plot(XValues, YValues, marker = 'o', color = colour)
    # plt.plot(XValues, YValues, color = colour, label = dataTitle)

def AddExtraDay(currentLastDate):
    oneDay = datetime.timedelta(days=1)
    weekendsAndHolidays = holidays.US()

    next_day = currentLastDate + oneDay
    while next_day.weekday() in holidays.WEEKEND or next_day in weekendsAndHolidays:
        next_day += oneDay
    return next_day

def PrepMultiStepTestingData(n_past, n_future, n_future_values, dataset):
    
    ### Set up data indexing for dependent and independent variables
    x_test = []

    ### Use past 60 days of all 5 columns to predict the next 5 days of the 4th column (close)
    for i in range(n_past, len(dataset)+1):
        x_test.append(dataset[i - n_past:i, 0:dataset.shape[1]])

    x_test = np.array(x_test)


    # print("multistep x_train shape =={}".format(x_test.shape))
    # print("y_train shape =={}".format(y_train.shape))

    return x_test


def PrepTestingData(n_past, n_future, n_future_values, testingData):

    xTest = []
    yTest = []
    for i in range(n_past + 1, len(testingData) - n_future - n_future_values + 1):
        # print("I HAPPENED")
        xTest.append(testingData[i - n_past:i, 0:testingData.shape[1]])
        yTest.append(testingData[i + n_future - 1:i + n_future + n_future_values, 3])
        # print(xTest)
        # print(yTest)
    xTest, yTest = np.array(xTest), np.array(yTest)

    # print("xTest shape: {}".format(xTest.shape))
    # print("xTest: {}".format(xTest))

    return xTest, yTest


def EvaluateForecast(actual, predicted):

    mse = mean_squared_error(actual[:], predicted[:])
    rmse = sqrt(mse)

    print("mean squred error: " + str(mse))
    print("root mean squared error: " + str(rmse))

    total = 0
    temp = 0

    for valueIndex in range(len(actual)):
        temp = math.sqrt((actual[valueIndex] - predicted[valueIndex]) ** 2)
        total += temp

    meanDifference = total / len(actual)

    print("Mean difference: {}".format(float(meanDifference)))

    increase = sum(actual[:]) - sum(predicted[:])
    increase = increase / sum(actual[:])
    print("Percentage change {}".format(abs(increase * 100)))

def MultiStepFutureValues(dataset):

    # Data range for number of days to train with, and number of days to predict forward
    n_future = 1            # days forward from last day in history data
    n_future_values = 5     # number of days in to predict in vector format
    n_past = 20             # number of days to look at in the past

    # number of days to look at in the past
    n_day_to_predict = 1
   
    trainingDataLength = math.floor(len(dataset.iloc[:, 1:2])) - n_past - n_day_to_predict + 1

    model = load_model('C:/Users/Jaime Kershaw Brown/Documents/Final year project/MultivariateMultiStepModel.h5')

    scaler = MinMaxScaler(feature_range=(0,1))
    testingData = dataset.iloc[trainingDataLength:, 1:].values

    testingData = scaler.fit_transform(testingData)

    xTest = PrepMultiStepTestingData(n_past, n_future, n_future_values, testingData)
    futureValues = model.predict(xTest)
 
    ### Test predictor scalar

    scalerPredict = MinMaxScaler(feature_range = (0, 1))
    predictValues = dataset.iloc[trainingDataLength+15: math.floor(len(dataset.iloc[:, 1:2])) -1, 4].values

    predictValues = predictValues.reshape(-1, 1)
    scalerPredict.fit_transform(predictValues)

    futureValues = scalerPredict.inverse_transform(futureValues)

    currentDay = dataset['Date'].iloc[-1]

    futureDates = []

    for i in range(0, n_future_values):
        newDay = AddExtraDay(currentDay)
        futureDates.append(newDay)
        currentDay = newDay


    testXPoints = np.array(futureDates)
    tempTestYPoints = np.array(futureValues)

    testYPoints = []
    for i in range(len(testXPoints)):
        testYPoints.append(tempTestYPoints[0][i])

    testYPoints = np.array(testYPoints)

    return testYPoints, testXPoints

def TestFullFile():

    dataset=pd.read_csv("C:/Users/Jaime Kershaw Brown/Documents/Final year project/TSLA.csv")

    del dataset['Adj Close']


    rowsToDrop = 0
    dataset.drop(dataset.tail(rowsToDrop).index,inplace=True) # drop last rowsToDrop rows


    dataset['Date'] = pd.to_datetime(dataset['Date'], format="%Y/%m/%d")

    ### Plot actual data
    ActualXPoints = np.array(dataset['Date'])
    ActualYPoints = np.array(dataset['Close'])
    PlotData(ActualXPoints, ActualYPoints, "blue", "Actual Data")

    ### Data range for number of days to train with, and number of days to predict forward
    n_future = 1            # days forward from last day in history data
    n_future_values = 0     # number of days in to predict in vector format
    n_past = 60              # number of days to look at in the past
    
    n_day_to_predict = 30

    # trainingDataLength = math.floor(len(dataset.iloc[:, 1:2])*0.9)
    trainingDataLength = math.floor(len(dataset.iloc[:, 1:2])) - n_past - n_future - n_future_values - n_day_to_predict

    model = load_model('C:/Users/Jaime Kershaw Brown/Documents/Final year project/MultivariateModel.h5')
    scaler = MinMaxScaler(feature_range=(0,1))
    testingData = dataset.iloc[trainingDataLength:, 1:].values
    testingData = scaler.fit_transform(testingData)
    xTest, yTest = PrepTestingData(n_past, n_future, n_future_values, testingData)
    testPredict = model.predict(xTest)

    ### Test predictor scalar

    scalerPredict = MinMaxScaler(feature_range = (0, 1))
    predictValues = dataset.iloc[trainingDataLength+5: math.floor(len(dataset.iloc[:, 1:2])) -1, 4].values
    predictValues = predictValues.reshape(-1, 1)
    scalerPredict.fit_transform(predictValues)

    testPredict = scalerPredict.inverse_transform(testPredict)
    testActual = scalerPredict.inverse_transform(yTest)

    EvaluateForecast(testActual, testPredict)

 

    testDates = dataset['Date'][trainingDataLength + n_past + n_future:]
    # testDates = testDates.iloc[1:]

    # newDay = AddExtraDay(testDates.iloc[-1])
    # testDates.loc[len(testingData)-1] = newDay

    # testXPoints = np.array(dataset['Date'][trainingDataLength + n_past + n_future:])
    testXPoints = np.array(testDates)
    testYPoints = np.array(testPredict)
    PlotData(testXPoints, testYPoints, "red", "Testing Data")

    futureYPoints, futureXPoints = MultiStepFutureValues(dataset)

    PlotData(futureXPoints, futureYPoints, "purple", "Future values")

    ShowGraph()


if __name__ == "__main__":

    TestFullFile()