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
    # plt.plot(XValues, YValues, marker = 'o', color = colour, label = dataTitle)
    plt.plot(XValues, YValues, marker = 'o', color = colour)
    # plt.plot(XValues, YValues, color = colour, label = dataTitle)

def TestingMultiStepDates(dateDataFrame, n_past, n_future, trainingDataLength, n_future_values):
    dates = dateDataFrame.tolist()
    multiStepDates = []

    for i in range(n_past + 2, len(dates) - n_future - n_future_values + 2):
        multiStepDates.append(dates[i + n_future - 1:i + n_future + n_future_values - 1])
    
    return multiStepDates

def TrainingMultiStepDates(dateDataFrame, n_past, n_future, trainingDataLength, n_future_values):
    
    dates = dateDataFrame.tolist()
    multiStepDates = []

    for i in range(n_past, trainingDataLength - n_future - n_future_values):
        multiStepDates.append(dates[i + n_future - 1:i + n_future + n_future_values - 1])
    
    return multiStepDates

def PredictOneIteration(n_past, n_future, n_future_values, historicData, model):
    ### Set up data indexing for dependent and independent variables
    x_train = []

    endOfData = len(historicData) - n_past

    ### Use past 50 days of all 5 columns to predict the next day of the 4th column (close)
    # for i in range(0, len(historicData) - n_future_values):
    x_train.append(historicData[endOfData:, 1:historicData.shape[1]])

    x_train = np.asarray(x_train).astype('float32')

    # print("x_train shape =={}".format(x_train.shape))


    scalerPredict = MinMaxScaler(feature_range = (0, 1))
    predictValues = historicData[endOfData:, 4]
    predictValues = predictValues.reshape(-1, 1)
    scalerPredict.fit_transform(predictValues)


    testDates = historicData[endOfData:, 0]
    # print(testDates[-1])
    newDay = AddExtraDay(testDates[-1])

    testYPoints = model.predict(x_train)
    testYPoints = scalerPredict.inverse_transform(testYPoints)

    testYPoints = np.array(testYPoints)
    singleXValue = np.array(newDay)

    # print("singleXValue {}".format(singleXValue))
    # print("testYPoints {}".format(testYPoints))

    PlotData(singleXValue, testYPoints, "purple", "predicted Data")

def PrepTrainingData(n_past, n_future, n_future_values, trainingData):
    
    ### Set up data indexing for dependent and independent variables
    x_train = []
    y_train = []

    ### Use past 50 days of all 5 columns to predict the next day of the 4th column (close)
    for i in range(n_past, len(trainingData) - n_future - n_future_values):
        x_train.append(trainingData[i - n_past:i, 0:trainingData.shape[1]])
        y_train.append(trainingData[i + n_future - 1:i + n_future + n_future_values - 1, 3])
        # print("x train {}".format(trainingData[i - n_past:i, 0:trainingData.shape[1]]))
        # print("y train {}".format(trainingData[i + n_future - 1:i + n_future + n_future_values, 3]))

    x_train, y_train = np.array(x_train), np.array(y_train)

    print("x_train shape =={}".format(x_train.shape))
    print("y_train shape =={}".format(y_train.shape))

    return x_train, y_train

def PrepTestingData(n_past, n_future, n_future_values, testingData):

    # print(testingData)

    xTest = []
    yTest = []
    for i in range(n_past + 1, len(testingData) - n_future - n_future_values + 1):
        # print("I HAPPENED")
        xTest.append(testingData[i - n_past:i, 0:testingData.shape[1]])
        yTest.append(testingData[i + n_future - 1:i + n_future + n_future_values - 1, 3])
        # print(xTest)
        # print(yTest)
    xTest, yTest = np.array(xTest), np.array(yTest)

    # print("xTest shape: {}".format(xTest.shape))

    return xTest, yTest

def TrainModel(n_past, x_train, y_train):

    print("Model input shape {}, {}".format(n_past, x_train.shape[2]))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(n_past, x_train.shape[2])))
    model.add(Dropout(0.05))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.05))
    model.add(Dense(units=5, activation=None))
    model.compile(loss='mean_squared_error', optimizer = 'adam')

    history = model.fit(x_train, y_train, epochs = 30, batch_size = 32)

    # plt.plot(history.history['loss'], label='train')
    # plt.legend()
    # plt.show()

    return model

def AddExtraDay(currentLastDate):
    oneDay = datetime.timedelta(days=1)
    weekendsAndHolidays = holidays.US()

    next_day = currentLastDate + oneDay
    while next_day.weekday() in holidays.WEEKEND or next_day in weekendsAndHolidays:
        next_day += oneDay
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

    print("mean squred error: " + str(mse))
    print("root mean squared error: " + str(rmse))

    increase = sum(actual[:]) - sum(predicted[:])
    increase = increase / sum(actual[:])
    print("Percentage change {}".format(abs(increase * 100)))

def TrainPrediction(model, x_train, trainingDataLength, dataset, y_train, n_past, n_future, n_future_values):

    trainPredict = model.predict(x_train)

    trainPredictionScaler = MinMaxScaler(feature_range=(0,1))
    trainPredictionDataRange = dataset.iloc[:trainingDataLength, 4].values
    trainPredictionDataRange = trainPredictionDataRange.reshape(-1, 1)
    trainPredictionScaler.fit_transform(trainPredictionDataRange)

    trainPredict = trainPredictionScaler.inverse_transform(trainPredict)
    trainActual = trainPredictionScaler.inverse_transform(y_train)

    EvaluateForecast(trainActual, trainPredict)

    # trainDates = dataset['Date'][ n_past + n_future:trainingDataLength]
    trainDates = TrainingMultiStepDates(dataset['Date'], n_past, n_future, trainingDataLength, n_future_values)

    # trainDates = trainDates.iloc[0:]

    # newDay = AddExtraDay(trainDates.iloc[-1])
    # trainDates.loc[trainingDataLength-1] = newDay
    

    # trainXPoints = np.array(dataset['Date'][ n_past + n_future:trainingDataLength])
    trainXPoints = np.array(trainDates)
    trainYPoints = np.array(trainPredict)
    PlotData(trainXPoints, trainYPoints, "green", "Training Data")

def TestPrediction(dataset, trainingDataLength, n_past, n_future, n_future_values, model):

    scaler = MinMaxScaler(feature_range=(0,1))
    testingData =  dataset.iloc[trainingDataLength:, 1:].values
    testingData = scaler.fit_transform(testingData)

    xTest, yTest = PrepTestingData(n_past, n_future, n_future_values, testingData)
    print(type(xTest))
    testPredict = model.predict(xTest)

    ### Test predictor scalar

    scalerPredict = MinMaxScaler(feature_range = (0, 1))
    predictValues = dataset.iloc[trainingDataLength + n_past:, 4].values
    predictValues = predictValues.reshape(-1, 1)
    scalerPredict.fit_transform(predictValues)

    testPredict = scalerPredict.inverse_transform(testPredict)
    testActual = scalerPredict.inverse_transform(yTest)

    print(testActual.shape)

    print(testPredict.shape)


    EvaluateForecast(testActual, testPredict)

 

    # testDates = dataset['Date'][trainingDataLength + n_past + n_future:]
    testDates = TestingMultiStepDates(dataset['Date'][trainingDataLength:], n_past, n_future, trainingDataLength, n_future_values)

    # testDates = testDates.iloc[1:]

    # newDay = AddExtraDay(testDates.iloc[-1])
    # testDates.loc[len(testingData)-1] = newDay

    # testXPoints = np.array(dataset['Date'][trainingDataLength + n_past + n_future:])
    testXPoints = np.array(testDates)
    testYPoints = np.array(testPredict)

    for i in range(len(testXPoints)):
        PlotData(testXPoints[i], testYPoints[i], "red", "Testing Data")


    # PlotData(testXPoints, testYPoints, "red", "Testing Data")


def TrainAndTest():
    dataset=pd.read_csv("C:/Users/Jaime Kershaw Brown/Documents/Final year project/TSLA.csv")

    del dataset['Adj Close']

    rowsToDrop = 0
    dataset.drop(dataset.tail(rowsToDrop).index,inplace=True) # drop last rowsToDrop rows


    dataset['Date'] = pd.to_datetime(dataset['Date'], format="%Y/%m/%d")
    # DisplayFullDataset(dataset)

    ### Plot actual data
    ActualXPoints = np.array(dataset['Date'])
    ActualYPoints = np.array(dataset['Close'])
    PlotData(ActualXPoints, ActualYPoints, "blue", "Actual Data")

    ### Data range for number of days to train with, and number of days to predict forward
    n_future = 1            # days forward from last day in history data
    n_future_values = 5     # number of days in to predict in vector format
    n_past = 60             # number of days to look at in the past
    

    print("dataset shape {}".format(dataset.shape))

    ### Set up training data
    trainingDataLength = math.floor(len(dataset.iloc[:, 1:2])*0.70)

    trainingData = dataset.iloc[:trainingDataLength, 1:].values
    print("training data shape {}".format(trainingData.shape))

    scaler = MinMaxScaler(feature_range=(0,1))
    trainingData = scaler.fit_transform(trainingData)

    x_train, y_train = PrepTrainingData(n_past, n_future, n_future_values, trainingData)

    ### Train Model
    model = TrainModel(n_past, x_train, y_train)

    ### Run and plot training data
    TrainPrediction(model, x_train, trainingDataLength, dataset, y_train, n_past, n_future, n_future_values)

    ### Testing data
    TestPrediction(dataset, trainingDataLength, n_past, n_future, n_future_values, model)

    ### Single value prediction
    # PredictOneIteration(n_past, n_future, n_future_values, dataset.iloc[trainingDataLength:, :].values)

    # for i in range(n_past, len(dataset) - n_future - n_future_values):
    #     PredictOneIteration(n_past, n_future, n_future_values, dataset.iloc[:i, :].values, model)
    #     # yTest.append(testingData[i + n_future - 1:i + n_future + n_future_values, 3])



    ShowGraph()


def TrainFullFile():
    dataset=pd.read_csv("C:/Users/Jaime Kershaw Brown/Documents/Final year project/stockTesting/AMZN.csv")

    del dataset['Adj Close']

    rowsToDrop = 0
    dataset.drop(dataset.tail(rowsToDrop).index,inplace=True) # drop last rowsToDrop rows


    dataset['Date'] = pd.to_datetime(dataset['Date'], format="%Y/%m/%d")

    ### Data range for number of days to train with, and number of days to predict forward
    n_future = 1            # days forward from last day in history data
    n_future_values = 0     # number of days in to predict in vector format
    n_past = 60             # number of days to look at in the past
    

    print("dataset shape {}".format(dataset.shape))
    # DisplayFullDataset(dataset)

    ### Set up training data
    trainingDataLength = math.floor(len(dataset.iloc[:, 1:2]))

    trainingData = dataset.iloc[:trainingDataLength, 1:].values
    print("training data shape {}".format(trainingData.shape))

    scaler = MinMaxScaler(feature_range=(0,1))
    trainingData = scaler.fit_transform(trainingData)

    x_train, y_train = PrepTrainingData(n_past, n_future, n_future_values, trainingData)

    ### Train Model
    model = TrainModel(n_past, x_train, y_train)
    model.save('C:/Users/Jaime Kershaw Brown/Documents/Final year project/MultivariateModel.h5')  # creates a HDF5 file 'my_model.h5'


def TestFullFile():

    dataset=pd.read_csv("C:/Users/Jaime Kershaw Brown/Documents/Final year project/stockTesting/TSLA.csv")

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
    n_past = 1             # number of days to look at in the past
    
    n_day_to_predict = 1

    print("dataset shape {}".format(dataset.shape))
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
    predictValues = dataset.iloc[trainingDataLength: math.floor(len(dataset.iloc[:, 1:2])) -1, 4].values
    print(dataset.tail(5))
    print("PREDICT VALUES")
    print(predictValues)
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


    ### Single value prediction
    # PredictOneIteration(n_past, n_future, n_future_values, dataset.iloc[trainingDataLength:, :].values)

    # for i in range(n_past, len(dataset) - n_future - n_future_values):
    #     PredictOneIteration(n_past, n_future, n_future_values, dataset.iloc[:i, :].values, model)
    #     # yTest.append(testingData[i + n_future - 1:i + n_future + n_future_values, 3])



    ShowGraph()


if __name__ == "__main__":
    pass
    TrainAndTest()
    # TrainFullFile()
    # TestFullFile()