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
    plt.ylabel('Stock Value')
    plt.legend()
    plt.show()

def PlotData(XValues, YValues, colour, dataTitle):
    # plt.plot(XValues, YValues, marker = 'o', color = colour, label = dataTitle)
    plt.plot(XValues, YValues, marker = 'o', color = colour)
    # plt.plot(XValues, YValues, color = colour, label = dataTitle)

def EvaluateForecast(actual, predicted):

    mse = mean_squared_error(actual[:], predicted[:])
    rmse = sqrt(mse)

    print("mean squred error: " + str(mse))
    print("root mean squared error: " + str(rmse))

    increase = sum(actual[:]) - sum(predicted[:])
    increase = increase / sum(actual[:])
    print("Percentage change {}".format(abs(increase * 100)))

def TestingMultiStepDates(dateDataFrame, n_past, n_future, trainingDataLength, n_future_values):
    dates = dateDataFrame.tolist()
    multiStepDates = []

    for i in range(n_past + 2, len(dates) - n_future - n_future_values + 2):
        multiStepDates.append(dates[i + n_future - 1:i + n_future + n_future_values - 1])
    
    return multiStepDates

def PrepTestingData(n_past, n_future, n_future_values, trainingData):
    
    ### Set up data indexing for dependent and independent variables
    x_test = []
    y_test = []

    ### Use past 60 days of all 5 columns to predict the next 5 days of the 4th column (close)
    for i in range(n_past, len(trainingData) - n_future_values + 1):
        x_test.append(trainingData[i - n_past:i, 0:trainingData.shape[1]])
        y_test.append(trainingData[i + n_future - 1:i + n_future + n_future_values - 1, 3])

    x_test, y_test = np.array(x_test), np.array(y_test)

    # print("x_train shape =={}".format(x_train.shape))
    # print("y_train shape =={}".format(y_train.shape))

    return x_test, y_test


def FormatTestingFileData(dataset):
    del dataset['Adj Close']
    rowsToDrop = 0
    dataset.drop(dataset.tail(rowsToDrop).index,inplace=True) # drop last rowsToDrop rows
    return dataset

    dataset['Date'] = pd.to_datetime(dataset['Date'], format="%Y/%m/%d")

def MakeModel(n_past, x_train, y_train, n_future_values, layerOneHiddenUnits, layerTwoHiddenUnits, layerOneDropout, layerTwoDropout, numberOfEpochs):

    x_train, y_train = np.array(x_train), np.array(y_train)

    print("Model input shape {}, {}".format(n_past, x_train.shape[2]))

    model = Sequential()
    model.add(LSTM(units=layerOneHiddenUnits, return_sequences=True, input_shape=(n_past, x_train.shape[2])))
    model.add(Dropout(layerOneDropout))
    model.add(LSTM(units=layerTwoHiddenUnits, return_sequences=False))
    model.add(Dropout(layerTwoDropout))
    model.add(Dense(units=n_future_values, activation=None))
    model.compile(loss='mean_squared_error', optimizer = Adam(learning_rate=0.01))

    history = model.fit(x_train, y_train, epochs = numberOfEpochs, batch_size = 32)

    # plt.plot(history.history['loss'], label='train')
    # plt.legend()
    # plt.show()

    return model

def FormatTrainingData(trainingData, n_future, n_future_values, n_past):

    ### Set up data indexing for dependent and independent variables
    x_train = []
    y_train = []

    ### Use past 60 days of all 5 columns to predict the next 5 days of the 4th column (close)
    for i in range(n_past, len(trainingData) - n_future_values + 1):
        x_train.append(trainingData[i - n_past:i, 0:trainingData.shape[1]])
        y_train.append(trainingData[i + n_future - 1:i + n_future + n_future_values - 1, 3])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # print("x_train shape =={}".format(x_train.shape))
    # print("y_train shape =={}".format(y_train.shape))

    return x_train, y_train

def SmoothAndFormatData(dataSet, n_future, n_future_values, n_past):
    
    scalingWindow = 250

    scaledXData = []
    scaledYData = []

    numberOfBatches = len(dataSet.iloc[:, 1:2]) // scalingWindow
    print("number of batches {}".format(numberOfBatches))

    for index in range(0, (numberOfBatches * scalingWindow), scalingWindow):
        scaler = MinMaxScaler(feature_range=(0,1))
        trainingDataSegment = scaler.fit_transform(dataSet[index:index+scalingWindow])
        xTrainingData, yTrainingData = FormatTrainingData(trainingDataSegment, n_future, n_future_values, n_past)
        scaledXData.extend(xTrainingData)
        scaledYData.extend(yTrainingData)

    print("remaining data length: {}".format(len(dataSet[index+scalingWindow:])))
    if(len(dataSet[index+scalingWindow:]) > n_past):
        scaler = MinMaxScaler(feature_range=(0,1))
        trainingDataSegment = scaler.fit_transform(dataSet[index+scalingWindow:])
        xTrainingData, yTrainingData = FormatTrainingData(trainingDataSegment, n_future, n_future_values, n_past)
        scaledXData.extend(xTrainingData)
        scaledYData.extend(yTrainingData)

    return scaledXData, scaledYData

def FormatFileData(dataSet):

    del dataSet['Adj Close']
    del dataSet['Date']
    return dataSet

def OrganiseDataset(filePath, n_future, n_future_values, n_past):

    totalTrainingXSet = []
    totalTrainingYSet = []

    directory = os.fsencode(filePath)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"): 
            print(filename)
            df=pd.read_csv(filePath+"/"+filename, sep=",")
            dataSet = df.iloc[:]
            dataSet = FormatFileData(dataSet)
            print(dataSet.shape)
            trainingXSet, trainingYSet = SmoothAndFormatData(dataSet, n_future, n_future_values, n_past)
            totalTrainingXSet.extend(trainingXSet)
            totalTrainingYSet.extend(trainingYSet)

    return totalTrainingXSet, totalTrainingYSet

def TrainModel(filePath, n_future, n_future_values, n_past, layerOneHiddenUnits, layerTwoHiddenUnits, layerOneDropout, layerTwoDropout, epochs):

    totalTrainingXSet, totalTrainingYSet = OrganiseDataset(filePath, n_future, n_future_values, n_past)

    ### Train Model
    model = MakeModel(n_past, totalTrainingXSet, totalTrainingYSet, n_future_values, layerOneHiddenUnits, layerTwoHiddenUnits, layerOneDropout, layerTwoDropout, epochs)
    model.save('C:/Users/Jaime Kershaw Brown/Documents/Final year project/MultivariateMultiStepModel.h5')  # creates a HDF5 file 'MultivariateMultiStepModel.h5'



def TestModel(dataset, n_future, n_future_values, n_past):

    dataset = FormatTestingFileData(dataset)

    ### Plot full actual data
    ActualXPoints = np.array(dataset['Date'])
    ActualYPoints = np.array(dataset['Close'])
    PlotData(ActualXPoints, ActualYPoints, "blue", "Actual Data")

    # number of days to look at in the past
    n_day_to_predict = 100

    print("dataset shape {}".format(dataset.shape))
   
    trainingDataLength = math.floor(len(dataset.iloc[:, 1:2])) - n_past - n_future - n_future_values - n_day_to_predict

    model = load_model('C:/Users/Jaime Kershaw Brown/Documents/Final year project/MultivariateMultiStepModel.h5')

    scaler = MinMaxScaler(feature_range=(0,1))
    testingData = dataset.iloc[trainingDataLength:, 1:].values
    testingData = scaler.fit_transform(testingData)

    xTest, yTest = PrepTestingData(n_past, n_future, n_future_values, testingData)
    testPredict = model.predict(xTest)
 
    ### Test predictor scalar

    scalerPredict = MinMaxScaler(feature_range = (0, 1))
    predictValues = dataset.iloc[trainingDataLength: math.floor(len(dataset.iloc[:, 1:2])) -1, 4].values

    predictValues = predictValues.reshape(-1, 1)
    scalerPredict.fit_transform(predictValues)

    testPredict = scalerPredict.inverse_transform(testPredict)
    testActual = scalerPredict.inverse_transform(yTest)

    print(testActual)
    print(testPredict)

    EvaluateForecast(testActual, testPredict)

 

    testDates = TestingMultiStepDates(dataset['Date'][trainingDataLength:], n_past, n_future, trainingDataLength, n_future_values)
    # testDates = testDates.iloc[1:]

    # newDay = AddExtraDay(testDates.iloc[-1])
    # testDates.loc[len(testingData)-1] = newDay

    # testXPoints = np.array(dataset['Date'][trainingDataLength + n_past + n_future:])
    testXPoints = np.array(testDates)
    testYPoints = np.array(testPredict)

    for i in range(len(testXPoints)):
        PlotData(testXPoints[i], testYPoints[i], "red", "Testing Data")

    plt.xticks(np.arange(0, len(ActualXPoints), 200))

    # PlotData(testXPoints, testYPoints, "red", "Testing Data")


    ### Single value prediction
    # PredictOneIteration(n_past, n_future, n_future_values, dataset.iloc[trainingDataLength:, :].values)

    # for i in range(n_past, len(dataset) - n_future - n_future_values):
    #     PredictOneIteration(n_past, n_future, n_future_values, dataset.iloc[:i, :].values, model)
    #     # yTest.append(testingData[i + n_future - 1:i + n_future + n_future_values, 3])



    ShowGraph()



if __name__ == "__main__":

    folderPath = "C:/Users/Jaime Kershaw Brown/Documents/Final year project/stockTesting"
    
    # Data range for number of days to train with, and number of days to predict forward
    n_future = 1            # days forward from last day in history data
    n_future_values = 5     # number of days in to predict in vector format
    n_past = 20             # number of days to look at in the past
    
    layerOneHiddenUnits = 100
    layerTwoHiddenUnits = 50
    layerOneDropout = 0.2
    layerTwoDropout = 0.4
    epochs = 5

    testingData=pd.read_csv("C:/Users/Jaime Kershaw Brown/Documents/Final year project/TSLA.csv")


    TrainModel(folderPath, n_future, n_future_values, n_past, layerOneHiddenUnits, layerTwoHiddenUnits, layerOneDropout, layerTwoDropout, epochs)
    TestModel(testingData, n_future, n_future_values, n_past)