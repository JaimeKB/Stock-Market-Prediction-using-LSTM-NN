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
from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import os

# def MultiFileReading():
#     filePath = "C:/Users/Jaime Kershaw Brown/Documents/Final year project/trainingStocks"
#     data = []
    
#     directory = os.fsencode(filePath)
#     for file in os.listdir(directory):
#         filename = os.fsdecode(file)
#         if filename.endswith(".csv"): 
#             print(filename)
#             df=pd.read_csv(filePath+"/"+filename, sep=",")
#             data = df.iloc[:]
               
def OrganiseTrainingData(df): 
    #split data
    

    trainingDataLength = math.floor(len(df.iloc[:, 1:2])*0.7)

    training_set = df.iloc[:trainingDataLength, 1:2].values
    # 881 for training
    # 387 for testing
    
    return trainingDataLength, training_set


def OrganiseTestingData(df, trainingDataLength, sc): 
    
    # Get length of data in file
    dataLength = len(df.iloc[:, 1:2])

    dataset_train = df.iloc[:trainingDataLength, 1:2]
    dataset_test = df.iloc[trainingDataLength:, 1:2]

    # print("Dataset_train: " + str(dataset_train.shape))
    # print("Dataset_test: " + str(dataset_test.shape))

    dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)

    print(dataset_total.shape)

    # print("Dataset_total: " + str(dataset_total.shape))

    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    # print("inputs length: " + str(len(inputs)))

    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)
    print(inputs.shape)
    x_test = []
    y_test = []

    for i in range(60, dataLength - trainingDataLength + 60):
        x_test.append(inputs[i-60:i, 0])
        y_test.append(inputs[i, 0])

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    print(x_test.shape)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # print(len(x_test))
    return x_test, dataset_test, y_test

def TrainWithUnivariateInputAndVectorOutput():
    training_set_scaled = sc.fit_transform(training_set)

    # data structure with 60 time-steps and 1 output
    x_train = []
    y_train = []
    for i in range(60, trainingDataLength):
        x_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # print("x_train shape:")
    # print(x_train.shape)
    # (821, 60, 1)

    model = Sequential()
    # Layer 1
    model.add(LSTM(units = 60, return_sequences = True, input_shape = (x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    # Layer 2
    model.add(LSTM(units = 60, return_sequences = True))
    model.add(Dropout(0.2))
    # Layer 3
    model.add(LSTM(units = 60, return_sequences = True))
    model.add(Dropout(0.2))
    # Layer 4
    model.add(LSTM(units = 60))
    model.add(Dropout(0.2))
    # Output layer
    model.add(Dense(units = 1))
    # Compile and fit model to training dataset
    model.compile(optimizer = 'adam', loss= 'mean_squared_error')
    model.fit(x_train, y_train, epochs = 100, batch_size = 32)
    model.save('C:/Users/Jaime Kershaw Brown/Documents/Final year project/Stock-Market-Prediction-using-LSTM-NN/StockModel2/Model_Test.h5')  # creates a HDF5 file 'my_model.h5'

    return model



def TrainModel(trainingDataLength, training_set, sc): 

    training_set_scaled = sc.fit_transform(training_set)

    # data structure with 60 time-steps and 1 output
    x_train = []
    y_train = []
    for i in range(60, trainingDataLength):
        x_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # print("x_train shape:")
    # print(x_train.shape)
    # (821, 60, 1)

    model = Sequential()
    # Layer 1
    model.add(LSTM(units = 100, return_sequences = True, input_shape = (x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    # Layer 2
    model.add(LSTM(units = 50, return_sequences = False))
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(units = 1))
    # Compile and fit model to training dataset
    model.compile(loss='mean_squared_error', optimizer = Adam(learning_rate=0.01))
    model.fit(x_train, y_train, epochs = 50, batch_size = 32)
    model.save('C:/Users/Jaime Kershaw Brown/Documents/Final year project/Stock-Market-Prediction-using-LSTM-NN/StockModel2/Model_Test.h5')  # creates a HDF5 file 'my_model.h5'

    return model

def PredictData(model, x_test, y_test, sc): 

    predicted_stock_price = model.predict(x_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    return predicted_stock_price

def EvaluateForecast(actual, predicted):

    mse = mean_squared_error(actual[:], predicted[:])
    rmse = sqrt(mse)

    print("mean squred error: " + str(mse / len(predicted)))
    print("root mean squared error: " + str(rmse / len(predicted)))


if __name__ == "__main__":

    df=pd.read_csv("C:/Users/Jaime Kershaw Brown/Documents/Final year project/TSLA.csv")

    # Scale/normalise data
    sc = MinMaxScaler(feature_range = (0, 1))

    trainingDataLength, training_set = OrganiseTrainingData(df)
    # f = open("trainingData.txt", "w")
    # for row in training_set:
    #     np.savetxt(f, row)
    # f.close()

    storedTrainingSet = pd.read_csv('C:/Users/Jaime Kershaw Brown/Documents/Final year project/Stock-Market-Prediction-using-LSTM-NN/StockModel2/trainingData.txt', header = None)


    trainingDataShape = storedTrainingSet.shape

    #model = load_model('C:/Users/Jaime Kershaw Brown/Documents/Final year project/Stock-Market-Prediction-using-LSTM-NN/StockModel2/Model_Test.h5')
    # Training function
    model = TrainModel(trainingDataLength, training_set, sc)
    
    trainingData = np.loadtxt("C:/Users/Jaime Kershaw Brown/Documents/Final year project/Stock-Market-Prediction-using-LSTM-NN/StockModel2/trainingData.txt").reshape(trainingDataShape[0], trainingDataShape[1])

    sc.fit_transform(trainingData)
    
    # Prepare testing data Function
    x_test, dataset_test, y_test = OrganiseTestingData(df, trainingDataLength, sc)

    # Predicted data
    # predicted_stock_price = model.predict(x_test)
    # predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    predicted_stock_price = PredictData(model, x_test, y_test, sc)
    # Visualise the results

    print(df.loc[trainingDataLength:, 'Date'])
    print(predicted_stock_price.shape)

    EvaluateForecast(dataset_test, predicted_stock_price)

    plt.plot(df.loc[trainingDataLength:, 'Date'], dataset_test.values, color = 'red', label = 'Actual Stock Price')
    plt.plot(df.loc[trainingDataLength:, 'Date'], predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
    plt.xticks(np.arange(0, len(x_test), 100))
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()