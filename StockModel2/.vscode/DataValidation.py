import math
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

def ValidateData():
    pass


def OrganiseTestingData(df): 

    storedTrainingSet = pd.read_csv('trainingData.txt', header = None)
    trainingDataShape = storedTrainingSet.shape

    sc = MinMaxScaler(feature_range = (0, 1))
    trainingData = np.loadtxt("trainingData.txt").reshape(trainingDataShape[0], trainingDataShape[1])
    sc.fit_transform(trainingData)
    testingDataLength = math.floor(len(df.iloc[:, 1:2]))

    tempVal1 = df.iloc[:10, 1:2]
    tempVal2 = df.iloc[10:, 1:2]

    dataSet = df.iloc[:, 1:2]

    dataset_total = pd.concat((tempVal1, tempVal2), axis = 0)
    inputs = dataset_total.values

    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)
    x_test = []
    for i in range(60, testingDataLength):
        x_test.append(inputs[i-60:i, 0])
    x_test = np.array(x_test)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_test