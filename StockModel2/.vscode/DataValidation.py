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

# def OrganiseTrainingData(df): 
#     #split data
#     dataLength = len(df.iloc[:, 1:2])

#     trainingDataLength = math.floor(len(df.iloc[:, 1:2])*0.7)

#     training_set = df.iloc[:trainingDataLength, 1:2].values
#     # 881
#     test_set = df.iloc[trainingDataLength:, 1:2].values
#     # 387

#     print("Traing length: " + str(len(training_set)))
#     print("test length: " + str(len(test_set)))

#     # Scale/normalise data
#     sc = MinMaxScaler(feature_range = (0, 1))
    
#     return dataLength, trainingDataLength, training_set, test_set, sc

def OrganiseTestingData(df, trainingDataLength, sc): 

    # Get length of data in file
    dataLength = len(df.iloc[:, 1:2])

    dataSet = df.iloc[:, 1:2]

    dataset_total = pd.concat((dataSet), axis = 0)
    inputs = dataset_total[len(dataset_total) - 60:].values

    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)
    x_test = []
    for i in range(60, dataLength - trainingDataLength + 60):
        x_test.append(inputs[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_test