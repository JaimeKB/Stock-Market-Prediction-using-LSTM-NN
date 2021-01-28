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

# def TrainModel(trainingDataLength, training_set, sc): 

#     training_set_scaled = sc.fit_transform(training_set)

#     # data structure with 60 time-steps and 1 output
#     x_train = []
#     y_train = []
#     for i in range(60, trainingDataLength):
#         x_train.append(training_set_scaled[i-60:i, 0])
#         y_train.append(training_set_scaled[i, 0])

#     x_train = np.array(x_train)
#     y_train = np.array(y_train)

#     x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#     print("x_train shape:")
#     print(x_train.shape)
#     # (821, 60, 1)

#     model = Sequential()
#     # Layer 1
#     model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
#     model.add(Dropout(0.2))
#     # Layer 2
#     model.add(LSTM(units = 50, return_sequences = True))
#     model.add(Dropout(0.2))
#     # Layer 3
#     model.add(LSTM(units = 50, return_sequences = True))
#     model.add(Dropout(0.2))
#     # Layer 4
#     model.add(LSTM(units = 50))
#     model.add(Dropout(0.2))
#     # Output layer
#     model.add(Dense(units = 1))
#     # Compile and fit model to training dataset
#     model.compile(optimizer = 'adam', loss= 'mean_squared_error')
#     model.fit(x_train, y_train, epochs = 5, batch_size = 32) # changing epochs from 100 to 5 for coding purposes

#     return model