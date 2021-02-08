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
from tensorflow.keras.models import load_model
from DataValidation import OrganiseTestingData

def PredictData(x_test): 

    model = load_model('Model_Test.h5')
    sc = MinMaxScaler(feature_range = (0, 1))
    storedTrainingSet = pd.read_csv('trainingData.txt', header = None)
    trainingDataShape = storedTrainingSet.shape
    trainingData = np.loadtxt("trainingData.txt").reshape(trainingDataShape[0], trainingDataShape[1])
    sc.fit_transform(trainingData)

    predicted_stock_price = model.predict(x_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    
    return predicted_stock_price

def TestUserModel():
    model = load_model("userModels/userModel.h5")

    df=pd.read_csv("C:/Users/Jaime Kershaw Brown/Documents/Final year project/Stock-Market-Prediction-using-LSTM-NN/StockModel2/TeslaTestData.csv")
    dateRange = df.loc[60:, 'Date']
    x_test = OrganiseTestingData(df)

    sc = MinMaxScaler(feature_range = (0, 1))
    storedTrainingSet = pd.read_csv('trainingData.txt', header = None)
    trainingDataShape = storedTrainingSet.shape
    trainingData = np.loadtxt("trainingData.txt").reshape(trainingDataShape[0], trainingDataShape[1])
    sc.fit_transform(trainingData)

    predicted_stock_price = model.predict(x_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    
    return predicted_stock_price, dateRange
