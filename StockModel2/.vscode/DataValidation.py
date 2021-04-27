import math
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
import datetime
import os


def validateDate(date_text):
    result = ""
    try:
        datetime.datetime.strptime(date_text, "%d/%m/%Y") # "%Y-%m-%d" for all stocks folder
    except:
        result = "Fail"
        print("Incorrect date format, should be DD-MM-YYYY") 

def validateNumbers(data):
    result = ""
    
    try:
        if(isinstance(float(data[0]), float) and isinstance(float(data[1]), float) and isinstance(float(data[2]), float) and isinstance(float(data[3]), float) and isinstance(int(data[4]), int)):
            pass
        else:
            print("Number data does not match required format.")
            result = "Fail"
    except:
        print("Number data does not match required format.")
        result = "Fail"

    return result

def ValidateYahooCSVData(filePath, filename):
    """
    Check that data in file follows required format and won't cause any crashes.
    Required format: Date,Open,High,Low,Close,Volume,OpenInt
    Must not have any missing fields
    """

    # Check if file is empty, if is return file fail
    if(os.stat(filePath+"/"+filename).st_size == 0):
        print("File is empty!")
        return("Fail", 0)
    else:
        df=pd.read_csv(filePath+"/"+filename, sep=",")
        dataSet = df.iloc[:]
        if(len(dataSet.index) < 132):
            print("Not enough data in file!")
            return("Fail", 0)
        else:
            for index, row in dataSet.iterrows():
                # validateDate(row['Date'])             
                if(pd.isna(row['Open']) or pd.isna(row['High']) or pd.isna(row['Low']) or pd.isna(row['Close']) or pd.isna(row['Volume'])):
                    print("Null data in file")
                    return("Fail", 0)
                else:
                    data = [row['Open'], row['High'], row['Low'], row['Close'], row['Volume']]
                    result2 = validateNumbers(data)
                    if(result2 == "Fail"):
                        print("Numbers were not right type")
                        return("Fail", 0) 
                    
    return("Pass", len(dataSet.index))


def OrganiseTestingData(df): 

    storedTrainingSet = pd.read_csv(os.path.join(os.path.dirname(__file__), "../trainingData.txt"), header = None)
    trainingDataShape = storedTrainingSet.shape

    sc = MinMaxScaler(feature_range = (0, 1))
    trainingData = np.loadtxt(os.path.join(os.path.dirname(__file__), "../trainingData.txt")).reshape(trainingDataShape[0], trainingDataShape[1])
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

if __name__ == "__main__":
    # LoopThroughFiles()
    pass