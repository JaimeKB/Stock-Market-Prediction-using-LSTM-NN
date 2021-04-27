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


def FindKnownSectorStocks():

    filePath = "C:/Users/Jaime Kershaw Brown/Documents/Final year project/stocks"
    
    stockNames = []


    directory = os.fsencode(filePath)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"): 
            size = len(filename)
            filenameTemp = filename[:size - 4]
            stockNames.append(filenameTemp)
            
    print("Currently available stocks: {}".format(len(stockNames)))

    stockSectorsDF=pd.read_csv("C:/Users/Jaime Kershaw Brown/Documents/Final year project/stock_symbols_sectors.csv")
    del stockSectorsDF['Name']
    del stockSectorsDF['Last Sale']
    del stockSectorsDF['Net Change']
    del stockSectorsDF['% Change']
    del stockSectorsDF['Market Cap']
    del stockSectorsDF['Country']
    del stockSectorsDF['IPO Year']
    del stockSectorsDF['Volume']
    del stockSectorsDF['Industry']

    StocksWithSectors = stockSectorsDF['Symbol'].tolist()

    stocksAndSector = stockSectorsDF.values.tolist()

    # stocksWithNoSector = []

    # for stock in stocksAndSector:
    #     if(stock[0] in stockNames):
    #         if(pd.isna(stock[1])):
    #             stocksWithNoSector.append(stock[0])

    # print(len(stocksWithNoSector))

    # directory = os.fsencode(filePath)
    # for file in os.listdir(directory):
    #     filename = os.fsdecode(file)
    #     if filename.endswith(".csv"): 
    #         size = len(filename)
    #         filenameTemp = filename[:size - 4]
    #         if filenameTemp in stocksWithNoSector:
    #             print(filenameTemp)
    #             if os.path.exists(filePath+"/"+filename):
    #                 os.remove(filePath+"/"+filename)
    #             else:
    #                 print("The file does not exist")

    for stock in stocksAndSector:
        if(stock[0] in stockNames):
            print(stock[0])
            # df = pd.read_csv("C:/Users/Jaime Kershaw Brown/Documents/Final year project/stocks/"+stock[0]+".csv")
            # df.to_csv('C:/Users/Jaime Kershaw Brown/Documents/Final year project/SectorStocks/'+stock[1]+'/'+stock[0]+'.csv', index=False)


    # print(stocksAndSector)

    # validStocks = set(stockNames).intersection(StocksWithSectors)
    # print("Stocks with sectors: {}".format(len(validStocks)))




    # directory = os.fsencode(filePath)
    # for file in os.listdir(directory):
    #     filename = os.fsdecode(file)
    #     if filename.endswith(".csv"): 
    #         size = len(filename)
    #         filenameTemp = filename[:size - 4]

            



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
        if(len(dataSet.index) < 252):
            print("Not enough data in file!")
            return("Fail", 0)
        # else:
        #     for index, row in dataSet.iterrows():
        #         validateDate(row['Date'])             
        #         if(pd.isna(row['Open']) or pd.isna(row['High']) or pd.isna(row['Low']) or pd.isna(row['Close']) or pd.isna(row['Volume'])):
        #             print("Null data in file")
        #             return("Fail", 0)
        #         else:
        #             data = [row['Open'], row['High'], row['Low'], row['Close'], row['Volume']]
        #             result2 = validateNumbers(data)
        #             if(result2 == "Fail"):
        #                 print("Numbers were not right type")
        #                 return("Fail", 0) 
                    
    return("Pass", len(dataSet.index))

def ValidateNasdaqData(filename):
    """
    Check Nasdaq csv data file follows required format and won't cause any crashes.
    Required format: Date,Open,High,Low,Close,Volume,OpenInt
    Must not have any missing fields
    """
    df=pd.read_csv("C:/Users/Jaime Kershaw Brown/Documents/Final year project/stock_comparison/nasdaq/"+filename)
    dataSet = df.iloc[:]
    fileResult = ""
    print(dataSet)
    for index, row in dataSet.iterrows():
        # result = validateDate(row['Date']) 
        if(row[' Close/Last'] == " N/A" or row[' Volume'] == " N/A" or row[' Open'] == " N/A" or row[' High'] == " N/A" or row[' Low'] == " N/A"):
            pass
        else:
            data = [float(row[' Close/Last'][2:]), int(row[' Volume']), float(row[' Open'][2:]), float(row[' High'][2:]), float(row[' Low'][2:])]

    # print(data)
    # print(type(data[3]))
    
    #     result2 = validateNumbers(data)
    #     if(result == "Fail" or result2 == "Fail"):
    #         fileResult = "Fail"
    # if(fileResult == "Fail"):
    #     print("This file fails")
    # else:
    #     print("This file passes")    

def ValidateTXTData(filename):
    """
    Check that data in txt file follows required format and won't cause any crashes.
    Required format: Date,Open,High,Low,Close,Volume,OpenInt
    Must not have any missing fields
    """
    fileResult = ""
    if(os.stat('C:/Users/Jaime Kershaw Brown/Documents/Final year project/huge_stock_market_dataset/Stocks/'+filename).st_size == 0):
        print("File is empty!")
        fileResult = "Fail"
    else:
        dataSet = pd.read_csv('C:/Users/Jaime Kershaw Brown/Documents/Final year project/huge_stock_market_dataset/Stocks/'+filename, sep=",")
        dataSet.columns = ["Date", "Open", "High", "Low", "Close", "Volume", "OpenInt"]
        
        if(len(dataSet.index) < 132):
            print("Not enough data in file!")
            fileResult = "Fail"
        else:
            for index, row in dataSet.iterrows():
                validateDate(row['Date']) 
                data = [row['Open'], row['High'], row['Low'], row['Close'], row['Volume']]        
                result2 = validateNumbers(data)
                if(result2 == "Fail"):
                    fileResult = "Fail"

    if(fileResult == "Fail"):
        print("Fail")
        return "Fail"
    else:
        return "Pass"


def LoopThroughFiles():
    """
    Function to loop through files for validation
    """
    filePath = "C:/Users/Jaime Kershaw Brown/Documents/Final year project/stocks"
    failedFiles = 0
    passedFiles = 0
    fileLineNumbers = 0
    totalLineNumbers = 0
    now = datetime.datetime.now()

    StartTime = now.strftime("%H:%M:%S")

    directory = os.fsencode(filePath)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"): 
            print(filename)
            result, fileLineNumbers = ValidateYahooCSVData(filePath, filename)
            if(result == "Fail"):
                print("This file fails")
                failedFiles +=1
                if os.path.exists(filePath+"/"+filename):
                    os.remove(filePath+"/"+filename)
                else:
                    print("The file does not exist")
            else:
                print("this file passes")
                passedFiles +=1
                totalLineNumbers += fileLineNumbers
                fileLineNumbers = 0
        else:
            pass

    now = datetime.datetime.now()
    endTime = now.strftime("%H:%M:%S")

    print("Start Time =", StartTime)
    print("End Time =", endTime)
    print("Passed files:", passedFiles)
    print("Failed files:", failedFiles)
    print("Total line numbers:", totalLineNumbers)

def OrganiseTestingData(df): 

    storedTrainingSet = pd.read_csv('C:/Users/Jaime Kershaw Brown/Documents/Final year project/Stock-Market-Prediction-using-LSTM-NN/StockModel2/trainingData.txt', header = None)
    trainingDataShape = storedTrainingSet.shape

    sc = MinMaxScaler(feature_range = (0, 1))
    trainingData = np.loadtxt("C:/Users/Jaime Kershaw Brown/Documents/Final year project/Stock-Market-Prediction-using-LSTM-NN/StockModel2/trainingData.txt").reshape(trainingDataShape[0], trainingDataShape[1])
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
    FindKnownSectorStocks()