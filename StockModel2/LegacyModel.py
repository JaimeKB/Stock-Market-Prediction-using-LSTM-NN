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

if __name__ == "__main__":

    df=pd.read_csv("C:/Users/Jaime Kershaw Brown/Documents/Final year project/TSLA.csv")
    #df=pd.read_csv("C:/Users/Jaime Kershaw Brown/Documents/Final year project/nasdaqStocks/AAPL.csv")
    print("Number of rows and columns:", df.shape)
    # 1259 by 7

    #split data
    dataLength = len(df.iloc[:, 1:2])

    #for index, row in df.iterrows():
    #    print(len(str(row[1:2])))
    #    newRow = str(row[1:2]).strip()
    #    print(len(newRow))
    #    print(newRow)

    trainingDataLength = math.floor(len(df.iloc[:, 1:2])*0.7)

    training_set = df.iloc[:trainingDataLength, 1:2].values
    # 881
    test_set = df.iloc[trainingDataLength:, 1:2].values
    # 387

    print("Traing length: " + str(len(training_set)))
    print("test length: " + str(len(test_set)))

    # Scale/normalise data
    sc = MinMaxScaler(feature_range = (0, 1))
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
    print("x_train shape:")
    print(x_train.shape)
    # (821, 60, 1)

    model = Sequential()

    # Layer 1
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
    model.add(Dropout(0.2))

    # Layer 2
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))

    # Layer 3
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))

    # Layer 4
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(units = 1))

    model.compile(optimizer = 'adam', loss= 'mean_squared_error')

    # Fit model to training dataset
    model.fit(x_train, y_train, epochs = 100, batch_size = 32)

    dataset_train = df.iloc[:trainingDataLength, 1:2]
    dataset_test = df.iloc[trainingDataLength:, 1:2]

    print("Dataset_train: " + str(dataset_train.shape))
    print("Dataset_test: " + str(dataset_test.shape))

    dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)

    print("Dataset_total: " + str(dataset_total.shape))

    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    print("inputs length: " + str(len(inputs)))

    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)
    x_test = []
    for i in range(60, dataLength - trainingDataLength + 60):
        x_test.append(inputs[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    print("x_test shape: " + str(x_test.shape))

    print(len(x_test))

    # Predictions

    predicted_stock_price = model.predict(x_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    #print(predicted_stock_price)
    #print(predicted_stock_price.shape)

    #print((df.loc[trainingDataLength:]))
    #print((df.loc[trainingDataLength:]).shape)


    # Visualise the results

    plt.plot(df.loc[trainingDataLength:, 'Date'], dataset_test.values, color = 'red', label = 'Real TESLA Stock Price')
    plt.plot(df.loc[trainingDataLength:, 'Date'], predicted_stock_price, color = 'blue', label = 'Predicted TESLA Stock Price')
    plt.xticks(np.arange(0, len(x_test), 50))
    plt.title('TESLA Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('TESLA Stock Price')
    plt.legend()
    plt.show()