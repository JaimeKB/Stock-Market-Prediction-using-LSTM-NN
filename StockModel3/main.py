import math
import matplotlib.pyplot as plt
import tensorflow.keras
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

from CreateModel import TrainModel

df=pd.read_csv("C:/Users/Jaime Kershaw Brown/Documents/Final year project/TSLA.csv")

#split data
dataLength = len(df.iloc[:, 1:2])

trainingDataLength = math.floor(len(df.iloc[:, 1:2])*0.7)

training_set = df.iloc[:trainingDataLength, 1:2].values
# 881
test_set = df.iloc[trainingDataLength:, 1:2].values
# 387

print("Traing length: " + str(len(training_set)))
print("test length: " + str(len(test_set)))

# Scale/normalise data
sc = MinMaxScaler(feature_range = (0, 1))

# Training function
model = TrainModel(trainingDataLength, training_set, sc)

# Prepare testing data Function
#############################################################

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

#############################################################

# Predictions Function
#############################################################

predicted_stock_price = model.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#############################################################

# Visualise the results

plt.plot(df.loc[trainingDataLength:, 'Date'], dataset_test.values, color = 'red', label = 'Real TESLA Stock Price')
plt.plot(df.loc[trainingDataLength:, 'Date'], predicted_stock_price, color = 'blue', label = 'Predicted TESLA Stock Price')
plt.xticks(np.arange(0, len(x_test), 50))
plt.title('TESLA Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('TESLA Stock Price')
plt.legend()
plt.show()