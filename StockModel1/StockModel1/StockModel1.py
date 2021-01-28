
filePath = open("C:/Users/Jaime Kershaw Brown/Documents/Final year project/Stock-Market-Prediction-using-LSTM-NN/StockModel1/a.us.txt", "r")



import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


def create_dataset(dataset, time_step):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)



# DATA COLLECTION

data = pd.read_csv(filePath,
                 header=1,
                 usecols=[0, 1, 2, 3, 4, 5],
                 names=["Date", "Open", "High", "Low", "Close", "Volume", "OpenInt"])

#print(date)
data['Date'] = pd.to_datetime(data["Date"])
#data.plot()
#plt.show()

df1 = data.reset_index()['Close']
#df1.plot()
#plt.show()

scaler = MinMaxScaler(feature_range=(0,1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))
#print(df1)

# Preprocessing - Train and Testing

# Split dataset
training_size = int(len(df1)*0.60)
test_size = len(df1) - training_size
train_data = df1[0:training_size, :]
test_data = df1[training_size:len(df1),:1]


# Reshape data
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

print(X_train)
print(y_train)
print(X_test)
print(ytest)

# Reshape input to be samples, time steps, features
X_train = X_train.reshape(X_train[0], X_train.shape[1], 1)
x_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Create LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape=(100,1)))
model.add(LSTM(50, return_sequences = True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

print(model.summary())


#class datapoint:
#  def __init__(self, date, open, high, low, close, volume, openint):
#    self.date = date
#    self.open = open
#    self.high = high
#    self.low = low
#    self.close = close
#    self.volume = volume
#    self.openint = openint

#data = []


#for x in f:
#    dataline = x.split(',')
#    dataobject = datapoint(dataline[0], dataline[1], dataline[2], dataline[3], dataline[4], dataline[5], dataline[6])
#    data.append(dataobject)