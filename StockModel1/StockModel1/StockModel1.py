

filePath = open("C:/Users/Jaime Kershaw Brown/Documents/Final year project/StockModel1/a.us.txt", "r")



import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


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




#data.plot(x='Date', y='Close')

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