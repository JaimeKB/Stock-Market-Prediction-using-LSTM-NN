# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# from statsmodels.tsa.stattools import acf, pacf

# df=pd.read_csv("C:/Users/Jaime Kershaw Brown/Documents/Final year project/TSLA.csv")

# # Partial Autocorrelation  Function
# plt.bar(x=np.arange(0,41), height=pacf(df.Close))
# plt.title("Partial Autocorrelation Function") 
# plt.legend()
# plt.show()

# #Plot ACF
# plt.bar(x=np.arange(0,41), height=acf(df.Close))
# plt.title("Autocorrelation Function")
# plt.legend()
# plt.show()


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

df=pd.read_csv("C:/Users/Jaime Kershaw Brown/Documents/Final year project/TSLA.csv")

# plt.figure()
# lag_plot(df['Open'], lag=3)
# plt.title('TESLA Stock - Autocorrelation plot with lag = 3')
# plt.show()

# plt.plot(df["Date"], df["Close"])
# plt.xticks(np.arange(0,1259, 200), df['Date'][0:1259:200])
# plt.title("TESLA stock price over time")
# plt.xlabel("time")
# plt.ylabel("price")
# plt.show()

train_data, test_data = df[0:int(len(df)*0.7)], df[int(len(df)*0.7):]
training_data = train_data['Close'].values
test_data = test_data['Close'].values
history = [x for x in training_data]
model_predictions = []
N_test_observations = len(test_data)
for time_point in range(N_test_observations):
    model = ARIMA(history, order=(4,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    model_predictions.append(yhat)
    true_test_value = test_data[time_point]
    history.append(true_test_value)

print(test_data.shape)
print(model_predictions.shape)

MSE_error = mean_squared_error(test_data, model_predictions)
print('Testing Mean Squared Error is {}'.format(MSE_error))

test_set_range = df[int(len(df)*0.7):].index
plt.plot(test_set_range, model_predictions, color='blue', marker='o', linestyle='dashed',label='Predicted Price')
plt.plot(test_set_range, test_data, color='red', label='Actual Price')
plt.title('TESLA Prices Prediction')
plt.xlabel('Date')
plt.ylabel('Prices')
plt.xticks(np.arange(881,1259,50), df.Date[881:1259:50])
plt.legend()
plt.show()