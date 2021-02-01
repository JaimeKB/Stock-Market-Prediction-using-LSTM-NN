from flask import Flask
from flask import render_template
from pandas import DataFrame
from DataValidation import OrganiseTestingData
from TestModel import PredictData
import matplotlib.pyplot as plt
import numpy as np


app = Flask(__name__)

def to_matrix(l):
    return [l[i:i+7] for i in range(0, len(l), 7)]


@app.route("/")
def hello():
    return render_template('Index.html')


@app.route('/stockDataFile/<stockData>',methods=['GET'])
def ProcessStockData(stockData):
    print("Stock data recieved:")
    stockData = stockData.split(",")
    twoDStockData = to_matrix(stockData)
    twoDStockData.pop(0)
    df = DataFrame (twoDStockData,columns=['Date','Open','High','Low','Close','Adj Close','Volume'])

    # print(df)

    testData = OrganiseTestingData(df)
    predictedData = PredictData(testData)

    plt.plot(df.loc[60:, 'Date'], predictedData, color = 'blue', label = 'Predicted TESLA Stock Price')
    plt.xticks(np.arange(0, len(testData), 50))
    plt.title('TESLA Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('TESLA Stock Price')
    plt.legend()
    plt.show()

    print("GAP")
    print("GAP")
    print("GAP")
    print("GAP")
    print("GAP")

    return ("File was recieved")



if __name__ == "__main__":
    app.run()

#C:\\Program Files (x86)\\Microsoft Visual Studio\\Shared\\Python37_64\\python.exe