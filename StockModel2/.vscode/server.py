import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask
from flask import render_template
from flask import jsonify
from flask import request
from pandas import DataFrame
from DataValidation import OrganiseTestingData
from TestModel import PredictData
from TestModel import TestUserModel
import numpy as np
import pandas as pd
import os
# import json

app = Flask(__name__)

def to_matrix(l):
    return [l[i:i+7] for i in range(0, len(l), 7)]

@app.route("/")
def hello():
    return render_template('Index.html')

@app.route('/uploadajax', methods = ['POST'])
def upldfile():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            uploaded_file.save(os.path.join('userModels', "userModel.h5"))
            predictedData, dateRange = TestUserModel()

            testDict = {
                "predictedData": predictedData.tolist(),
                "dateRange": dateRange.tolist(),
            }

            return jsonify(testDict)

    return("Success")

@app.route('/stockDataFile/<stockData>',methods=['GET'])
def ProcessStockData(stockData):
    print("Stock data recieved:")
    stockData = stockData.split(",")
    twoDStockData = to_matrix(stockData)
    twoDStockData.pop(0)
    df = DataFrame (twoDStockData,columns=['Date','Open','High','Low','Close','Adj Close','Volume'])

    testData = OrganiseTestingData(df)
    predictedData = PredictData(testData)

    # create_figure(testData, predictedData)

    # plt.plot(df.loc[60:, 'Date'], predictedData, color = 'blue', label = 'Predicted TESLA Stock Price')
    # plt.xticks(np.arange(0, len(testData), 50))
    # plt.title('TESLA Stock Price Prediction')
    # plt.xlabel('Time')
    # plt.ylabel('TESLA Stock Price')
    # plt.legend()
    # plt.show()
    # plt.clf()
# , predictedData, testData, df.loc[60:, 'Date'], len(testData)

    dateRange = df.loc[60:, 'Date']

    testDict = {
        "message": "Data was received!",
        "predictedData": predictedData.tolist(),
        "testData": testData.tolist(),
        "dateRange": dateRange.tolist(),
        "dataLength": len(testData)
    }

    return jsonify(testDict)



if __name__ == "__main__":
    app.run()

#C:\\Program Files (x86)\\Microsoft Visual Studio\\Shared\\Python37_64\\python.exe