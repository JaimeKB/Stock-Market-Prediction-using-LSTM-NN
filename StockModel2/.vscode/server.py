import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask
from flask import render_template
from flask import jsonify
from flask import request
from pandas import DataFrame
from DataValidation import OrganiseTestingData
from DataValidation import ValidateYahooCSVData
from TestModel import PredictData
from TestModel import TestUserModel
from TestModel import RunOwn
import numpy as np
import pandas as pd
import os

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

            myModelPredictedData = RunOwn()

            testDict = {
                "predictedData": predictedData.tolist(),
                "myModelPredictedData": myModelPredictedData.tolist(),
                "dateRange": dateRange.tolist()
            }

            return jsonify(testDict)

    return("Success")


@app.route('/uploadCSV', methods = ['POST'])
def uploadCSVFile():
    if request.method == 'POST':
        uploaded_file = request.files['CSVfile']
        if uploaded_file.filename != '':
            uploaded_file.save(os.path.join('userCSVFiles', "userCSV.csv"))

            result, length = ValidateYahooCSVData("C:/Users/Jaime Kershaw Brown/Documents/Final year project/Stock-Market-Prediction-using-LSTM-NN/StockModel2/userCSVFiles", "userCSV.csv")

            if(result == "Pass"):

                df=pd.read_csv("C:/Users/Jaime Kershaw Brown/Documents/Final year project/Stock-Market-Prediction-using-LSTM-NN/StockModel2/userCSVFiles/userCSV.csv")
                testData = OrganiseTestingData(df)
                predictedData = PredictData(testData)

                dateRange = df.loc[60:, 'Date']

                testDict = {
                    "message": "Data was received!",
                    "predictedData": predictedData.tolist(),
                    "testData": testData.tolist(),
                    "dateRange": dateRange.tolist(),
                    "dataLength": len(testData)
                }

                return jsonify(testDict)

            else:
                return("Uploaded file was unusable!")

    return("Success")


# @app.route('/stockDataFile/<stockData>',methods=['GET'])
# def ProcessStockData(stockData):
#     print("Stock data recieved:")
#     stockData = stockData.split(",")
#     twoDStockData = to_matrix(stockData)
#     twoDStockData.pop(0)
#     df = DataFrame (twoDStockData,columns=['Date','Open','High','Low','Close','Adj Close','Volume'])

#     testData = OrganiseTestingData(df)
#     predictedData = PredictData(testData)

#     dateRange = df.loc[60:, 'Date']

#     testDict = {
#         "message": "Data was received!",
#         "predictedData": predictedData.tolist(),
#         "testData": testData.tolist(),
#         "dateRange": dateRange.tolist(),
#         "dataLength": len(testData)
#     }

#     return jsonify(testDict)



if __name__ == "__main__":
    app.run()