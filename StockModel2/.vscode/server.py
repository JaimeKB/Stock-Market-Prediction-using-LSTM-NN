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

@app.route("/")
def IndexPage():
    """
    Render the index/home template
    """
    return render_template('Index.html')

@app.route("/TermsOfService")
def Redirect():
    """
    Render the terms of service and use template
    """
    return render_template('TermsOfService.html')

@app.route('/uploadajax', methods = ['POST'])
def uploadfile():
    """
    When a user uploads their own forecasting model, store it for use.
    Test the model using pre-set testing data.
    Create a forecast with application model as well to compare.
    Delete the uploaded model and return forecast data to user.
    """
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            uploaded_file.save(os.path.join('C:/Users/Jaime Kershaw Brown/Documents/Final year project/Stock-Market-Prediction-using-LSTM-NN/StockModel2/userModels', "userModel.h5"))
            predictedData, dateRange = TestUserModel()

            myModelPredictedData = RunOwn()

            testDict = {
                "predictedData": predictedData.tolist(),
                "myModelPredictedData": myModelPredictedData.tolist(),
                "dateRange": dateRange.tolist()
            }

            if os.path.exists('C:/Users/Jaime Kershaw Brown/Documents/Final year project/Stock-Market-Prediction-using-LSTM-NN/StockModel2/userModels/userModel.h5'):
                os.remove('C:/Users/Jaime Kershaw Brown/Documents/Final year project/Stock-Market-Prediction-using-LSTM-NN/StockModel2/userModels/userModel.h5')

            return jsonify(testDict)

    return("Success")


@app.route('/uploadCSV', methods = ['POST'])
def uploadCSVFile():
    """
    When a user uploads a CSV of historical stock data, save the file for use.
    Validate the data with integer/float checks and data length.
    Run data through application model to create a forecast.
    Delete CSV file, and return forecast to user.
    """
    if request.method == 'POST':
        uploaded_file = request.files['CSVfile']
        if uploaded_file.filename != '':
            uploaded_file.save(os.path.join('C:/Users/Jaime Kershaw Brown/Documents/Final year project/Stock-Market-Prediction-using-LSTM-NN/StockModel2/userCSVFiles', "userCSV.csv"))

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

                if os.path.exists("C:/Users/Jaime Kershaw Brown/Documents/Final year project/Stock-Market-Prediction-using-LSTM-NN/StockModel2/userCSVFiles/userCSV.csv"):
                    os.remove("C:/Users/Jaime Kershaw Brown/Documents/Final year project/Stock-Market-Prediction-using-LSTM-NN/StockModel2/userCSVFiles/userCSV.csv")

                return jsonify(testDict)
            else:
                return("Uploaded file was unusable!")
    return("Success")

if __name__ == "__main__":
    app.run()