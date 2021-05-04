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

from ApplicationFunctions import TestFullFile
from ApplicationFunctions import CompareModels

import numpy as np
import pandas as pd
import os
import tempfile

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
            uploaded_file.save(os.path.join(myTempdir, "Model_Test.h5"))

            dataset=pd.read_csv(os.path.join(os.path.dirname(__file__), "../TeslaTestData.csv"))

            userModelYPoints, UMmse, UMrmse, UMmeanAverage, UMpercentageChange, ActualYPoints = CompareModels(myTempdir)

            ActualX, ActualY, testXPoints, testYPoints, futureXPoints, futureYPoints, mse, rmse, meanAverage, percentageChange = TestFullFile(dataset, "Technology", 100)

            dateRange = dataset[['Date']].tail(100)
            dateRange = dateRange.loc[:, 'Date']

            testDict = {
                "message": "Data was received!",
                "ActualX": dateRange.tolist(),
                "ActualYPoints": ActualYPoints.tolist(),
                "testYPoints": testYPoints.tolist(),
                "userModelYPoints": userModelYPoints.tolist(),
                "mse": mse,
                "rmse": rmse,
                "meanAverage": meanAverage,
                "percentageChange": percentageChange,
                "UMmse": UMmse,
                "UMrmse": UMrmse,
                "UMmeanAverage": UMmeanAverage,
                "UMpercentageChange": UMpercentageChange
            }

            if os.path.exists(os.path.join(myTempdir, "Model_Test.h5")):
                os.remove(os.path.join(myTempdir, "Model_Test.h5"))

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
        sector = request.form['sector']

        if uploaded_file.filename != '':

            uploaded_file.save(os.path.join(myTempdir, "userCSV.csv"))
           
            result, length = ValidateYahooCSVData(myTempdir, "userCSV.csv")

            if(result == "Pass"):

                numberOfDaysToPredict = 30

                df=pd.read_csv(os.path.join(myTempdir, "userCSV.csv"))
                
                ActualX, ActualY, testXPoints, testYPoints, futureXPoints, futureYPoints, mse, rmse, meanAverage, percentageChange = TestFullFile(df, sector, numberOfDaysToPredict)
            
                dateRange = df[['Date']].tail(30)
                dateRange = dateRange.loc[:, 'Date']

                testDict = {
                    "message": "Data was received!",
                    "ActualX": dateRange.tolist(),
                    "ActualY": ActualY.tolist(),
                    "testXPoints": testXPoints.tolist(),
                    "testYPoints": testYPoints.tolist(),
                    "futureXPoints": futureXPoints.tolist(),
                    "futureYPoints": futureYPoints.tolist(),
                    "mse": mse,
                    "rmse": rmse,
                    "meanAverage": meanAverage,
                    "percentageChange": percentageChange
                }

                if os.path.exists(os.path.join(myTempdir, "userCSV.csv")):
                    os.remove(os.path.join(myTempdir, "userCSV.csv"))

                return jsonify(testDict)
            else:
                return("Uploaded file was unusable!")
    return("Success")

if __name__ == "__main__":
    myTempdir = tempfile.gettempdir()
    app.run(debug=True)
# app.run(debug=True, port=5000, host='0.0.0.0')