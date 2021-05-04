# Stock-Market-Prediction-using-LSTM-NN
Final year project COMP3000, using Long Short Term Memory to predict future stock market trends

The goal for this project is to allow users to upload files of stock market data, process that data through a model, 
and produce both a graphical and numerical prediction for the stock values, time frame is to be determined.

Must install python before doing imports.

Required imports: matplotlib

My contact details: jaime.kershawbrown@students.plymouth.ac.uk

Supervisor: David Walker

### AWS deployment


- open putty with saved AWS session

- cd into below dir	

	/home/ubuntu/Stock-Market-Prediction-using-LSTM-NN/StockModel2/.vscode

- create virtual environment

	python3 -m venv venv

- activate virtual environment

	source venv/bin/activate

- go to correct branch

	git checkout <correct branch>

- install required libraries

	pip install -r requirements.txt

- if new installs needed then pip install and run line below

	pip freeze > requirements.txt 

- to start application

 	python3 server.py 

- go to link below to view deployment, subject to change

[click here](http://ec2-54-201-179-80.us-west-2.compute.amazonaws.com:5000)
