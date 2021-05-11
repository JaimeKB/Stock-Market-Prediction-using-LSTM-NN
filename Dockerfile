FROM python:3

COPY . .

RUN pip install -r StockModel2/.vscode/requirements.txt

CMD python StockModel2/.vscode/server.py
