import flask
from flask import request, jsonify
import joblib
import pandas as pd
import json
import numpy as np


app = flask.Flask(__name__)
app.config["DEBUG"] = False
model = joblib.load("/home/kenjilamy/.virtualenvs/venv/dash_package/model_GBM.pkl")
df = pd.read_csv("/home/kenjilamy/.virtualenvs/venv/dash_package//df.csv")

@app.route('/get_data')
def get_data():
    data = df.to_json(orient='records')
    return data

@app.route('/', methods=['GET'])
def index():
    return 'Home page'

def make_prediction(client_id):
    X = df[df['SK_ID_CURR'] == client_id]
    X = X.drop(columns=['TARGET', 'SK_ID_CURR', 'index'])
    return np.around(model.predict_proba(X),2)

@app.route('/predict', methods=["GET"])
def proba():
    if 'client_id' in request.args:
        client_id = int(request.args["client_id"])
        pred = make_prediction(client_id).tolist()[0]
        return pred
        
if __name__ == "__main__":
    app.run(port=8000)