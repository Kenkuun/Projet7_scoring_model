import flask
from flask import request, jsonify
import joblib
import pandas as pd
import numpy as np
import git
import pickle

app = flask.Flask(__name__)
app.config["DEBUG"] = False
df = pd.read_csv("df.csv")

@app.route('/', methods=['GET'])
def index():
    return 'Home page'

@app.route('/get_data')
def get_data():
    data = df.to_json(orient='records')
    return data

def make_prediction(client_id):
    model = pickle.load(open("model_GBM", 'rb'))
    X = df[df['SK_ID_CURR'] == client_id]
    X = X.drop(columns=['TARGET', 'SK_ID_CURR', 'index'])
    result = np.around(model.predict_proba(X),2)
    return result

@app.route('/predict', methods=['GET'])
def proba():
    if 'client_id' in request.args:
        client_id = int(request.args["client_id"])
        pred = make_prediction(client_id).tolist()[0]
        return pred

@app.route('/update_server', methods=['POST', 'GET'])
def webhook():
    repo = git.Repo('./Projet7_scoring_model')
    origin = repo.remotes.origin
    origin.pull()
    return 'Updated pythonanywhere successfully', 200
          
if __name__ == "__main__":
    app.run(port=8000)
