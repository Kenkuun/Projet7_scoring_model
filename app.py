import flask
from flask import request, jsonify
import joblib
import pandas as pd
import json
import numpy as np
import git

app = flask.Flask(__name__)
app.config["DEBUG"] = False
print("start loading model")
model = joblib.load("model_GBM.pkl")
print("model loaded ok")
df = pd.read_csv("df.csv")
# /home/kenjilamy/Projet7_scoring_model/

@app.route('/', methods=['GET'])
def index():
    return 'Home page'

@app.route('/get_data')
def get_data():
    data = df.to_json(orient='records')
    return data

def make_prediction(client_id):
    X = df[df['SK_ID_CURR'] == client_id]
    print("data filter ok")
    X = X.drop(columns=['TARGET', 'SK_ID_CURR', 'index'])
    result = [0.5,0.95]
    # result = np.around(model.predict_proba(X),2)
    print("result =", result)
    return result

@app.route('/predict', methods=["GET"])
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

'push test 10'
          
if __name__ == "__main__":
    app.run(port=8000)
