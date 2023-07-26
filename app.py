import flask
from flask import request, jsonify
import joblib
import pandas as pd
import numpy as np
import git
import pickle

app = flask.Flask(__name__)
app.config["DEBUG"] = False
print("start loading model")
model = pickle.load(open("model_GBM", 'rb'))
print("model loaded ok")
df = pd.read_csv("dff.csv")
X = df[df['SK_ID_CURR'] == 100002]
X = X.drop(columns=['TARGET', 'SK_ID_CURR', 'index'])
pred = np.around(model.predict_proba(X),2).tolist()[0]
print('result', pred)

@app.route('/', methods=['GET'])
def index():
    return 'Home page'

@app.route('/get_data')
def get_data():
    data = df.to_json(orient='records')
    return data

@app.route('/predict', methods=['GET'])
def proba():
    print("Received a request to /predict") 
    if 'client_id' in request.args:
        client_id = int(request.args["client_id"])
        print("client ID is", client_id)
        X = df[df['SK_ID_CURR'] == client_id]
        X = X.drop(columns=['TARGET', 'SK_ID_CURR', 'index'])
        print("data filter ok")
        pred = np.around(model.predict_proba(X),2).tolist()[0]
        print('prediction done')
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
