import joblib
import pandas as pd
import numpy as np
import requests
from app import make_prediction

api = "http://127.0.0.1:8000"

def test_client1():
    assert make_prediction(100002)[0][0] ==  0.05
    assert make_prediction(100002)[0][1] ==  0.95

def test_client2():
    assert make_prediction(100003)[0][0] ==  0.99
    assert make_prediction(100003)[0][1] ==  0.01

def test_client3():
    assert make_prediction(172551)[0][0] ==  0.96
    assert make_prediction(172551)[0][1] ==  0.04

# def test_api():
#     response = requests.get(api)
#     assert response.status_code == 200
