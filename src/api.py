from flask import Flask, Response, request
import pandas as pd
import os
from io import StringIO
from flask_cors import CORS
from sklearn.linear_model import LinearRegression
# import pickle

app = Flask(__name__)
CORS(app)

training_data = pd.read_csv(os.path.join('data', 'auto-mpg.csv'))

trained_model = pd.read_pickle(os.path.join(
    'data', 'models', 'auto-mpg.sav'))

@app.route('/', methods=['GET'])
def main():
    return {"hello": "world"}

@app.route('/hello_world', methods=['GET'])
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/training_data', methods=['GET'])
def get_training_data():
    return Response(training_data.to_json(), mimetype='application/json')

@app.route('/predict', methods=['GET'])
def predict():
    zylinder = request.args.get('zylinder')
    ps = request.args.get('ps')
    gewicht = request.args.get('gewicht')
    beschleunigung = request.args.get('beschleunigung')
    baujahr = request.args.get('baujahr')

    csv_string = ",".join([zylinder, ps, gewicht, beschleunigung, baujahr])

    csv_data = StringIO(csv_string)

    attribute_names = ['zylinder', 'ps', 'gewicht', 'beschleunigung', 'baujahr']

    prediction_data = pd.read_csv(csv_data, names=attribute_names)

    prediction = trained_model.predict(prediction_data)
    prediction = float(prediction)

    return {"result": prediction}
