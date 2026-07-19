from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('random_forest_subsidy.pkl')

@app.route('/predict', methods=['POST'])
def predict():

    data = request.json

    features = [[
        data['subsidy_type'],
        data['farm_size'],
        data['crop_yield_before'],
        data['crop_yield_after'],
        data['income_before'],
        data['income_after'],
        data['pest'],
        data['calamity']
    ]]

    prediction = model.predict(features)

    return jsonify({
        'effectiveness': str(prediction[0])
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)