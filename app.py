from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("random_forest_subsidy.pkl")

@app.route('/')
def home():
    return "AgriSubsidy AI API is running"

@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()

    df = pd.DataFrame([data])

    prediction = model.predict(df)

    return jsonify({
        "prediction": prediction[0]
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)