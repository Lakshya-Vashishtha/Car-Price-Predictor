from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
cors = CORS(app)
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

def load_car_data():
    df = pd.read_csv('Cleaned_Car_predict.csv')    
    car_companies = sorted(df["company"].unique().tolist())  # Get unique car companies
    car_models = df.groupby("company")["name"].apply(list).to_dict()  # Group models by company
    years = sorted(df["year"].unique().tolist(), reverse=True)  # Get unique years
    fuel_types = sorted(df["fuel_type"].unique().tolist())  # Get unique fuel types
    return car_companies, car_models, years, fuel_types

@app.route('/')
def index():
    car_companies, car_models, years, fuel_types = load_car_data()
    return render_template("index.html", 
                           car_companies=car_companies, 
                           car_models=car_models, 
                           years=years, 
                           fuel_types=fuel_types)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        company = request.form.get('car_company')
        car_model = request.form.get('car_model')
        year = request.form.get('year')
        fuel_type = request.form.get('fuel_type')
        driven = request.form.get('km_driven')

        # Check for missing values
        if not company or not car_model or not year or not fuel_type or not driven:
            return jsonify({"error": "Missing input fields. Please fill in all values."}), 400  # Bad Request
        
        # Convert to appropriate data types
        year = int(year)
        driven = int(driven)

        # Ensure inputs exist in the dataset
        df = pd.read_csv('Cleaned_Car_predict.csv')

        if company not in df['company'].unique():
            return jsonify({"error": f"Unknown company '{company}'"}), 400
        if car_model not in df['name'].unique():
            return jsonify({"error": f"Unknown model '{car_model}'"}), 400
        if fuel_type not in df['fuel_type'].unique():
            return jsonify({"error": f"Unknown fuel type '{fuel_type}'"}), 400

        # Make prediction
        input_data = pd.DataFrame([[car_model, company, year, driven, fuel_type]], 
                                  columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

        prediction = model.predict(input_data)

        return jsonify({"price": str(np.round(prediction[0], 2))})

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Internal Server Error

if __name__ == '__main__':
    app.run(debug=True)