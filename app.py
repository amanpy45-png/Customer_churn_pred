from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load model, encoders, and scaler
with open('best_model.pkl', 'rb') as f: loaded_model = pickle.load(f)
with open('encoder.pkl', 'rb') as f: encoders = pickle.load(f)
with open('scaler.pkl', 'rb') as f: scaler_data = pickle.load(f)

# Define the exact order used during model training
FEATURE_ORDER = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
    'MonthlyCharges', 'TotalCharges'
]

def make_prediction(input_data):
    # Create DataFrame with forced column order
    input_df = pd.DataFrame([input_data], columns=FEATURE_ORDER)

    # Encode categorical features
    for col, encoder in encoders.items():
        if col in input_df.columns:
            input_df[col] = encoder.transform(input_df[col])

    # Scale numerical features
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_df[numerical_cols] = scaler_data.transform(input_df[numerical_cols])

    # Predict
    prediction = loaded_model.predict(input_df)[0]
    probability = loaded_model.predict_proba(input_df)[0, 1]
    
    return "Churn" if prediction == 1 else "No Churn", probability

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction, probability = None, None
    if request.method == 'POST':
        try:
            # Map form data to dictionary
            input_data = {
                'gender': request.form['gender'],
                'SeniorCitizen': int(request.form['SeniorCitizen']),
                'Partner': request.form['Partner'],
                'Dependents': request.form['Dependents'],
                'tenure': int(request.form['tenure']),
                'PhoneService': request.form['PhoneService'],
                'MultipleLines': request.form['MultipleLines'],
                'InternetService': request.form['InternetService'],
                'OnlineSecurity': request.form['OnlineSecurity'],
                'OnlineBackup': request.form['OnlineBackup'],
                'DeviceProtection': request.form['DeviceProtection'],
                'TechSupport': request.form['TechSupport'],
                'StreamingTV': request.form['StreamingTV'],
                'StreamingMovies': request.form['StreamingMovies'],
                'Contract': request.form['Contract'],
                'PaperlessBilling': request.form['PaperlessBilling'],
                'PaymentMethod': request.form['PaymentMethod'],
                'MonthlyCharges': float(request.form['MonthlyCharges']),
                'TotalCharges': float(request.form['TotalCharges']),
            }
            prediction, probability = make_prediction(input_data)
        except Exception as e:
            return f"Error processing prediction: {str(e)}"

    return render_template('index.html', prediction=prediction, probability=probability)

if __name__ == '__main__':
    app.run(debug=True)