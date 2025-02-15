from flask import Flask, request, render_template
import pickle
import pandas as pd
from scripts.utils import preprocess_input

app = Flask(__name__)

# Load artifacts

with open(r'C:\Users\liulj\Desktop\KAIM\KAIM-Week-8-9\Fraud-Detection-Model\ModelDeployment\model\fraud_detection_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open(r'C:\Users\liulj\Desktop\KAIM\KAIM-Week-8-9\Fraud-Detection-Model\ModelDeployment\model\expected_columns.pkl', 'rb') as f:
    expected_columns = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect data from the form
        data = {
            'user_id': [request.form['user_id']],
            'signup_time': [request.form['signup_time']],
            'purchase_time': [request.form['purchase_time']],
            'purchase_value': [float(request.form['purchase_value'])],
            'device_id': [request.form['device_id']],
            'source': [request.form['source']],
            'browser': [request.form['browser']],
            'sex': [request.form['sex']],
            'age': [int(request.form['age'])],
            'ip_address': [request.form['ip_address']],
            'country': [request.form['country']]
        }

        # Convert input data to DataFrame
        df = pd.DataFrame(data)

        # Preprocess the input data
        processed_df = preprocess_input(df, expected_columns)

        # Make prediction using the pre-trained model
        prediction = model.predict(processed_df)

        # Interpret the prediction result
        result = 'Fraudulent Transaction' if prediction[0] == 1 else 'Legitimate Transaction'

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        # Log error and display an error message on the front-end
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)