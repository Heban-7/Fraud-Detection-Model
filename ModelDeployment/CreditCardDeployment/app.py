from flask import Flask, request, render_template
import pickle
import pandas as pd
from scripts.utils import preprocess_input

app = Flask(__name__)

# Load the pre-trained model for credit card fraud detection
with open(r'C:\Users\liulj\Desktop\KAIM\KAIM-Week-8-9\Fraud-Detection-Model\ModelDeployment\CreditCardDeployment\model\creditcard_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # List of expected features
        features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

        # Extract form data into a dictionary
        data = {feature: request.form.get(feature) for feature in features}
        
        # Preprocess the input data
        input_df = preprocess_input(data)
        
        # Make a prediction using the pre-trained model
        prediction = model.predict(input_df)[0]
        
        # Interpret the result (assume 1 indicates fraud)
        result = "Fraudulent Transaction" if prediction == 1 else "Legitimate Transaction"
        
        return render_template('index.html', prediction_text=result)
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    # Listen on all interfaces for Docker compatibility
    app.run(debug=True, host='0.0.0.0')
