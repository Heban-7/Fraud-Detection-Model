import pandas as pd

def preprocess_input(data):
    """
    Preprocess input data for Credit Card Fraud Detection.
    """
    # List of expected features in the correct order
    features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
    
    processed_data = {}
    for feature in features:
        try:
            # Convert each value to float
            processed_data[feature] = float(data.get(feature, 0))
        except ValueError:
            # In case conversion fails, use 0.0 as a fallback
            processed_data[feature] = 0.0

    # Create a DataFrame with a single row
    df = pd.DataFrame([processed_data], columns=features)
    return df
