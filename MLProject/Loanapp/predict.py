import numpy as np
import joblib
import pandas as pd

import os
import sys


PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
print(PACKAGE_ROOT)
sys.path.append(str(PACKAGE_ROOT))

from Loanapp.config import config
from Loanapp.processing.data_handling import load_data,load_pipeline

_model = load_pipeline(config.MODEL_NAME)

'''
def generate_predictions():
    test_data = load_data(config.TEST_FILE)
    pred = _model.predict(test_data[config.FEATURES])
    result = {"Predictions" : pred}
    print(result)
    return result
'''

def generate_predictions(data_input):
    """Generate predictions for the given input data."""
    try:
        # Attempt to convert the input to a DataFrame
        data = pd.DataFrame(data_input)
    except ValueError as e:
        # Handle cases where the input cannot be converted into a DataFrame
        print(f"Invalid input format: {e}")
        raise ValueError(f"Unexpected data format: {data_input}. Ensure input is tabular data.")
    
    # Handle empty dataset
    if data.empty:
        print("The input dataset is empty. Returning empty predictions.")
        return {"Predictions": []}

    # Check if required features are present in the dataset
    missing_features = [feature for feature in config.FEATURES if feature not in data.columns]
    if missing_features:
        raise ValueError(f"Missing required features in the input data: {missing_features}")
    
    # Generate predictions
    pred = _model.predict(data[config.FEATURES])
    output = lambda x: 'Y' if x == 1 else 'N'
    status = [output(p) for p in pred]  # Handle multiple predictions
    result = {"Predictions": status}
    
    return result



if __name__ == '__main__':
    generate_predictions()