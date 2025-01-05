import numpy as np
import joblib
import pandas as pd

import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import Myloanapp

from Myloanapp.config import config
from Myloanapp.processing.data_handling import load_pipeline,load_dataset

#Read the saved model in trained_models folder
classification_pipeline = load_pipeline(config.MODEL_NAME)


def generate_predictions(data_input):
    data = pd.DataFrame(data_input)
    pred = classification_pipeline.predict(data[config.FEATURES])
    result = {"Predictions" : pred}
    return result

'''

def generate_predictions():
    test_data = load_dataset(config.TEST_FILE)
    pred = classification_pipeline.predict(test_data[config.FEATURES])
    print(pred)
    result = {"Predictions" : pred}
    return result
'''

if __name__ == '__main__':
    generate_predictions()