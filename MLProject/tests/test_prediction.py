import pytest
import sys
import os
from pathlib import Path

# Adding the below path to avoid module not found error
#PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(os.path.dirname(__file__)))).parent

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(PACKAGE_ROOT)
sys.path.append(str(PACKAGE_ROOT))

from Loanapp.config import config

from Loanapp.predict import generate_predictions
from Loanapp.processing.data_handling import load_data


@pytest.fixture
def single_prediction():
    print("Loading Dataset..")
    test_dataset = load_data(config.TEST_FILE)

    print("First row for prediction..")
    single_row = test_dataset[:1]

    try:
        result = generate_predictions(single_row)
        print(f"Predicted result : {result}")
        return result
    except Exception as e:
        print(f"Error occured : {e}")
        raise

def test_single_pred_not_none(single_prediction):
    print("Testing if the predicted result is not None....")
    assert single_prediction is not None
    print("Test Passed: the predicted result is not None.")


def test_single_pred_str_type(single_prediction):
    """Test to validate the prediction type is a string."""
    print("Testing if the prediction type is a string...")
    assert isinstance(single_prediction.get('Predictions')[0], str)
    print("Test passed: Prediction type is a string.")


def test_single_pred_validate(single_prediction):
    """Test to validate the prediction value matches expected output."""
    print("Testing if the prediction value matches the expected output...")
    assert single_prediction.get('Predictions')[0] == 'Y'
    print("Test passed: Prediction value matches the expected output.")


def test_empty_dataset():
    """Test to ensure predictions handle empty datasets gracefully."""
    print("Testing with an empty dataset...")
    empty_dataset = []  # Create an empty dataset
    
    print("Generating predictions for the empty dataset...")
    result = generate_predictions(empty_dataset)  # Generate prediction
    
    print(f"Generated result: {result}")
    assert result.get('Predictions') == []  # Validate the result
    print("Test passed: Predictions handle empty datasets gracefully.")


@pytest.mark.parametrize("test_input", [
    (0),  # Test first row
    (10), # Test 10th row
    (100) # Test 100th row
])
def test_multiple_rows(test_input):
    """Parameterized test for multiple rows in the dataset."""
    print(f"Testing prediction for row at index {test_input}...")
    
    print("Loading the test dataset...")
    test_dataset = load_data(config.TEST_FILE)  # Load the dataset
    
    print(f"Dataset loaded with {len(test_dataset)} rows. Selecting row {test_input}...")
    row = test_dataset[test_input:test_input + 1]  # Select the row for testing
    
    print(f"Generating predictions for the selected row: {row}")
    result = generate_predictions(row)  # Generate prediction
    
    print(f"Generated result: {result}")
    assert result is not None  # Validate the result is not None
    print("Assertion passed: Result is not None.")
    
    assert isinstance(result.get('Predictions')[0], str)  # Validate the type
    print("Assertion passed: Prediction is of type string.")


def test_unexpected_data_format():
    """Test to ensure the model gracefully handles unexpected data formats."""
    invalid_inputs = [
        "invalid_string",  # String input
        12345,             # Integer input
        {"key": "value"},  # Dictionary input
    ]
    
    for invalid_input in invalid_inputs:
        print(f"Testing with invalid input: {invalid_input}")
        try:
            generate_predictions(invalid_input)
        except ValueError as e:
            print(f"Handled invalid input correctly: {e}")
        else:
            raise AssertionError(f"Model did not raise an error for invalid input: {invalid_input}")
