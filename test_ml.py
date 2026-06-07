import pandas as pd
import pytest

from ml.data import process_data
from ml.model import compute_model_metrics, train_model
# TODO: add necessary import

# TODO: implement the first test. Change the function name and input as needed
@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "age": [39, 50, 38, 53],
            "workclass": ["State-gov", "Self-emp-not-inc", "Private", "Private"],
            "fnlgt": [77516, 83311, 215646, 234721],
            "education": ["Bachelors", "Bachelors", "HS-grad", "11th"],
            "education-num": [13, 13, 9, 7],
            "marital-status": ["Never-married", "Married-civ-spouse", "Divorced", "Married-civ-spouse"],
            "occupation": ["Adm-clerical", "Exec-managerial", "Handlers-cleaners", "Handlers-cleaners"],
            "relationship": ["Not-in-family", "Husband", "Not-in-family", "Husband"],
            "race": ["White", "White", "White", "Black"],
            "sex": ["Male", "Male", "Male", "Male"],
            "capital-gain": [2174, 0, 0, 0],
            "capital-loss": [0, 0, 0, 0],
            "hours-per-week": [40, 13, 40, 40],
            "native-country": ["United-States", "United-States", "United-States", "United-States"],
            "salary": ["<=50K", "<=50K", "<=50K", ">50K"],
        }
    )



# TODO: implement the second test. Change the function name and input as needed
def test_process_data(sample_data):
    """
    Test that process_data returns features, labels, encoder, and label binarizer.
    """
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, y, encoder, lb = process_data(
        sample_data,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    assert X.shape[0] == sample_data.shape[0]
    assert len(y) == sample_data.shape[0]
    assert encoder is not None
    assert lb is not None



# TODO: implement the third test. Change the function name and input as needed
def test_train_model(sample_data):
    """
    Test that train_model returns a fitted model.
    """
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, y, _, _ = process_data(
        sample_data,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    model = train_model(X, y)

    assert model is not None
    assert hasattr(model, "predict")

def test_compute_model_metrics():
    """
    Test that compute_model_metrics returns precision, recall, and fbeta between 0 and 1.
    """
    y = [1, 0, 1, 1]
    preds = [1, 0, 0, 1]

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1

