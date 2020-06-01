import pandas as pd
import pytest
from sklearn.model_selection import train_test_split


@pytest.fixture
def training_data():
    return pd.read_json('hisia/data/data.json')


@pytest.fixture
def custom_training_data():
    return pd.read_json('hisia/data/data_custom.json')

@pytest.fixture
def test_data(training_data):
    _, X_test, _, y_test = train_test_split(training_data['features'], 
                                            training_data['target'],
                                            test_size=.2,
                                            random_state=7,
                                            stratify=training_data['target'])

    return X_test, y_test