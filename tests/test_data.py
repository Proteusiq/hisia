import pandas as pd


def test_training_data_shape(training_data):
    # Assert the data is having the same number of rows and columns

    assert training_data.shape == (254464, 3), "data is changed!"


def test_training_data_columns(training_data):

    assert training_data.columns.to_list() == [
        "features",
        "target",
        "stars",
    ], "data does not have correct columns"


def test_training_data_dtypes(training_data):
    assert (
        training_data["target"].dtype.name == "int64"
    ), "target inputs is not datatype int"
    assert (
        training_data["features"].dtype.name == "object"
    ), "features input is not datatype object"


def test_custom_training_data_shape(custom_training_data):
    # Assert the data is having the same number of rows and columns

    assert custom_training_data.shape == (8 * 20, 3), "data is changed!"
