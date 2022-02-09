import sklearn


def test_scikit_learn_version():
    # Test the model is trained and pickled in scikit-learn "1.0.0"
    assert sklearn.__version__ == "1.0.2", "scikit-learn version is changed!"
