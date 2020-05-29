import sklearn

def test_scikit_learn_version():
       # Test the model is trained and pickled in scikit-learn "0.23.1"   
    assert sklearn.__version__ == '0.23.1', 'scikit-learn version is changed!'