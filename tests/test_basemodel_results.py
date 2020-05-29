from src.models.hisia import Hisia


def test_base_model_score(test_data):
    # Test the model score on test data is greater than 93%
    model = Hisia('Jeg elsker pizza :)').model
    X_test, y_test = test_data
    
    assert model.score(X_test, y_test) > 0.93, 'Model score is lower than 93%'

def test_base_model_positive():
    # Test the model score on test
    text = Hisia('Jeg elsker pizza :)')
    assert text.sentiment.sentiment == 'positive', 'model failed basic positive score'

def test_base_model_negative():
    # Test the model score on test
    text = Hisia('Jeg elsker ikke pizza :(')
    assert text.sentiment.sentiment == 'negative', 'model failed basic negative score'



