import pytest
from hisia import Hisia



def test_base_model_random_seed(seed=42):
    # Test the model seed is correct
    model = Hisia('Jeg elsker ikke pizza :(').model
    assert model.named_steps['logistic_regression'].random_state == seed, f'model seed {seed} is changed'

def test_base_model_score(test_data):
    # Test the model score on test data is greater than 93%
    model = Hisia('Jeg elsker pizza :)').model
    X_test, y_test = test_data
    
    assert model.score(X_test, y_test) > 0.93, 'Model score is lower than 93%'

examples = [("Jeg elsker pizza :(", "positive"), ("Jeg elsker ikke pizza :(", "negative")]
@pytest.mark.parametrize("text,prediction", examples)
def test_base_model(text, prediction):
    # Test the model score on test
    text = Hisia(text)
    assert text.sentiment.sentiment == prediction, f'model failed basic {prediction} score'
