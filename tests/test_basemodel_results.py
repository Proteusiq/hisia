import pytest
from hisia import Hisia


def test_base_model_random_seed(seed=42):
    # Test the model seed is correct
    model = Hisia("Jeg elsker ikke pizza :(").model
    assert (
        model.named_steps["logistic_regression"].random_state == seed
    ), f"model seed {seed} is changed"


def test_base_model_score(test_data):
    # Test the model score on test data is greater than 93%
    model = Hisia("Jeg elsker pizza :)").model
    X_test, y_test = test_data

    assert model.score(X_test, y_test) > 0.93, "Model score is lower than 93%"


sentiment_examples = [
    ("Jeg elsker pizza :)", "positive"),
    ("Jeg elsker slet ikke pizza :(", "negative"),
]

explain_examples = [
    ("Jeg elsker pizza :)", {":)", "elsker"}),
    ("Jeg elsker slet ikke pizza :(", {":(", "elsker", "ikke", "slet", "slet ikke"}),
]


@pytest.mark.parametrize("text,prediction", sentiment_examples)
def test_base_sentiment(text, prediction):
    # Test the model score on test
    text = Hisia(text)
    assert (
        text.sentiment.sentiment == prediction
    ), f"model failed basic {prediction} score"


@pytest.mark.parametrize("text,explanation", explain_examples)
def test_base_explain(text, explanation):
    # Test the model score on test
    text = Hisia(text)
    assert {
        feature for feature, _ in text.explain["features"]
    } == explanation, f"model failed basic explanation: {explanation}"
