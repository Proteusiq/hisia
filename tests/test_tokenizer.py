from src.models.helpers import tokenizer

def test_training_data():
    # check that the tokenizer is working correctly
    assert tokenizer('Jeg er vred på, at jeg ikke fik min pakke :(') == ['vred', 'ikke', ':('], 'tokenizer did not load correctly'