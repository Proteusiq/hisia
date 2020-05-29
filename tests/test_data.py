import pandas as pd

def test_training_data():
       # Assert the data is having the same number of rows and columns
    df = pd.read_json('src/data/data.json')
    assert df.shape == (254464, 3), 'data is changed!'
    assert df.columns == ['features','target','stars'], 'data does not have correct columns'
    assert df['target'].dtype.name == 'int', 'target features is not datatype int'