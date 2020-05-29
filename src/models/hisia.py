from abc import ABC, abstractproperty
from collections import namedtuple
import numpy as np
import pandas as pd
from helpers import persist_model


MODEL_PATH='src/models/base_model.pkl'
pre_load_model = persist_model(MODEL_PATH, method='load')
Sentiment = namedtuple('Sentiment', ['sentiment','positive_probability', 'negative_probability'])


class HisiaLoad(ABC):
    def __init__(self, model_path=None):
    
        if model_path is None:
            self.model = pre_load_model
        else:
            self.model = persist_model(model_path, method='load') 

    def __repr__(self):
        return f'{self.__class__.__name__}(Model=Logistic Regression)'

    @abstractproperty
    def sentiment(self):
        pass

class Hisia(HisiaLoad):
    def __init__(self, text, model_path=None):
        super().__init__(model_path)
        self.text = text
        self.sentiment

    def __repr__(self):
        return (f'Sentiment(sentiment={self.sentiment.sentiment}, '
                f'positive_probability={self.sentiment.positive_probability}, '
                f'negative_probability={self.sentiment.negative_probability})')

    @property
    def sentiment(self):
        
        if isinstance(self.text, str):
            X = [self.text]
        else:
            X = self.text

        response = self.model.predict_proba(X)
        response = pd.DataFrame(response)
        response.columns = ['negative_probability','positive_probability']
        response['sentiment'] = np.where(response['negative_probability'] > .5, 'negative', 'positive')

        self.results = Sentiment(**response.round(3).to_dict(orient='index')[0])
        return self.results

class HisiaRetrainable(HisiaLoad):
    '''Base Model that is retrainable
    '''
    def __init__(self, text, model_path=None):
        super().__init__(model_path)
        self.text = text
        self.sentiment

    @property
    def sentiment(self):
        pass

    def train(self, X, y):
        pass

    def reenforce(self,X, y, weight=None):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass

    def score(self, X,y):
        pass