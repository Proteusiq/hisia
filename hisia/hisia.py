from abc import ABC, abstractproperty
from collections import namedtuple
import numpy as np
import pandas as pd
from loguru import logger
#from helpers import persist_model

@logger.catch
def persist_model(name,clf=None, method='load'):
    'Pass in the file name, object to be saved or loaded'
    import dill
    
    if method == 'load':
        with open(name,'rb') as f:
            return dill.load(f)
    elif method == 'save':
        logger.info(f'[+] Persisting {name} ...')
        if clf is None:
            raise ValueError('Pass Model/Pipeline/Transformation')
        with open(name,'wb') as f:
            dill.dump(clf,f)
            logger.info(f'[+] Persistence Complete. Model {name} is saved')
    else:
        raise ValeuError('Wrong arguments')


MODEL_PATH='hisia/models/base_model.pkl'
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
            self.X = [self.text]
        else:
            self.X = self.text

        response = self.model.predict_proba(self.X)
        response = pd.DataFrame(response)
        response.columns = ['negative_probability','positive_probability']
        response['sentiment'] = np.where(response['negative_probability'] > .5, 'negative', 'positive')

        self.results = Sentiment(**response.round(3).to_dict(orient='index')[0])
        return self.results

    @property
    def explain(self):
        feature_names = self.model.named_steps['count_verctorizer'].get_feature_names()
        best_features = [feature_names[i] for i in \
                        self.model.named_steps['feature_selector'].get_support(indices=True)]
        coefficients = self.model.named_steps['logistic_regression'].coef_[0]
        index_range = range(len(best_features))

        look_table = {index:(token,coef) for index, coef, token in zip(index_range, coefficients, best_features)}

        v = self.model.named_steps['count_verctorizer'].transform(self.X)
        v = self.model.named_steps['feature_selector'].transform(v)
        v = pd.DataFrame.sparse.from_spmatrix(v)
        v = set(v.loc[:,v.iloc[0]==1].columns)

        return {'decision': self.model.decision_function(self.X)[0],
                'intercept':self.model.named_steps['logistic_regression'].intercept_[0],
                'features': {look_table[item] for item in v}
        }
        





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