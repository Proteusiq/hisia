import joblib
import re
import dill
import lemmy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import  LogisticRegressionCV
from loguru import logger


from hisia.models.helpers import tokenizer
from hisia.models.helpers import persist_model
from hisia.models.helpers import show_diagram
from hisia.models.helpers import show_most_informative_features

logger.info('[+] Model Training\n\n\tData Loading and spliting dataset')

df = pd.read_json('hisia/data/data.json')
dt = pd.read_json('hisia/data/data_custom.json')


logger.info('[+] Dataset')
X_train, X_test, y_train, y_test = train_test_split(df['features'], 
                                                     df['target'],
                                                     test_size=.2,
                                                     random_state=42,
                                                     stratify=df['target'])

# adding 8*20 custom fake reviews
X_train, y_train = (X_train.append(dt['features'],ignore_index=True),
                    y_train.append(dt['target'],ignore_index=True))

logger.info(f'Traing Size: {X_train.shape[0]}\nTest Size: {X_test.shape[0]:>8}')
logger.info(f'\nTraing Size\n\tPositive||Negative Samples\n\t  {y_train[y_train==1].shape[0]}||{y_train[y_train==0].shape[0]}')
logger.info(f'\nTest Size\n\tPositive||Negative Samples\n\t  {y_test[y_test==1].shape[0]}||{y_test[y_test==0].shape[0]}')


hisia = Pipeline(steps =[
        ('count_verctorizer',  CountVectorizer(ngram_range=(1, 2), 
                                 max_features=150000,
                                 tokenizer=tokenizer, 
                                )
        ),
        ('feature_selector', SelectKBest(chi2, k=10000)),
        ('tfidf', TfidfTransformer(sublinear_tf=True)),
        ('logistic_regression', LogisticRegressionCV(cv=5,
                                                    solver='saga',
                                                    scoring='accuracy',
                                                    max_iter=200,
                                                    n_jobs=-1,
                                                    random_state=42, 
                                                    verbose=0))
])

logger.info('Cleaning, feature engineering and Training Logistic Regression in 5 CVs')
logger.info(f'Model Steps:\n\t{hisia}')
logger.info('\n[+] This will take ca. 3-4 minutes. Ignore Convergence Warning') 
logger.info('-'*75)                                                 
hisia.fit(X_train, y_train)

logger.info('-'*75)
logger.info('[+] Model Evaluation in progress ...')
logger.info(f'\nScore: {hisia.score(X_test, y_test):.2%} on vaidation dataset with {len(y_test)} examples')
logger.info('[+] Generating ROC digrams')
show_diagram(hisia, X_train, y_train, X_test, y_test, compare_test=True)
feature_names = hisia.named_steps['count_verctorizer'].get_feature_names()
best_features = [feature_names[i] for i in hisia.named_steps['feature_selector'].get_support(indices=True)]
predictor =  hisia.named_steps['logistic_regression']

N = 100
logger.info(f'Showing {N} models learned features for negative and postive decisions')
logger.info('_'*70)
logger.info('\n')
show_most_informative_features(best_features, predictor, n=N)
logger.info(f'\n[+] Model Saving')
persist_model('hisia/models/base_model.pkl', clf=hisia, method='save')

logger.info('[+] Completed! Hurrah :)')
