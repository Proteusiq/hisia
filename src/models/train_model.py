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


from helpers import tokenizer
from helpers import persist_model
from helpers import show_diagram
from helpers import show_most_informative_features

print('[+] Model Training\n\n\tTesting tokenizers and dataset')
# Assert the toenizer and stops are working
assert tokenizer('Jeg er vred på, at jeg ikke fik min pakke :(') == ['vred', 'ikke', ':('], 'tokenizer did not load correctly'
# Assert the data is having the same number of rows and columns
df = pd.read_json('src/data/data.json')
assert df.shape == (254464, 3), 'data is changed!'

X_train, X_test, y_train, y_test = train_test_split(df['features'], 
                                                     df['target'],
                                                     test_size=.2,
                                                     random_state=7,
                                                     stratify=df['target'])
print('[+] Dataset')
print(f'Traing Size: {X_train.shape[0]}\nTest Size: {X_test.shape[0]:>8}')
print(f'\nTraing Size\n\tPositive||Negative Samples\n\t  {y_train[y_train==1].shape[0]}||{y_train[y_train==0].shape[0]}')
print(f'\nTest Size\n\tPositive||Negative Samples\n\t  {y_test[y_test==1].shape[0]}||{y_test[y_test==0].shape[0]}')


hisia = Pipeline(steps =[
        ('count_verctorizer',  CountVectorizer(ngram_range=(1, 2), 
                                 max_features=100000,
                                 tokenizer=tokenizer, 
                                 #stop_words=STOP_WORDS
                                )
        ),
        ('feature_selector', SelectKBest(chi2, k=5000)),
        ('tfidf', TfidfTransformer(sublinear_tf=True)),
        ('logistic_regression', LogisticRegressionCV(cv=5,
                                                    solver='saga',
                                                    scoring='accuracy',
                                                    n_jobs=-1, 
                                                    verbose=0))
])

print('Cleaning, feature engineering and Training Logistic Regression in 5 CVs')
print(f'Model Steps:\n\t{hisia}')
print('\n[+] This will take ca. 3-4 minutes. Ignore Convergence Warning') 
print('-'*75)                                                 
hisia.fit(X_train, y_train)

print('-'*75)
print('[+] Model Evaluation in progress ...')
print(f'\nScore: {hisia.score(X_test, y_test):.2%} on vaidation dataset with {len(y_test)} examples')
print('[+] Generating ROC digrams')
show_diagram(hisia, X_train, y_train, X_test, y_test, compare_test=True)
feature_names = hisia.named_steps['count_verctorizer'].get_feature_names()
best_features = [feature_names[i] for i in hisia.named_steps['feature_selector'].get_support(indices=True)]
predictor =  hisia.named_steps['logistic_regression']

N = 100
print(f'Showing {N} models learned features for negative and postive decisions')
print('_'*70)
print('\n')
show_most_informative_features(best_features, predictor, n=N)
print(f'\n[+] Model Saving')
persist_model('src/models/base_model.pkl', clf=hisia, method='save')

print('[+] Completed! Hurrah :)')
