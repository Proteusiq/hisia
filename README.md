Hisia
==============================

:blush: :smiley: :relaxed: :yum: :wink: :smirk: :flushed: :worried: :frowning: :triumph: :disappointed: :angry: :persevere: :confounded: :shit: 


A Danish sentiment analyzer using scikit-learn LogisticRegression

![hisia cover](cover_image.png)

```python
from hisia import Hisia

negative_joe = Hisia('Det er simpelthen ikke okay :(')
negative_joe.sentiment
# Sentiment(sentiment='negative', positive_probability=0.01, negative_probability=0.99)
negative_joe.explain
# {'decision': -2.762615727464413, 
# 'intercept': 0.9959111721097562, 
# 'features': {:(', -4.334440323513933),
#               ('ikke', -3.425370797679455), 
#               ('simpelthen ikke', -1.6951652194361089),
#               ('simpelthen', -2.702688329340911)}
# }

positive_gro = Hisia('Det var ikke dårligt')
positive_gro
# Sentiment(sentiment=positive, positive_probability=0.707, negative_probability=0.293)
positive_gro.explain
# {'decision': 0.8791611359368057, 
# 'intercept': 0.9959111721097562, 
# 'features': {('ikke dårlig', 5.141224438943983), 
#              ('dårlig', -9.311879340739669), 
#              ('ikke', -3.425370797679455)}
# }
```
### Hisia (Emotions)
_Hisia_ is a Swahili word for emotion/feeling. My initial thought was to call it _Følelser_, a Danish word for feeling but it was just not right. As a Tanzanian, I went with Swahili as it was much more of a package name I would like to install. :) `pip install hisia`

### Data and Models Used in Hisia

**Data:** 2016 TrustPilot's 254,464 Danish reviews' body and stars and [8 fake reviews]*20 see notes for explanation.<br>

**Models**<br>
Hisia, _LogisticRegression_ with SAGA, a variant of Stochastic Average Gradient (SAG) as a solver, L2 penalty was select for the base model. Test score **accuracy is ca. 93%** and **recall of 93%**. SAGA is a faster solver for large datasets (rows and columns wise). As stochastic gradient, the memory of the previous gradient is incorporated/feed-forward current epoch. This allows a faster convergence rate. Seeds: 42 in data split of 80% training, 20% test, and 42 in the model used for reproducibility. Check notebooks for other parameters.

HisiaTrain, _SGDClassifier_, Stochastic Gradient Descent learner with smooth loss 'modified_huber as loss function and L2 penalty. Test score **accuracy 94%** and **recall of 94%**. SGDClassifier was select because of partial_fit. It allows batch/online training.

**Note:** This score reflects models in regards to TrustPilot reviews style of writing.<b>
 >8*10 fake reviews. TrustPilot reviews are directed towards products and services. A word like 'elsker'(love) or 'hader'(hate) were rare. To make sure the model learns such relationship, I added 8 reviews and duplicated them 20 times. These new sentences did not increase or decrease the model accuracy but correctly added the coefficient of the words love, hate and not bad (ikke dårligt). 

Notebook folder contains model_train notebook as a playground to reproduce the scores and also explore what the model has learned.

:warning: Running `from hisia import Hisia` will create a directory `hisia` and move with `model/base_model.pkl` and `data/stops.pkl` in it on the current directory. Until now, I cannot figure out how I can package this library with `poetry` and direct to correct location.

Features
--------
- Sentiment analysis
- Sentiment explainer
- Sentiment reinforcement learning (Coming Soon)
- Sentiment retrainer (Coming Soon)



Project Organization
------------

    ├── LICENSE
    ├── README.md         
    │
    ├── notebooks          <-  Jupyter notebook. Reproduce the results, and show model explanations
    │   └── model_training.ipynb
    │   └── helpers.py          
    │                         
    │
    ├── hisia              <-   Source code for use in this project.
    │   ├── __init__.py    <-   Makes hisia a Python module
    │   ├── hisia.py       <-   hisia a sentiment predictor and explainer
    │   │
    │   ├── data           <-  Path to training and validating dataset and stopwords: data folder is inside hisia for retrain
    │   │   └── data.json
    │   │   └── data_custom.json
    │   │   └── stops.pkl
    │   │
    │   ├── models         <-  Helpers, frozen model, models trainer
    │   │   │                 
    │   │   ├── base_model.pkl
    │   │   ├── helpers.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <-  Results oriented visualizations
    │       └── ROC.png
    │       └── ROC_test.png
    │
    ├── tests              <-   Path to tests to check models accurance, datatypes, scikit-learn version
    │   ├── __init__.py
    │   ├── conftest.py
    │   ├── test_basemodel_results.py
    │   ├── test_data.py
    │   ├── test_scikit_version.py
    │   ├── test_tokenizer.py  
    │
    │
    └── tox.ini            <- tox file to trains base models and run pytests


--------
# Retrain and Test: For Developers
Coming Soon
