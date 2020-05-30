hisia
==============================

Danish sentiment analysis using scikit-learn and Trustpilot dataset

Project Organization
------------

    ├── LICENSE
    ├── README.md         
    │
    ├── notebooks          <- Jupyter notebooks. Reproduce the results, and show model explainations
    │   └── model_training.ipynb         
    │                         
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Path to training and validating dataset and stopwords
    │   │   └── data.json
    │   │   └── stops.pkl
    │   │
    │   ├── models         <- Helpers, train models and then use trained models to make
    │   │   │                 predictions scripts
    │   │   ├── helpers.py
    │   │   ├── hisia.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <-  Results oriented visualizations
    │       └── ROC.png
    │
    ├── tests             <- Path to tests to check models accurance, datatypes, scikit-learn version
    │   ├── __init__.py
    │   ├── conftest.py
    │   ├── test_basemodel_results.py
    │   ├── test_data.py
    │   ├── test_scikit_version.py
    │   ├── test_tokenizer.py  
    │
    │
    └── tox.ini            <- tox file to trains models and run pytests


--------
