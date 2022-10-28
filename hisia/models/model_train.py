from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.utils._testing import ignore_warnings


from hisia.models.helpers import (
    logger,
    config,
    tokenizer,
    persist_model,
    STOP_WORDS,
)



def model_training(config, X, y):

    hisia = Pipeline(
        steps=[
            (
                "count_verctorizer",
                CountVectorizer(
                    ngram_range=(1, 2),
                    max_features=config["train"]["vectorizer_max_features"],
                    tokenizer=tokenizer,
                    stop_words=STOP_WORDS,
                ),
            ),
            ("feature_selector", SelectKBest(chi2, k=config["train"]["select_k_best"])),
            ("tfidf", TfidfTransformer(sublinear_tf=True)),
            (
                "logistic_regression",
                LogisticRegressionCV(
                    cv=5,
                    solver=config["train"]["lr_solver"],
                    scoring=config["train"]["lr_scoring"],
                    max_iter=config["train"]["lr_max_iter"],
                    n_jobs=-1,
                    random_state=config["base"]["random_state"],
                    verbose=config["train"]["lr_verbose"],
                ),
            ),
        ]
    )

    logger.info("Cleaning, feature engineering and Training Logistic Regression in 5 CVs")
    logger.info(f"Model Steps:\n\t{hisia}")
    logger.info("\n[+] This will take ca. 3-4 minutes.")
    logger.info("-" * 75)

    with ignore_warnings():  # ConvergenceWarning
        hisia.fit(X, y)

    return hisia




if __name__ == "__main__":

    logger.info(f"\n[+] Model training")
    train_data = pd.read_json(config["data"]["train_data"])


    X_train, y_train = (
        train_data["features"],
        train_data["target"],
    )

    hisia = model_training(config, X=X_train, y=y_train)
    
    logger.info(f"\n[+] Model Saving")
    persist_model(f"{Path(__file__).parent}/base_model.pkl", clf=hisia, method="save")

    logger.info(f"\n[+] Model Saving Completed")

    
