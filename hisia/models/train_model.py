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


from hisia.reports.visualization import show_diagram
from hisia.metrics.performance import classification_report
from hisia.reports.visualization import show_most_informative_features


if __name__ == "__main__":

    print("Hello World!")
    print(config["data"])
    train_data = pd.read_pickle(config["data"]["train_data"])
    test_data = pd.read_pickle(config["data"]["test_data"])


    X_train, y_train = (
        train_data["features"],
        train_data["target"],
    )

    print(X_train.head())
    print(X_train.shape, y_train.shape)

    X_test, y_test = (
        test_data["features"],
        test_data["target"],
    )


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
        hisia.fit(X_train, y_train)

    logger.info("-" * 75)
    logger.info("[+] Model Evaluation in progress ...")
    logger.info(
        f"\nScore: {hisia.score(X_test, y_test):.2%} on validation dataset with {len(y_test)} examples"
    )

    y_pred = hisia.predict(X_test)
    classification_report(y_test=y_test, y_pred=y_pred)


    logger.info("[+] Generating ROC digrams")
    show_diagram(hisia, X_train, y_train, X_test, y_test, compare_test=True)
    feature_names = hisia.named_steps["count_verctorizer"].get_feature_names_out()
    best_features = [
        feature_names[i]
        for i in hisia.named_steps["feature_selector"].get_support(indices=True)
    ]
    predictor = hisia.named_steps["logistic_regression"]

    N = 100
    logger.info(f"Showing {N} models learned features for negative and postive decisions")
    logger.info("_" * 70)
    logger.info("\n")
    show_most_informative_features(best_features, predictor, n=N)
    logger.info(f"\n[+] Model Saving")


    persist_model(f"{Path(__file__).parent}/base_model.pkl", clf=hisia, method="save")
    logger.info("[+] Completed! Hurrah :)")
