
import pandas as pd
from hisia.models.helpers import (
    logger,
    config,
    persist_model,
)

from hisia.metrics.performance import classification_report



def model_evaluation(model, X, y):

    logger.info("[+] Model Evaluation")
    logger.info(
        f"Score: {model.score(X, y):.2%} on validation dataset with {len(y)} examples"
    )

    y_pred = model.predict(X)
    classification_report(y_test=y, y_pred=y_pred)


    

if __name__ == "__main__":
    

    test_data = pd.read_json(config["data"]["test_data"])
    X_test, y_test = (
        test_data["features"],
        test_data["target"],
    )

    hisia = persist_model(name=config["model"]["lr"], method="load")
    model_evaluation(hisia, X=X_test, y=y_test)
    