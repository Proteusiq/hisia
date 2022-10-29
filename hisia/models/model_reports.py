import pandas as pd
from hisia.models.helpers import (
    logger,
    config,
    persist_model,
)

from hisia.reports.visualization import show_diagram



def model_diagostics(model, X_train, y_train, X_test, y_test):

    logger.info("[+] Generating ROC digrams")
    show_diagram(model, X_train, y_train, X_test, y_test, compare_test=True)

    

if __name__ == "__main__":
    
    train_data = pd.read_json(config["data"]["train_data"])
    X_train, y_train = (
        train_data["features"],
        train_data["target"],
    )

    test_data = pd.read_json(config["data"]["test_data"])
    X_test, y_test = (
        test_data["features"],
        test_data["target"],
    )

    hisia = persist_model(name=config["model"]["lr"], method="load")
    model_diagostics(hisia, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)