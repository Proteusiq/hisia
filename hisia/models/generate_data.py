import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from hisia.models.helpers import logger, config


def get_data(config):

    logger.info("[+] Model Training\n\n\tData Loading and spliting dataset")

    df = pd.read_json(config["data"]["data_trustpilot"])
    dt = pd.read_json(config["data"]["data_custom"])

    logger.info("[+] \n\n\tAdding SAM Data. Droping Zero Score")
    SAM = (
        "https://raw.githubusercontent.com/"
        "steffan267/Sentiment-Analysis-on-Danish-Social-Media/master/all_sentences.csv"
    )
    sam = pd.read_csv(SAM, names=["target", "features"])
    sam = (
        sam.loc[lambda d: d["target"].ne(0)].assign(
            target=lambda d: np.where(d["target"].gt(0), 1, 0)
        )
    )[["features", "target"]]

    dt = pd.concat((dt, sam), ignore_index=True)

    logger.info("[+] Dataset")
    X_train, X_test, y_train, y_test = train_test_split(
        df["features"],
        df["target"],
        test_size=config["train"]["test_size"],
        random_state=config["base"]["random_state"],
        stratify=df["target"],
    )

    # adding 8*20 custom fake reviews
    logger.info("Faking ..")
    X_train, y_train = (
        pd.concat((X_train, dt["features"]), ignore_index=True),
        pd.concat((y_train, dt["target"]), ignore_index=True),
    )

    data_stats = {
        "Traing Size": X_train.shape[0],
        "Test Size": X_test.shape[0],
        "Train Positive Samples": y_train[y_train == 1].shape[0],
        "Train Negative Samples": y_train[y_train == 0].shape[0],
        "Test Positive Samples": y_test[y_test == 1].shape[0],
        "Test Negative Sampples": y_test[y_test == 0].shape[0],
    }

    with open("hisia/metrics/train_data_report.json", "w") as f:
        json.dump(data_stats, f)
    
    train_data = pd.DataFrame()
    train_data["features"] = X_train
    train_data["target"] = y_train

    print(X_train.head(), y_train.head())

    test_data = pd.DataFrame()
    test_data["features"] = X_test
    test_data["target"] = y_test

    train_data.to_pickle(config["data"]["train_data"])
    test_data.to_pickle(config["data"]["test_data"])

if __name__ == "__main__":
    get_data(config=config)