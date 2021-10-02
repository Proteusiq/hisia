from abc import ABC, abstractproperty
from collections import namedtuple
from pathlib import Path
import typing as t
import dill
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.pipeline import Pipeline


SentimentType = t.NamedTuple(
    "Sentiment",
    [
        ("sentiment", str),
        ("positive_probability", float),
        ("negative_probability", float),
    ],
)
Sentiment = namedtuple(
    "Sentiment", ["sentiment", "positive_probability", "negative_probability"]
)


@logger.catch
def persist_model(name: str, clf: Pipeline = None, method: str = "load") -> None:
    """Persist Model
     Function use to save or load model

    Arguments:
        name {str} -- name of the saved model

    Keyword Arguments:
        clf {trained model} -- required only during save (default: {None})
        method {str} -- [takes in 'load' or 'save' argument to load or save models] (default: {'load'})

    Raises:
        ValueError: [raised when the arguments are not correct]
    """

    if method == "load":
        with open(name, "rb") as f:
            return dill.load(f)

    elif method == "save":
        logger.info(f"[+] Persisting {name} ...")
        if clf is None:
            raise ValueError("Pass Model/Pipeline/Transformation")

        with open(name, "wb") as f:
            dill.dump(clf, f)
            logger.info(f"[+] Persistence Complete. Model {name} is saved")
    else:
        raise ValeuError("Wrong arguments")


MODEL_PATH = Path(__file__).parent / "models/base_model.pkl"
PRE_LOAD_MODEL = persist_model(f"{MODEL_PATH}", method="load")


class HisiaLoad(ABC):
    def __init__(self, model_path: str = None):
        """Factory Class

        This is used to ensure a single model loading instance
        and a abstract property sentiment that is overiden in child classes

        Keyword Arguments:
            model_path {str} -- path to the trained model (default: {None})
        """

        if model_path is None:
            self.model = PRE_LOAD_MODEL
        else:
            self.model = persist_model(model_path, method="load")

    def __repr__(self):
        return f"{self.__class__.__name__}(Model=Logistic Regression)"

    @abstractproperty
    def sentiment(self):
        pass


class Hisia(HisiaLoad):
    """Hisia

    Keyword Arguments:
        text {str} -- text to analyze
        model_path {str} -- path to the trained model (default: {None})

    ...

    Attributes
    ----------
    text : str
        a text to analyze
    nmodel : Pipeline
        a loaded model as a scikit-learn pipeline with both features transformers and classifier


    Property
    -------
    sentiment
        returns the sentiment of text

    explain
        returns a dictionary of sentiment score explanation
        calculation decission = W1(word1) + W2(word2) + .. + intercept

    Usage:
    ```python
    from hisia import Hisia

    positive_gro = Hisia('det var super deligt')
    print(positive_gro.sentiment)
    print(positive_gro.explain)
    ```
    """

    def __init__(self, text: str, model_path: str = None):
        super().__init__(model_path)
        self.text = text
        self.sentiment

    def __repr__(self):
        return (
            f"Sentiment(sentiment={self.sentiment.sentiment}, "
            f"positive_probability={self.sentiment.positive_probability}, "
            f"negative_probability={self.sentiment.negative_probability})"
        )

    @property
    def sentiment(self) -> SentimentType:

        if isinstance(self.text, str):
            self.X = [self.text]
        else:
            self.X = self.text

        response = self.model.predict_proba(self.X)
        response = pd.DataFrame(response)
        response.columns = ["negative_probability", "positive_probability"]
        response["sentiment"] = np.where(
            response["negative_probability"] > 0.5, "negative", "positive"
        )

        self.results = Sentiment(**response.round(3).to_dict(orient="index")[0])
        return self.results

    @property
    def explain(self) -> t.Dict[str, float]:

        feature_names = self.model.named_steps[
            "count_verctorizer"
        ].get_feature_names_out()
        best_features = [
            feature_names[i]
            for i in self.model.named_steps["feature_selector"].get_support(
                indices=True
            )
        ]
        coefficients = self.model.named_steps["logistic_regression"].coef_[0]
        index_range = range(len(best_features))

        look_table = {
            index: (token, coef)
            for index, coef, token in zip(index_range, coefficients, best_features)
        }

        v = self.model.named_steps["count_verctorizer"].transform(self.X)
        v = self.model.named_steps["feature_selector"].transform(v)
        v = pd.DataFrame.sparse.from_spmatrix(v)
        v = set(v.loc[:, v.iloc[0] == 1].columns)

        return {
            "decision": self.model.decision_function(self.X)[0],
            "intercept": self.model.named_steps["logistic_regression"].intercept_[0],
            "features": {look_table[item] for item in v},
        }
