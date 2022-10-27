import re
from pathlib import Path

import dill
import joblib
import yaml
import lemmy
import matplotlib.pyplot as plt


from hisia.models.lazylogger import logger


BASE_FORDER = Path(__file__).parent.parent.parent
PARAMS_FILE = BASE_FORDER / "params.yaml"


config = yaml.safe_load(Path(PARAMS_FILE).open())

STOP_WORDS = joblib.load(config["data"]["stop_words"])
lemmatizer = lemmy.load("da")

# Add more stopwords
STOP_WORDS.update({"kilometer", "alme", "bank", "brand", "dansk", "presi"})


@logger.catch
def tokenizer(blob, stop_words=STOP_WORDS, remove_digits=True):

    if stop_words is None:
        stop_words = {}

    blob = blob.lower()

    # eyes [nose] mouth | mouth [nose] eyes pattern
    emoticons = r"(?:[<>]?[:;=8][\-o\*\']?[\)\]\(\[dDpP/\:\}\{@\|\\]|[\)\]\(\[dDpP/\:\}\{@\|\\][\-o\*\']?[:;=8][<>]?)"
    emoticon_re = re.compile(emoticons, re.VERBOSE | re.I | re.UNICODE)

    text = re.sub(r"[\W]+", " ", blob)

    # remove 3+ repetitive characters i.e. hellllo -> hello, jaaaa -> jaa
    repetitions = re.compile(r"(.)\1{2,}")
    text = repetitions.sub(r"\1\1", text)

    # remove 2+ repetitive words e.g. hej hej hej -> hej

    repetitions = re.compile(r"\b(\w+)\s+(\1\s*)+\b")
    text = repetitions.sub(r"\1 ", text)

    # 14år --> 14 år
    text = re.sub(r"([0-9]+(\.[0-9]+)?)", r" \1 ", text).strip()

    emoji = "".join(re.findall(emoticon_re, blob))

    # remove stopwords
    text_nostop = [word for word in text.split() if word not in stop_words]

    # tokenization lemmatize
    lemmatized_text = [lemmatizer.lemmatize("", word)[-1] for word in text_nostop]

    remove_stopwords = " ".join(word for word in lemmatized_text if len(word) > 1)

    if remove_digits:
        remove_stopwords = re.sub(r"\b\d+\b", "", remove_stopwords)

    # remove extra spaces
    remove_stopwords = " ".join(remove_stopwords.split())
    result = f"{remove_stopwords} {emoji}".encode("utf-8").decode("utf-8")

    return result.split()


@logger.catch
def persist_model(name, clf=None, method="load"):
    "Pass in the file name, object to be saved or loaded"

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
