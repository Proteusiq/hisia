import json
from sklearn.metrics import precision_recall_fscore_support



def classification_report(y_test, y_pred):
    precision, recall, f1, *_ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted"
    )
    performance = {
        "precision": f"{precision: .3f}",
        "recall": f"{recall: .3f}",
        "f1": f"{f1: .3f}",
    }

    with open("hisia/metrics/classification_report.json", "w") as f:
        json.dump(performance, f)
    return performance
