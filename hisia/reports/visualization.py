import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    roc_curve,
)

from hisia.models.lazylogger import logger


@logger.catch
def show_diagram(trained_clf, X_train, y_train, X_test, y_test, compare_test=True):

    print("Classification Report")
    print("\t", "_" * 45)
    print(
        classification_report(
            y_test, trained_clf.predict(X_test), target_names=["Negative", "Positive"]
        )
    )

    print("\t", "_" * 45, "\n" * 2)

    plt.figure(figsize=(10, 5))

    data = [[X_test, y_test, "red", "Test"], [X_train, y_train, "blue", "Train"]]

    for row in data:
        X, y, color, split = row
        y_pred_prob = trained_clf.predict_proba(X)[:, 1]
        clf_score = trained_clf.score(X, y)

        fpr, tpr, _ = roc_curve(y, y_pred_prob)  # remember we need binary

        plt.plot(
            fpr,
            tpr,
            lw=4,
            color=color,
            label=f"{split} ROC curve (area ={clf_score:.2f})",
        )

        if not compare_test:
            break

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig("hisia/reports/ROC.png")


# Function modification of Mike Lee Williams(mike@mike.place)
def show_most_informative_features(feature_names, clf, n=1000):
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[: -(n + 1) : -1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % ((coef_1), fn_1, (coef_2), fn_2))
