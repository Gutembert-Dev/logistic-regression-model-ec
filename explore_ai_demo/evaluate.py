"""
Evaluate
"""
# pylint: disable=W1514

import os
import json
import tarfile
import joblib

import pandas as pd
import numpy
from sklearn.metrics import confusion_matrix, balanced_accuracy_score


def metrics(y_true, y_score):
    """
    Metrics dict
    :param y_true:
    :param y_score:
    :return:
    """
    report_dict = {}

    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        y_true, y_score
    ).ravel()
    score = balanced_accuracy_score(y_true, y_score)

    # True positive rate (sensitivity or recall)
    # tpr = true_positive / (true_positive + false_negative)
    # False positive rate (fall-out)
    # fpr = false_positive / (false_positive + true_negative)
    # Precision
    precision = true_positive / (true_positive + false_positive)
    # Recall
    recall = true_positive / (true_positive + false_negative)
    # True negatvie tate (specificity)
    # tnr = 1 - fpr
    # F1 score
    f1_score = 2 * true_positive / (2 * true_positive + false_positive + false_negative)
    # MCC
    mcc = (
        true_positive * true_negative - false_positive * false_negative
    ) / numpy.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )

    report_dict["true_positive"] = str(true_positive)
    report_dict["false_positive"] = str(false_positive)
    report_dict["true_negative"] = str(true_negative)
    report_dict["false_negative"] = str(false_negative)
    report_dict["precision"] = str(precision)
    report_dict["recall"] = str(recall)
    report_dict["f1_score"] = str(f1_score)
    report_dict["mcc"] = str(mcc)
    report_dict["balanced_accuracy_score"] = str(score)
    report_dict["gini"] = str(0.5)
    return report_dict


if __name__ == "__main__":
    model_path = os.path.join("/opt/ml/processing/model", "model.tar.gz")
    print(f"Extracting model from path: {model_path}")
    with tarfile.open(model_path) as tar:
        tar.extractall(path="")
    print("Loading model")
    model = joblib.load("model.joblib")

    print("Loading test input data")
    test_features_data = os.path.join(
        "/opt/ml/processing/test", "validation_featuresMBPTP.csv"
    )
    test_labels_data = os.path.join(
        "/opt/ml/processing/test", "validation_labelsMBPTP.csv"
    )

    X_test = pd.read_csv(test_features_data, header=None)
    y_test = pd.read_csv(test_labels_data, header=None)
    #     predictions_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    #     pos_probs = predictions_proba[:, 1]

    #     # calculate the precision-recall auc
    #     precision, recall, thresholds = precision_recall_curve(y_test, pos_probs)
    #     # convert to f score
    #     fscore = (2 * precision * recall) / (precision + recall)
    #     # locate the index of the largest f score
    #     ix = numpy.argmax(fscore)
    #     print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
    #     threshold = thresholds[ix]

    #     y_pred = (pos_probs >= threshold).astype('int')

    metrics_dict = metrics(y_test, y_pred)

    print("Report:\n{report_dict}")

    evaluation_output_path = os.path.join(
        "/opt/ml/processing/evaluation", "evaluation.json"
    )
    print(f"Saving report to {evaluation_output_path}")

    with open(evaluation_output_path, "w") as f:
        f.write(json.dumps(metrics_dict))
