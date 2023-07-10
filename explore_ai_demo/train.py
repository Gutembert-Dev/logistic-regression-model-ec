"""
Train
"""
# pylint: disable=C0103
import os
import logging
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

logger = logging.getLogger()

if __name__ == "__main__":
    training_data_dir = "/opt/ml/processing/train"
    train_features_data = os.path.join(training_data_dir, "train_featuresMBPTP.csv")
    train_labels_data = os.path.join(training_data_dir, "train_labelsMBPTP.csv")
    train_weight_data = os.path.join(training_data_dir, "train_weightMBPTP.csv")
    print("Reading input data")

    X_train = pd.read_csv(train_features_data, header=None)
    print(f"X_train = {X_train.head()}")

    y_train = pd.read_csv(train_labels_data, header=None)
    print(f"y_train = {y_train.head()}")

    weight_train = pd.read_csv(train_weight_data, header=None)
    print(f"weight_train = {weight_train.head()}")

    model = LogisticRegression(random_state=0, penalty="l1", solver="liblinear")
    print("Training LR model")
    model.fit(X_train, y_train, sample_weight=weight_train.values.ravel())

    model_output_directory = os.path.join("/opt/ml/model", "model.joblib")
    print(f"Saving model to {model_output_directory}")
    # logging.INFO(f"*****model path is {model_output_directory}")

    joblib.dump(model, model_output_directory)
    # path = Path(model_output_directory)

    # print(f"File exists at {model_output_directory} =  {path.isfile()}")
