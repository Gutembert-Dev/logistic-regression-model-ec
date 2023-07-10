import logging
import os

import pandas as pd

woevars = [
    "w_mob",
    "w_ACC011CRT",
    "w_ACC100CRT",
    "w_ACC101REV",
    "w_ACC104CRT",
    "w_ACC233CRT",
    "w_ACC234CRT",
    "w_ACC309NCT",
    "w_ACC314CRT",
    "w_Age",
    "w_ENQ004OTH",
    "w_LEG200OTH",
    "w_NumPTPsL24",
    "w_NumRPCL9",
    "w_NumRecsL9",
    "w_ACC230CRT",
    "w_ACC001UCR",
    "w_ACC209CRT",
]
# from constants import COLUMNS

# from preprocessing_functions import (
#     replace_missing,
#     exclusion,
#     mob,
#     ptp24,
#     rec9,
#     rpc9,
#     age,
#     acc001ucr,
#     acc011crt,
#     acc100crt,
#     acc101rev,
#     acc104crt,
#     acc209crt,
#     acc230crt,
#     acc233crt,
#     acc234crt,
#     acc309nct,
#     acc314crt,
#     enq004oth,
#     leg200oth,
#     get_input_dataMBPTP,
#     train_test_split2,
# )

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

IDENTIFIER_COLUMNS = ["contractKey", "score_month"]

LABEL = ["target"]

FEATURES_COLUMNS = [
    "months_on_book",
    "ACC011CRT",
    "ACC100CRT",
    "ACC101REV",
    "ACC104CRT",
    "ACC233CRT",
    "ACC234CRT",
    "ACC309NCT",
    "ACC314CRT",
    "Age",
    "ENQ004OTH",
    "LEG200OTH",
    "NumPTPsL24M",
    "NumRPCsL9M",
    "NumRecsL9M",
    "ACC230CRT",
    "ACC001UCR",
    "ACC209CRT",
    "CON001OTH",
    "CON002OTH",
    "Selected",
    "weight",
]

COLUMNS = IDENTIFIER_COLUMNS + LABEL + FEATURES_COLUMNS


def get_input_dataMBPTP(COLUMNS, filename="overall3.csv"):  # pragma: no cover
    input_data_path = os.path.join("/opt/ml/processing/input", filename)
    df = pd.read_csv(input_data_path, usecols=COLUMNS)
    return df


def replace_missing(df):
    df = df.fillna(-999)
    return df


def exclusion(df):
    df["Exclusion"] = (
        (df["CON001OTH"] == 1)
        | (df["CON002OTH"] == 1)
        | ((df["ACC100CRT"] > -999) & (df["ACC100CRT"] <= 6))
    ).astype("int")
    df = df.drop(df[df.Exclusion == 1].index)
    return df


def mob(row):
    if row["months_on_book"] >= 0 and row["months_on_book"] <= 7:
        return 0.31
    if row["months_on_book"] >= 8 and row["months_on_book"] <= 105:
        return -0.01
    if row["months_on_book"] >= 106:
        return -0.12


def ptp24(row):
    if row["NumPTPsL24M"] == 0:
        return -0.27
    elif row["NumPTPsL24M"] == 1:
        return 0.7
    elif row["NumPTPsL24M"] == 2:
        return 0.93
    elif row["NumPTPsL24M"] >= 3:
        return 1.35


def rpc9(row):
    if row["NumRPCsL9M"] == 0:
        return -0.3
    elif row["NumRPCsL9M"] == 1:
        return 0.43
    elif row["NumRPCsL9M"] == 2:
        return 0.74
    elif row["NumRPCsL9M"] >= 3:
        return 1.08


def rec9(row):
    if row["NumRecsL9M"] == -1:
        return 1.39
    elif row["NumRecsL9M"] == 0:
        return -0.07
    elif row["NumRecsL9M"] >= 1:
        return 1.09


def age(row):
    if row["Age"] == -999:
        return 0.04
    elif row["Age"] >= 0 and row["Age"] <= 27:
        return 0.33
    elif row["Age"] >= 28 and row["Age"] <= 31:
        return 0.1
    elif row["Age"] >= 32 and row["Age"] <= 35:
        return 0.03
    elif row["Age"] >= 36 and row["Age"] <= 59:
        return -0.03
    elif row["Age"] >= 60:
        return -0.19


def acc001ucr(row):
    if row["ACC001UCR"] == -999:
        return -0.11
    elif row["ACC001UCR"] >= 0 and row["ACC001UCR"] <= 1:
        return 0.12
    elif row["ACC001UCR"] == 2:
        return -0.01
    elif row["ACC001UCR"] >= 3 and row["ACC001UCR"] <= 4:
        return -0.09
    elif row["ACC001UCR"] >= 5 and row["ACC001UCR"] <= 7:
        return -0.15
    elif row["ACC001UCR"] >= 8:
        return 0.02


def acc011crt(row):
    if row["ACC011CRT"] == -999:
        return -0.11
    elif row["ACC011CRT"] == 0:
        return 0.18
    elif row["ACC011CRT"] == 1:
        return 0.1
    elif row["ACC011CRT"] == 2:
        return 0
    elif row["ACC011CRT"] == 3:
        return -0.11
    elif row["ACC011CRT"] == 4:
        return -0.22
    elif row["ACC011CRT"] >= 5:
        return -0.31


def acc100crt(row):
    if row["ACC100CRT"] == -999:
        return -0.11
    elif row["ACC100CRT"] >= 0 and row["ACC100CRT"] <= 36:
        return 0.01
    elif row["ACC100CRT"] >= 37 and row["ACC100CRT"] <= 59:
        return -0.13
    elif row["ACC100CRT"] >= 60 and row["ACC100CRT"] <= 89:
        return 0.02
    elif row["ACC100CRT"] >= 90 and row["ACC100CRT"] <= 181:
        return 0.1
    elif row["ACC100CRT"] >= 182:
        return 0


def acc101rev(row):
    if row["ACC101REV"] == -999:
        return -0.11
    elif row["ACC101REV"] == -3:
        return -0.08
    elif row["ACC101REV"] == -2:
        return -0.05
    elif row["ACC101REV"] >= 0 and row["ACC101REV"] <= 13:
        return 0.15
    elif row["ACC101REV"] >= 14 and row["ACC101REV"] <= 69:
        return -0.04
    elif row["ACC101REV"] >= 70:
        return 0.09


def acc104crt(row):
    if row["ACC104CRT"] == -999:
        return -0.11
    elif row["ACC104CRT"] >= 0 and row["ACC104CRT"] <= 32:
        return 0.05
    elif row["ACC104CRT"] >= 33 and row["ACC104CRT"] <= 75:
        return -0.04
    elif row["ACC104CRT"] >= 76:
        return 0.05


def acc209crt(row):
    if row["ACC209CRT"] == -999:
        return -0.11
    elif row["ACC209CRT"] == 0:
        return -0.18
    if row["ACC209CRT"] == 1:
        return 0.16
    elif row["ACC209CRT"] == 2:
        return 0.1
    elif row["ACC209CRT"] == 3:
        return -0.04
    elif row["ACC209CRT"] == 4:
        return -0.13
    elif row["ACC209CRT"] >= 5:
        return -0.25


def acc230crt(row):
    if row["ACC230CRT"] == -999:
        return -0.11
    elif row["ACC230CRT"] == -5:
        return -0.4
    elif row["ACC230CRT"] == -2:
        return -0.76
    elif row["ACC230CRT"] == 0:
        return -0.11
    elif row["ACC230CRT"] >= 1 and row["ACC230CRT"] <= 3:
        return 0.33
    elif row["ACC230CRT"] == 4:
        return 0.26
    elif row["ACC230CRT"] >= 5 and row["ACC230CRT"] <= 6:
        return 0.1
    elif row["ACC230CRT"] >= 7 and row["ACC230CRT"] <= 8:
        return -0.1
    elif row["ACC230CRT"] >= 9:
        return -0.23


def acc233crt(row):
    if row["ACC233CRT"] == -999:
        return -0.11
    elif row["ACC233CRT"] >= 0 and row["ACC233CRT"] <= 3:
        return -0.12
    elif row["ACC233CRT"] >= 4 and row["ACC233CRT"] <= 8:
        return 0.17
    elif row["ACC233CRT"] >= 9:
        return 0


def acc234crt(row):
    if row["ACC234CRT"] == -999:
        return -0.11
    elif row["ACC234CRT"] == 0:
        return -0.31
    elif row["ACC234CRT"] >= 1 and row["ACC234CRT"] <= 10:
        return 0.13
    elif row["ACC234CRT"] >= 11 and row["ACC234CRT"] <= 55:
        return 0.04
    elif row["ACC234CRT"] >= 56 and row["ACC234CRT"] <= 99:
        return -0.05
    elif row["ACC234CRT"] >= 100:
        return 0.17


def acc309nct(row):
    if row["ACC309NCT"] == -999:
        return -0.11
    elif row["ACC309NCT"] == -2:
        return 0.07
    elif row["ACC309NCT"] == -1:
        return -0.01
    elif row["ACC309NCT"] >= 0 and row["ACC309NCT"] <= 824:
        return 0.13
    elif row["ACC309NCT"] >= 825 and row["ACC309NCT"] <= 2749:
        return -0.08
    elif row["ACC309NCT"] >= 2750:
        return -0.33


def acc314crt(row):
    if row["ACC314CRT"] == -999:
        return -0.11
    elif row["ACC314CRT"] == -2:
        return -0.76
    elif row["ACC314CRT"] == 0:
        return -0.14
    elif row["ACC314CRT"] >= 1 and row["ACC314CRT"] <= 804:
        return 0.21
    elif row["ACC314CRT"] >= 805 and row["ACC314CRT"] <= 1649:
        return 0.47
    elif row["ACC314CRT"] >= 1650 and row["ACC314CRT"] <= 2499:
        return 0.33
    elif row["ACC314CRT"] >= 2500 and row["ACC314CRT"] <= 3499:
        return 0.26
    elif row["ACC314CRT"] >= 3500 and row["ACC314CRT"] <= 4499:
        return 0.21
    elif row["ACC314CRT"] >= 4500 and row["ACC314CRT"] <= 5549:
        return 0.16
    elif row["ACC314CRT"] >= 5550 and row["ACC314CRT"] <= 6949:
        return 0.10
    elif row["ACC314CRT"] >= 6950 and row["ACC314CRT"] <= 8399:
        return 0.04
    elif row["ACC314CRT"] >= 8400 and row["ACC314CRT"] <= 14649:
        return -0.04
    elif row["ACC314CRT"] >= 14650 and row["ACC314CRT"] <= 44399:
        return -0.16
    elif row["ACC314CRT"] >= 44400 and row["ACC314CRT"] <= 117999:
        return -0.24
    elif row["ACC314CRT"] >= 118000:
        return -0.42


def enq004oth(row):
    if row["ENQ004OTH"] == -999:
        return -0.11
    elif row["ENQ004OTH"] == -4:
        return -0.17
    elif row["ENQ004OTH"] == 0:
        return 0.08
    elif row["ENQ004OTH"] == 1:
        return 0.14
    elif row["ENQ004OTH"] == 2:
        return 0.26
    elif row["ENQ004OTH"] >= 3:
        return 0.15


def leg200oth(row):
    if row["LEG200OTH"] == -999:
        return -0.11
    elif row["LEG200OTH"] == 0:
        return -0.01
    elif row["LEG200OTH"] >= 1:
        return 0.18


def process_column_types(df):  # pragma: no cover
    df.apply(pd.to_numeric, errors="ignore")
    df["score_month"] = pd.to_datetime(df["score_month"])
    return df


def fill_in_missing_values(df):
    "Fill in missing values"
    df["Gender"] = df["Gender"].fillna(-999)
    df["Age"] = df["Age"].fillna(df["Age"].median())
    return df


def preprocess_categorical_columns(df):
    df["Gender"] = df["Gender"].apply(lambda gender: 1 if gender == "F" else 0)
    return df


def train_test_split2(df, woevars=woevars):  # pragma: no cover
    training_df = df[(df.Selected == 1)]
    # Validation data
    validation_df = df[(df.Selected == 0)]

    print(f"Train data shape after preprocessing: {training_df.shape}")
    print(f"Test data shape after preprocessing: {validation_df.shape}")

    # Training dataset
    train_y = training_df["target"]
    train_X = training_df[woevars]
    train_weight = training_df["weight"]
    # Validation dataset
    val_y = validation_df["target"]
    val_X = validation_df[woevars]
    val_weight = validation_df["weight"]

    train_features_output_path = os.path.join(
        "/opt/ml/processing/train", "train_featuresMBPTP.csv"
    )
    train_labels_output_path = os.path.join(
        "/opt/ml/processing/train", "train_labelsMBPTP.csv"
    )
    train_weight_output_path = os.path.join(
        "/opt/ml/processing/train", "train_weightMBPTP.csv"
    )

    test_features_output_path = os.path.join(
        "/opt/ml/processing/validation", "validation_featuresMBPTP.csv"
    )
    test_labels_output_path = os.path.join(
        "/opt/ml/processing/validation", "validation_labelsMBPTP.csv"
    )
    test_weight_output_path = os.path.join(
        "/opt/ml/processing/validation", "validation_weightMBPTP.csv"
    )

    print(f"Saving MB PTP training features to {train_features_output_path}")
    train_X.to_csv(train_features_output_path, header=False, index=False)

    print(f"Saving MB PTP validation features to {test_features_output_path}")
    val_X.to_csv(test_features_output_path, header=False, index=False)

    print(f"Saving MB PTP training weights to {train_labels_output_path}")
    train_y.to_csv(train_labels_output_path, header=False, index=False)

    print(f"Saving MB PTP validation labels to {test_labels_output_path}")
    val_y.to_csv(test_labels_output_path, header=False, index=False)

    print(f"Saving MB PTP training labels to {train_weight_output_path}")
    train_weight.to_csv(train_weight_output_path, header=False, index=False)

    print(f"Saving MB PTP validation weights to {test_weight_output_path}")
    val_weight.to_csv(test_weight_output_path, header=False, index=False)


def main(df):
    logger.info("Reading CSV data.")
    df = replace_missing(df)
    df = exclusion(df)
    df["w_mob"] = df.apply(mob, axis=1)
    df["w_NumPTPsL24"] = df.apply(ptp24, axis=1)
    df["w_NumRecsL9"] = df.apply(rec9, axis=1)
    df["w_NumRPCL9"] = df.apply(rpc9, axis=1)
    df["w_Age"] = df.apply(age, axis=1)
    df["w_ACC001UCR"] = df.apply(acc001ucr, axis=1)
    df["w_ACC011CRT"] = df.apply(acc011crt, axis=1)
    df["w_ACC100CRT"] = df.apply(acc100crt, axis=1)
    df["w_ACC101REV"] = df.apply(acc101rev, axis=1)
    df["w_ACC104CRT"] = df.apply(acc104crt, axis=1)
    df["w_ACC209CRT"] = df.apply(acc209crt, axis=1)
    df["w_ACC230CRT"] = df.apply(acc230crt, axis=1)
    df["w_ACC233CRT"] = df.apply(acc233crt, axis=1)
    df["w_ACC234CRT"] = df.apply(acc234crt, axis=1)
    df["w_ACC309NCT"] = df.apply(acc309nct, axis=1)
    df["w_ACC314CRT"] = df.apply(acc314crt, axis=1)
    df["w_ENQ004OTH"] = df.apply(enq004oth, axis=1)
    df["w_LEG200OTH"] = df.apply(leg200oth, axis=1)
    return df


if __name__ == "__main__":  # pragma: no cover
    df = get_input_dataMBPTP(COLUMNS, filename="overall3.csv")
    df = main(df)
    train_test_split2(df)
