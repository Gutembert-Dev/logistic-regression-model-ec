import os
import pandas as pd


def get_input_dataMBPTP(COLUMNS, filename="overall3.csv"):
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
        val = 0.31
    elif row["months_on_book"] >= 8 and row["months_on_book"] <= 105:
        val = -0.01
    elif row["months_on_book"] >= 106:
        val = -0.12
    return val


def ptp24(row):
    if row["NumPTPsL24M"] == 0:
        val = -0.27
    elif row["NumPTPsL24M"] == 1:
        val = 0.7
    elif row["NumPTPsL24M"] == 2:
        val = 0.93
    elif row["NumPTPsL24M"] >= 3:
        val = 1.35
    return val


def rpc9(row):
    if row["NumRPCsL9M"] == 0:
        val = -0.3
    elif row["NumRPCsL9M"] == 1:
        val = 0.43
    elif row["NumRPCsL9M"] == 2:
        val = 0.74
    elif row["NumRPCsL9M"] >= 3:
        val = 1.08
    return val


def rec9(row):
    if row["NumRecsL9M"] == -1:
        val = 1.39
    elif row["NumRecsL9M"] == 0:
        val = -0.07
    elif row["NumRecsL9M"] >= 1:
        val = 1.09
    return val


def age(row):
    if row["Age"] == -999:
        val = 0.04
    elif row["Age"] >= 0 and row["Age"] <= 27:
        val = 0.33
    elif row["Age"] >= 28 and row["Age"] <= 31:
        val = 0.1
    elif row["Age"] >= 32 and row["Age"] <= 35:
        val = 0.03
    elif row["Age"] >= 36 and row["Age"] <= 59:
        val = -0.03
    elif row["Age"] >= 60:
        val = -0.19
    return val


def acc001ucr(row):
    if row["ACC001UCR"] == -999:
        val = -0.11
    elif row["ACC001UCR"] >= 0 and row["ACC001UCR"] <= 1:
        val = 0.12
    elif row["ACC001UCR"] == 2:
        val = -0.01
    elif row["ACC001UCR"] >= 3 and row["ACC001UCR"] <= 4:
        val = -0.09
    elif row["ACC001UCR"] >= 5 and row["ACC001UCR"] <= 7:
        val = -0.15
    elif row["ACC001UCR"] >= 8:
        val = 0.02
    return val


def acc011crt(row):
    if row["ACC011CRT"] == -999:
        val = -0.11
    elif row["ACC011CRT"] == 0:
        val = 0.18
    elif row["ACC011CRT"] == 1:
        val = 0.1
    elif row["ACC011CRT"] == 2:
        val = 0
    elif row["ACC011CRT"] == 3:
        val = -0.11
    elif row["ACC011CRT"] == 4:
        val = -0.22
    elif row["ACC011CRT"] >= 5:
        val = -0.31
    return val


def acc100crt(row):
    if row["ACC100CRT"] == -999:
        val = -0.11
    elif row["ACC100CRT"] >= 0 and row["ACC100CRT"] <= 36:
        val = 0.01
    elif row["ACC100CRT"] >= 37 and row["ACC100CRT"] <= 59:
        val = -0.13
    elif row["ACC100CRT"] >= 60 and row["ACC100CRT"] <= 89:
        val = 0.02
    elif row["ACC100CRT"] >= 90 and row["ACC100CRT"] <= 181:
        val = 0.1
    elif row["ACC100CRT"] >= 182:
        val = 0
    return val


def acc101rev(row):
    if row["ACC101REV"] == -999:
        val = -0.11
    elif row["ACC101REV"] == -3:
        val = -0.08
    elif row["ACC101REV"] == -2:
        val = -0.05
    elif row["ACC101REV"] >= 0 and row["ACC101REV"] <= 13:
        val = 0.15
    elif row["ACC101REV"] >= 14 and row["ACC101REV"] <= 69:
        val = -0.04
    elif row["ACC101REV"] >= 70:
        val = 0.09
    return val


def acc104crt(row):
    if row["ACC104CRT"] == -999:
        val = -0.11
    elif row["ACC104CRT"] >= 0 and row["ACC104CRT"] <= 32:
        val = 0.05
    elif row["ACC104CRT"] >= 33 and row["ACC104CRT"] <= 75:
        val = -0.04
    elif row["ACC104CRT"] >= 76:
        val = 0.05
    return val


def acc209crt(row):
    if row["ACC209CRT"] == -999:
        val = -0.11
    elif row["ACC209CRT"] == 0:
        val = -0.18
    if row["ACC209CRT"] == 1:
        val = 0.16
    elif row["ACC209CRT"] == 2:
        val = 0.1
    elif row["ACC209CRT"] == 3:
        val = -0.04
    elif row["ACC209CRT"] == 4:
        val = -0.13
    elif row["ACC209CRT"] >= 5:
        val = -0.25
    return val


def acc230crt(row):
    if row["ACC230CRT"] == -999:
        val = -0.11
    elif row["ACC230CRT"] == -5:
        val = -0.4
    elif row["ACC230CRT"] == -2:
        val = -0.76
    elif row["ACC230CRT"] == 0:
        val = -0.11
    elif row["ACC230CRT"] >= 1 and row["ACC230CRT"] <= 3:
        val = 0.33
    elif row["ACC230CRT"] == 4:
        val = 0.26
    elif row["ACC230CRT"] >= 5 and row["ACC230CRT"] <= 6:
        val = 0.1
    elif row["ACC230CRT"] >= 7 and row["ACC230CRT"] <= 8:
        val = -0.1
    elif row["ACC230CRT"] >= 9:
        val = -0.23
    return val


def acc233crt(row):
    if row["ACC233CRT"] == -999:
        val = -0.11
    elif row["ACC233CRT"] >= 0 and row["ACC233CRT"] <= 3:
        val = -0.12
    elif row["ACC233CRT"] >= 4 and row["ACC233CRT"] <= 8:
        val = 0.17
    elif row["ACC233CRT"] >= 9:
        val = 0
    return val


def acc234crt(row):
    if row["ACC234CRT"] == -999:
        val = -0.11
    elif row["ACC234CRT"] == 0:
        val = -0.31
    elif row["ACC234CRT"] >= 1 and row["ACC234CRT"] <= 10:
        val = 0.13
    elif row["ACC234CRT"] >= 11 and row["ACC234CRT"] <= 55:
        val = 0.04
    elif row["ACC234CRT"] >= 56 and row["ACC234CRT"] <= 99:
        val = -0.05
    elif row["ACC234CRT"] >= 100:
        val = 0.17
    return val


def acc309nct(row):
    if row["ACC309NCT"] == -999:
        val = -0.11
    elif row["ACC309NCT"] == -2:
        val = 0.07
    elif row["ACC309NCT"] == -1:
        val = -0.01
    elif row["ACC309NCT"] >= 0 and row["ACC309NCT"] <= 824:
        val = 0.13
    elif row["ACC309NCT"] >= 825 and row["ACC309NCT"] <= 2749:
        val = -0.08
    elif row["ACC309NCT"] >= 2750:
        val = -0.33
    return val


def acc314crt(row):
    if row["ACC314CRT"] == -999:
        val = -0.11
    elif row["ACC314CRT"] == -2:
        val = -0.76
    elif row["ACC314CRT"] == 0:
        val = -0.14
    elif row["ACC314CRT"] >= 1 and row["ACC314CRT"] <= 804:
        val = 0.21
    elif row["ACC314CRT"] >= 805 and row["ACC314CRT"] <= 1649:
        val = 0.47
    elif row["ACC314CRT"] >= 1650 and row["ACC314CRT"] <= 2499:
        val = 0.33
    elif row["ACC314CRT"] >= 2500 and row["ACC314CRT"] <= 3499:
        val = 0.26
    elif row["ACC314CRT"] >= 3500 and row["ACC314CRT"] <= 4499:
        val = 0.21
    elif row["ACC314CRT"] >= 4500 and row["ACC314CRT"] <= 5549:
        val = 0.16
    elif row["ACC314CRT"] >= 5550 and row["ACC314CRT"] <= 6949:
        val = 0.10
    elif row["ACC314CRT"] >= 6950 and row["ACC314CRT"] <= 8399:
        val = 0.04
    elif row["ACC314CRT"] >= 8400 and row["ACC314CRT"] <= 14649:
        val = -0.04
    elif row["ACC314CRT"] >= 14650 and row["ACC314CRT"] <= 44399:
        val = -0.16
    elif row["ACC314CRT"] >= 44400 and row["ACC314CRT"] <= 117999:
        val = -0.24
    elif row["ACC314CRT"] >= 118000:
        val = -0.42
    return val


def enq004oth(row):
    if row["ENQ004OTH"] == -999:
        val = -0.11
    elif row["ENQ004OTH"] == -4:
        val = -0.17
    elif row["ENQ004OTH"] == 0:
        val = 0.08
    elif row["ENQ004OTH"] == 1:
        val = 0.14
    elif row["ENQ004OTH"] == 2:
        val = 0.26
    elif row["ENQ004OTH"] >= 3:
        val = 0.15
    return val


def leg200oth(row):
    if row["LEG200OTH"] == -999:
        val = -0.11
    elif row["LEG200OTH"] == 0:
        val = -0.01
    elif row["LEG200OTH"] >= 1:
        val = 0.18
    return val


def process_column_types(df):
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


def train_test_split2(df):
    training_df = df[(df.Selected == 1)]
    # Validation data
    validation_df = df[(df.Selected == 0)]

    print(f"Train data shape after preprocessing: {training_df.shape}")
    print(f"Test data shape after preprocessing: {validation_df.shape}")

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
