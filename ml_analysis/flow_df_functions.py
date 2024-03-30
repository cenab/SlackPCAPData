import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

# Assuming 'export_df', 'export_cm', 'display', 'app_to_num', 'cdic', and 'cdic_names' are defined in imported modules
from your_module import export_df, export_cm, display, app_to_num, cdic, cdic_names

def all_features(df):
    return df.drop(["lengths", "timestamps", "directions", "label"], axis=1)

def only_stat_features(df):
    return df.drop(["lengths", "timestamps", "directions", "label", "flow"], axis=1)

def filter_by_direction(df, direction):
    return df[df["directions"] == direction]

def in_all_features(df):
    return filter_by_direction(df, "B")

def out_all_features(df):
    return filter_by_direction(df, "A")

def in_stat_features(df):
    return only_stat_features(filter_by_direction(df, "B"))

def out_stat_features(df):
    return only_stat_features(filter_by_direction(df, "A"))

def input_label(df):
    labels = df["label"]
    inputs = df.drop("label", axis=1)
    return labels, inputs

def feature_selection(comb, name):
    y, X = input_label(comb)
    infos = mutual_info_classif(X, y)
    significant_features = {col: info for col, info in zip(X.columns, infos) if info > 0.2}
    df = pd.DataFrame({"Feature": significant_features.keys(), "Mutual Information": significant_features.values()})
    export_df(df, f"{name}/mutual_info_features")
    return comb[list(significant_features.keys()) + ["label"]]

def choose_features(comb, features, name, typee="intra"):
    label_mapping = {"in": 0, "out": 1} if typee == "inter" else app_to_num
    comb["label"] = comb["label"].map(lambda x: label_mapping[x])

    categorical_cols = comb.select_dtypes(include=['object']).columns.tolist()
    comb = comb.fillna(0).drop(["timeFirst", "timeLast", "flowInd"], axis=1)
    comb = comb.drop(columns=["macStat", "macPairs", "srcMac_dstMac_numP", "srcMacLbl_dstMacLbl", "srcMac", "dstMac", "srcPort", "hdrDesc", "duration"], errors='ignore')
    comb[categorical_cols] = comb[categorical_cols].apply(lambda col: col.astype("category").cat.codes)

    feature_sets = {
        "all": comb.columns.tolist(),
        "categorical": ["label"] + categorical_cols,
        "statistical": comb.drop(columns=categorical_cols, errors='ignore').columns.tolist(),
        "custom_categorical": ["dstIPOrg", "srcIPOrg", "%dir", "dstPortClass", "label"],
        "custom_statistical": ["label", "numPktsSnt", "numPktsRcvd", "numBytesSnt", "numBytesRcvd", "minPktSz", "maxPktSz", "avePktSize", "stdPktSize", "minIAT", "maxIAT", "aveIAT", "stdIAT", "bytps"],
        "mutual_info": feature_selection(comb, name)["label"]
    }

    selected_features = feature_sets.get(features, "all")
    return comb[selected_features]

def import_csv(name, features, typee):
    dfs = []
    for app in apps:
        df = pd.read_csv(tran_name(app, name, "both"), delimiter='\s+', index_col=False)
        df["label"] = app
        dfs.append(df)
    comb = pd.concat(dfs)
    comb = choose_features(comb, features, name, typee)
    comb = shuffle(comb).reset_index(drop=True)
    display(comb)
    comb.to_csv(f"{csvs}/{name}_{features}.csv", index=False)

def train_plotrange(comb, labels, name, cross_validate, direction, features, typee):
    y, X = input_label(comb)
    results = []

    if not cross_validate:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
        for clf_name, clf in cdic.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            results.append([accuracy_score(y_test, y_pred), *score(y_test, y_pred, average='macro')[:3]])
            export_cm(y_test, y_pred, f"{
