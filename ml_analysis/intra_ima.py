import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils import shuffle
from pathlib import Path
import sys
# config.py

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
# Define classifiers dictionary with default settings from scikit-learn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

from sklearn.impute import SimpleImputer



cdic = {
    'NearestNeighbors': KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean'),
    # 'LinearSVM': SVC(kernel='linear', C=1.0, class_weight='balanced'),
    'DecisionTree': DecisionTreeClassifier(max_depth=None, min_samples_split=2, min_samples_leaf=1),
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt'),
    'NeuralNetwork': None,
    'NaiveBayes': None,
    'LogisticRegression': None,
    'GradientBoost': GradientBoostingClassifier(n_estimators=10, learning_rate=0.4, max_depth=3)

}

# Corresponding names for the classifiers in a human-readable format
cdic_names = {
    'NearestNeighbors': 'Nearest Neighbors',
    # 'LinearSVM': 'Linear SVM',
    'DecisionTree': 'Decision Tree',
    'RandomForest': 'Random Forest',
    'NeuralNetwork': 'Neural Network',
    'NaiveBayes': 'Naive Bayes',
    'LogisticRegression': 'Logistic Regression',
    'GradientBoost': 'Gradient Boost'
}

# Updated list of applications you're analyzing, including 'IMA_Slack'
apps = ['IMA_Whatsapp', 'IMA_Messenger', 'IMA_Telegram', 'IMA_Teams', 'IMA_Discord', 'IMA_Signal', 'IMA_Slack']

# Full names for the applications for more readable output/reporting, with 'IMA_Slack' added
apps_fullname = {
    'IMA_Whatsapp': 'WhatsApp',
    'IMA_Messenger': 'Messenger',
    'IMA_Telegram': 'Telegram',
    'IMA_Teams': 'Microsoft Teams',
    'IMA_Discord': 'Discord',
    'IMA_Signal': 'Signal',
    'IMA_Slack': 'Slack',
}

# Root directory for plots and other output files
plots_root = '/Users/batu/Desktop/project_dal/IMA_traffic_analysis'  # Ensure this path is correct for your system

# Mapping app names to numeric identifiers
app_to_num = {app: i for i, app in enumerate(apps)}

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

def export_df(df, filepath):
    full_path = Path(plots_root) / filepath
    full_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(full_path, index=False)

def feature_selection(comb, name):
    # comb = comb.drop(columns=['ethType'])
    # print(comb)
    y, X = input_label(comb)
    infos = mutual_info_classif(X, y)
    significant_features = {col: info for col, info in zip(X.columns, infos) if info > 0.02}
    df = pd.DataFrame({"Feature": significant_features.keys(), "Mutual Information": significant_features.values()})
    export_df(df, f"{name}/mutual_info_features")
    return comb[list(significant_features.keys()) + ["label"]]

def choose_features(comb, features, name, typee="intra"):
    label_mapping = {"in": 0, "out": 1} if typee == "inter" else app_to_num
    comb["label"] = comb["label"].map(lambda x: label_mapping[x])

    # Drop unnecessary columns safely
    unnecessary_cols = ["timeFirst", "timeLast", "flowInd", "macStat", "macPairs", "srcMac_dstMac_numP", "srcMacLbl_dstMacLbl", "srcMac", "dstMac", "srcPort", "hdrDesc", "duration"]
    comb = comb.drop(columns=[col for col in unnecessary_cols if col in comb.columns])
    # print(comb)
    # Identify and transform categorical columns
    categorical_cols = [col for col in comb.select_dtypes(include=['object']).columns.tolist() if col in comb.columns]
    for col in categorical_cols:
        comb[col] = comb[col].astype('category').cat.codes

    feature_sets = {
        "all": comb.columns.tolist(),
        "categorical": ["label"] + categorical_cols,
        "statistical": [col for col in comb.columns if col not in categorical_cols + ['label']],
        "custom_categorical": [col for col in ["dstIPOrg", "srcIPOrg", "%dir", "dstPortClass", "label"] if col in comb.columns],
        "custom_statistical": [col for col in ["label", "numPktsSnt", "numPktsRcvd", "numBytesSnt", "numBytesRcvd", "minPktSz", "maxPktSz", "avePktSize", "stdPktSize", "minIAT", "maxIAT", "aveIAT", "stdIAT", "bytps"] if col in comb.columns],
        "mutual_info": feature_selection(comb, name)["label"]
    }

    selected_features = feature_sets.get(features, "all")
    return comb[selected_features]


# Function to generate the file name for a given application
def tran_name(app, name):
    return f"/Users/batu/Desktop/project_dal/android_pcap_data_collection_and_analysis/flows/{name}/{name}_encrypted_traffic_sanitized_flows.txt"

# Function

def setPipelines(comb, cdic=cdic):
     # Dynamically identify numerical and categorical features from 'comb'
    numerical_features = comb.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = comb.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())]), numerical_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features),
        ]
    )

    # Update the NeuralNetwork pipeline within the cdic dictionary
    cdic['NeuralNetwork'] = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', MLPClassifier(hidden_layer_sizes=(1024, 512, 256), activation='tanh', alpha=0.0001, max_iter=1000))
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', ColumnTransformer(
            transformers=[
                # You might consider commenting out or simplifying some of these preprocessing steps
                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())  # Consider removing the scaler to reduce preprocessing
                ]), numerical_features),
                ('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), categorical_features)
            ]
        )),  # First, apply the preprocessor
        ('gnb', GaussianNB(var_smoothing=1e-14))     # Then, use the classifier with very low var_smoothing
    ])

    # Reduced cross-validation folds and use all CPUs available for GridSearchCV
    cdic['NaiveBayes'] = GridSearchCV(pipeline, param_grid={'gnb__var_smoothing': [1e-14]}, cv=3, scoring='accuracy', n_jobs=-1)
    cdic['LogisticRegression'] = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(C=0.5, solver='saga', max_iter=500, class_weight='balanced'))])

    return;

# Function to process data for a given experiment
def process(name, direction, features, cdic=cdic, cross_validateq=False):
    arr = []
    for app in apps:
        df = pd.read_csv(tran_name(app,name), delimiter= ',')
        df = df.sample(n=len(df.index))
        df["label"] = app
        arr.append(df)
    Path(f"{plots_root}/{name}").mkdir(parents=True, exist_ok=True)
    comb = pd.concat(arr)
    # print(comb)
    comb = comb.dropna(axis=1, how='all')
    comb = choose_features(comb, features, name)
    comb = shuffle(comb)

    arr = []
    f1_ranges = []
    accur_ranges = []
    pred_ranges = []
    recall_ranges = []
    for clf_name in cdic.keys():
        clf = cdic[clf_name]
        if cross_validateq == False:
            xy_train = comb.groupby("label").sample(n=20, random_state=1)
            x_train = xy_train.drop("label", axis=1)
            y_train = xy_train["label"]
            setPipelines(x_train)
            
            xy_test = comb.drop(xy_train.index)
            x_test = xy_test.drop("label", axis=1)
            y_test = xy_test['label']
            clf.fit(x_train, y_train)
            # y_predic = clf.predict(x_test)
            train_predic = clf.predict(x_train)

            precision, recall, f1, _ = precision_recall_fscore_support(y_train, train_predic, average='macro', zero_division=0)

            arr.append([accuracy_score(y_train, train_predic), precision, recall, f1])
        else:
            f1s = []
            accurs = []
            precs = []
            recalls = []
            for i in range(1, 8):  # Example for a 10-fold scenario
                # Split the data into train and test sets before sampling for train
                train_df, remaining_df = train_test_split(comb, test_size=0.2, random_state=i)

                sampled_dfs = []
                for label, group in train_df.groupby("label"):
                    min_group_size = min(train_df.groupby("label").size())
                    sampled_group = group.sample(n=min(min_group_size, 3179), random_state=i)  # Sample 3179 instances from each class
                    sampled_dfs.append(sampled_group)  # Append the sampled DataFrame to the list

                xy_train = pd.concat(sampled_dfs)  # Concatenate all sampled DataFrames to form 'xy_train'
                x_train = xy_train.drop("label", axis=1)
                y_train = xy_train["label"]

                # Use the remaining_df to create the test set, similar to dropping xy_train from comb in the first snippet
                xy_test = remaining_df.drop(xy_train.index.intersection(remaining_df.index), errors='ignore')
                x_test = xy_test.drop("label", axis=1)
                y_test = xy_test['label']

                clf.fit(x_train, y_train)
                print(x_train.shape)
                print(x_test.shape)
                # y_predic = clf.predict(x_test)
                train_predic = clf.predict(x_train)



                accurs.append(accuracy_score(y_train, train_predic))
                precision, recall, f1, _ = precision_recall_fscore_support(y_train, train_predic, average='macro', zero_division=0)
                f1s.append(f1)
                precs.append(precision)
                recalls.append(recall)
            f1_ranges.append(f"{round(min(f1s),3)}-{round(max(f1s),3)}")
            accur_ranges.append(f"{round(min(accurs),3)}-{round(max(accurs),3)}")
            pred_ranges.append(f"{round(min(precs),3)}-{round(max(precs),3)}")
            recall_ranges.append(f"{round(min(recalls),3)}-{round(max(recalls),3)}")
            df = pd.DataFrame({"Accuracy": accurs, "Precision": precs, "Recall": recalls, "F1": f1s})
            export_df(df, f"{name}/{name}_fold_{clf_name}_{direction}_{features}")
            
    if cross_validateq == True:
        df = pd.DataFrame({"Model": cdic_names, "Accuracy": accur_ranges, "Precision": pred_ranges, "Recall": recall_ranges, "F1": f1_ranges})
        df["mins"] = df["Accuracy"].map(lambda x: float(x.split("-")[0]))
        df.sort_values(by=["mins"], ascending=False)
        df = df.drop("mins", axis=1)
        export_df(df, f"{name}/all_models_range/{name}_{features}_compare_models")
    if cross_validateq == False:
        df = pd.DataFrame(arr, columns=["Accuracy","Precision","Recall","F1"])
        df["Classifier"] = cdic.keys()
        export_df(df, f"{name}/{name}_allclass_{clf_name}_{direction}_{features}")


# Function to build a count table for the number of flows per app
def build_count_table(name):
    dfs = []
    for app in apps:
        df = pd.read_csv(tran_name(app, name), delimiter='\s+', index_col=False)
        df["label"] = app
        dfs.append(df)
    count_df = pd.concat(dfs)
    app_counts = count_df['label'].value_counts().rename_axis('App').reset_index(name='Number of Total Flows')
    app_counts['App'] = app_counts['App'].map(apps_fullname)
    export_df(app_counts, f"{name}/num_flows_{name}.csv")

# Main execution logic
if __name__ == "__main__":
    name = sys.argv[1]
    function = sys.argv[2]
    feature_options = ["all", "categorical", "statistical", "custom_statistical"]
    if function == "process":
        for feature in feature_options:
           process(name, "both", feature, cross_validateq=False)
    elif function == "count":
        build_count_table(name)

