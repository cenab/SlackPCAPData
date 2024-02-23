import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import shuffle


file_path = '/Users/batu/Desktop/project_dal/android_pcap_data_collection_and_analysis/flows/slack/slack_edited_flows.txt'  # Update this path

# Load the flow file
def load_flow_file(file_path):
    df = pd.read_csv(file_path, delimiter='\s+', index_col=False)
    df["label"] = "slack"
    return df


def drop_columns(df):
    columns_to_drop = ["timeFirst", "timeLast", "flowInd", "macStat", "macPairs", "srcMac_dstMac_numP", "srcMacLbl_dstMacLbl", "srcMac", "dstMac", "srcPort", "hdrDesc", "duration"]
    df = df.drop(columns_to_drop, axis=1)
    return df

# Function for feature selection based on mutual information
def feature_selection(df):
    labels = df["label"]
    inputs = df.drop("label", axis=1)
    mutual_infos = mutual_info_classif(labels, inputs)
    selected_features = X.columns[mutual_infos > 0.2]
    return df[selected_features.union(['label'])]

df = load_flow_file(file_path)
# Preprocess the DataFrame
df_processed = drop_columns(df)
df_selected_features = feature_selection(df_processed)
df_shuffled = shuffle(df_selected_features).reset_index(drop=True)

# Split into features and labels
X = df_selected_features.drop('label', axis=1)
y = df_selected_features['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))

# Linear SVM
svm = SVC(kernel='linear')
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)  # Not scaling for tree-based models
y_pred_dt = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)  # Not scaling for tree-based models
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# Multi-Layer Perceptron
mlp = MLPClassifier()
mlp.fit(X_train_scaled, y_train)
y_pred_mlp = mlp.predict(X_test_scaled)
print("MLP Accuracy:", accuracy_score(y_test, y_pred_mlp))

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
y_pred_nb = nb.predict(X_test_scaled)
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))

# Gradient Boost
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)  # Not scaling for tree-based models
y_pred_gb = gb.predict(X_test)
print("Gradient Boost Accuracy:", accuracy_score(y_test, y_pred_gb))
