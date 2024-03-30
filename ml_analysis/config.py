# config.py

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# Define classifiers dictionary with default settings from scikit-learn
cdic = {
    'NearestNeighbors': KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean'),
    # 'LinearSVM': SVC(kernel='linear', C=1.0, class_weight='balanced'),
    'DecisionTree': DecisionTreeClassifier(max_depth=None, min_samples_split=2, min_samples_leaf=1),
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto'),
    'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(100,), activation='relu', alpha=0.0001),
    'NaiveBayes': GaussianNB(var_smoothing=1e-9),
    'LogisticRegression': LogisticRegression(C=1.0, solver='liblinear', class_weight='balanced'),
    'GradientBoost': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
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
plots_root = '/Users/batu/Desktop/project_dal/android_pcap_data_collection_and_analysis'  # Ensure this path is correct for your system
