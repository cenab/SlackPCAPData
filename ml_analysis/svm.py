from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
df = pd.read_csv('your_flow_file.csv')

# Split data
X = df.drop('target', axis=1)
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize and fit SVM
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Predict and evaluate
predictions = svm_model.predict(X_test)
