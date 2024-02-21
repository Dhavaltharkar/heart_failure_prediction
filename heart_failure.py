import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv("heart_failure_clinical_records_dataset.csv")

# Splitting the data into X and y
# Select features
selected_features = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
                     'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']

X = data[selected_features]
y = data['DEATH_EVENT']

# Splitting the dataset into train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# It's a good practice to scale your data when using Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)  # Increasing max_iter for convergence
logistic_model.fit(X_train_scaled, y_train)

# Save the Logistic Regression model as a pickle file
with open('logistic_heart_failure_model.pkl', 'wb') as f:
    pickle.dump(logistic_model, f)

# Model Evaluation
y_pred = logistic_model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
