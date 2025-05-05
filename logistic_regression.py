# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# For binary classification, choose only two classes (e.g., Setosa and Versicolor)
X_binary = X[y != 2]
y_binary = y[y != 2]

# Select only two features (e.g., sepal length and sepal width)
X_binary = X_binary[:, :2]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Make predictions
y_pred = log_reg.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=target_names[:2]))

# Example: Predict on new data
import numpy as np
sample = np.array([[5.0, 3.5]])  # Sample sepal length and width
prediction = log_reg.predict(sample)
print("Prediction (0 = Setosa, 1 = Versicolor):", prediction)
