# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris

# Loading the Iris dataset
data = load_iris()
X = data.data  # Features
y = data.target  # Labels (multi-class classification)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initializing SVM model with RBF kernel (default)
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')  

# Fitting the model on the training data
svm_model.fit(X_train, y_train)

# Making predictions on the test data
y_pred = svm_model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)

# Example: Predicting for new data
sample_data = np.array([X_test[0]])   
prediction = svm_model.predict(sample_data)
print("Prediction:", prediction)
