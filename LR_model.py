import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_wine

# Load Wine Dataset
wine = load_wine()
X = wine.data
Y = wine.target

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Simple Linear Regression Model (Using first feature for demonstration)
simple_model = LinearRegression()
simple_model.fit(X_train[:, [0]], Y_train)  # Using only the first feature
Y_pred_simple = simple_model.predict(X_test[:, [0]])

# Multiple Linear Regression Model
multiple_model = LinearRegression()
multiple_model.fit(X_train, Y_train)
Y_pred_multiple = multiple_model.predict(X_test)

# Evaluation Metrics for Simple Linear Regression
simple_mse = mean_squared_error(Y_test, Y_pred_simple)
simple_r2 = r2_score(Y_test, Y_pred_simple)

# Evaluation Metrics for Multiple Linear Regression
multiple_mse = mean_squared_error(Y_test, Y_pred_multiple)
multiple_r2 = r2_score(Y_test, Y_pred_multiple)

# K-Fold Cross-Validation for Multiple Values of k
k_values = [3, 5, 7, 10]
simple_cv_results = {}
multiple_cv_results = {}

for k in k_values:
    simple_cv_scores = cross_val_score(simple_model, X[:, [0]], Y, cv=k, scoring='r2')
    multiple_cv_scores = cross_val_score(multiple_model, X, Y, cv=k, scoring='r2')

    simple_cv_results[k] = np.mean(simple_cv_scores)
    multiple_cv_results[k] = np.mean(multiple_cv_scores)

# Display Results
print("\nSimple Linear Regression:")
print(f"MSE: {simple_mse}")
print(f"R² Score: {simple_r2}")
for k, score in simple_cv_results.items():
    print(f"{k}-Fold CV Average R² Score: {score}")

print("\nMultiple Linear Regression:")
print(f"MSE: {multiple_mse}")
print(f"R² Score: {multiple_r2}")
for k, score in multiple_cv_results.items():
    print(f"{k}-Fold CV Average R² Score: {score}")
