import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load Wine dataset
wine = load_wine()
X = wine.data
y = X[:, 0]  # Let's predict 'Alcohol' (feature 0)
X = np.delete(X, 0, axis=1)  # Remove Alcohol from features

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)
ridge_mse = mean_squared_error(y_test, ridge_pred)
print(f"Ridge Regression MSE: {ridge_mse:.4f}")

# Lasso Regression
lasso_model = Lasso(alpha=1.0, max_iter=10000)
lasso_model.fit(X_train, y_train)
lasso_pred = lasso_model.predict(X_test)
lasso_mse = mean_squared_error(y_test, lasso_pred)
print(f"Lasso Regression MSE: {lasso_mse:.4f}")

# Plotting
plt.scatter(y_test, ridge_pred, color='red', label='Ridge Predictions', alpha=0.6)
plt.scatter(y_test, lasso_pred, color='green', label='Lasso Predictions', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2, label="Ideal Prediction")
plt.xlabel("Actual Alcohol")
plt.ylabel("Predicted Alcohol")
plt.title("Ridge vs Lasso Regression on Wine Dataset")
plt.legend()
plt.grid(True)
plt.show()
