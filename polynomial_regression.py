import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the Wine dataset
wine = load_wine()
X = wine.data[:, [0]]  # Feature: Alcohol
y = wine.data[:, 1]    # Target: Malic Acid
y = y.reshape(-1, 1)

# Transform features to polynomial features
degree = 3
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X)

# Train the polynomial regression model
model = LinearRegression()
model.fit(X_poly, y)

# Predict values
y_pred = model.predict(X_poly)

# Evaluate model performance
mse = mean_squared_error(y, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# Plot the results
plt.scatter(X, y, label="Actual Data", alpha=0.7)
# Sort X for a smooth curve
sorted_idx = X[:, 0].argsort()
plt.plot(X[sorted_idx], y_pred[sorted_idx], color='red', label="Polynomial Fit", linewidth=2)
plt.xlabel("Alcohol")
plt.ylabel("Malic Acid")
plt.title("Polynomial Regression on Wine Data")
plt.legend()
plt.show()
