# ====================================
# TASK 3 - Linear Regression
# ====================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# -----------------------------------
# Load Dataset
# -----------------------------------

housing = fetch_california_housing()

df = pd.DataFrame(housing.data, columns=housing.feature_names)

df['Price'] = housing.target

print(df.head())

# -----------------------------------
# Features and Target
# -----------------------------------

X = df[['MedInc']]
y = df['Price']

# -----------------------------------
# Train-Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------
# Train Model
# -----------------------------------

model = LinearRegression()

model.fit(X_train, y_train)

# -----------------------------------
# Predictions
# -----------------------------------

y_pred = model.predict(X_test)

# -----------------------------------
# Evaluation Metrics
# -----------------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("\nMAE:", mae)

print("MSE:", mse)

print("R2 Score:", r2)

# -----------------------------------
# Coefficients
# -----------------------------------

print("\nCoefficient:", model.coef_[0])

print("Intercept:", model.intercept_)

# -----------------------------------
# Regression Line Plot
# -----------------------------------

plt.figure(figsize=(8,5))

plt.scatter(X_test, y_test)

plt.plot(X_test, y_pred)

plt.xlabel("Median Income")

plt.ylabel("House Price")

plt.title("Linear Regression")

plt.show()

print("\nTask 3 Completed Successfully!")