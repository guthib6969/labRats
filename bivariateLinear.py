import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the Datasets
uci_diabetes = pd.read_csv("uci_diabetes (3).csv")
pima_diabetes = pd.read_csv("pima_diabetes (3).csv")

# Display first few rows
print("UCI Diabetes Dataset Sample:\n", uci_diabetes.head())
print("\nPima Indians Diabetes Dataset Sample:\n", pima_diabetes.head())

# ----------- Linear Regression Function ----------- #
def linear_regression_analysis(df, x_column, y_column):
    # Drop rows with missing values in relevant columns
    df = df[[x_column, y_column]].dropna()

    X = df[[x_column]]
    Y = df[y_column]

    model = LinearRegression()
    model.fit(X, Y)
    Y_pred = model.predict(X)
    r2 = r2_score(Y, Y_pred)

    print(f"\nLinear Regression (Predicting {y_column} using {x_column}):")
    print(f"R² Score: {r2:.4f}")

    # Plot
    plt.figure(figsize=(6, 4))
    plt.scatter(X, Y, color='blue', label='Actual Data')
    plt.plot(X, Y_pred, color='red', linewidth=2, label='Regression Line')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f"Linear Regression: {x_column} vs. {y_column}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Apply Linear Regression
linear_regression_analysis(uci_diabetes, "Glucose", "BMI")
linear_regression_analysis(pima_diabetes, "Glucose", "BMI")

# ----------- Logistic Regression Function ----------- #
def logistic_regression_analysis(df, features, target):
    # Drop rows with missing values in relevant columns
    df = df[features + [target]].dropna()

    X = df[features]
    Y = df[target]

    # Optional: Scale features for better convergence
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)

    print(f"\nLogistic Regression (Predicting {target} using {features}):")
    print(f"Accuracy Score: {accuracy:.4f}")

# Select features and target
features = ["Glucose", "BloodPressure", "BMI", "Age"]
target = "Outcome"  # Make sure this column exists in both datasets

# Apply Logistic Regression
logistic_regression_analysis(uci_diabetes, features, target)
logistic_regression_analysis(pima_diabetes, features, target)
