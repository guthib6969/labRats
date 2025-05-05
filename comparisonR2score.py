# Import Libraries
import pandas as pd
import numpy as np

# Load the Datasets
uci_stats = pd.read_csv("uci_diabetes (3).csv")  # Precomputed statistics
pima_stats = pd.read_csv("pima_diabetes (3).csv")  # Precomputed statistics

# Display Summary Statistics
print("Comparison of Univariate Analysis Results:")
print("\nUCI Diabetes Dataset Statistics:\n", uci_stats)
print("\nPima Indians Diabetes Dataset Statistics:\n", pima_stats)

# Compare Regression Model Performance
uci_r2 = 0.78  # Example R² score from Multiple Regression
pima_r2 = 0.72  # Example R² score from Multiple Regression
uci_accuracy = 82.4  # Example Logistic Regression Accuracy
pima_accuracy = 79.1  # Example Logistic Regression Accuracy

print(f"\nLinear Regression R² Scores: UCI - {uci_r2}, Pima - {pima_r2}")
print(f"Logistic Regression Accuracy: UCI - {uci_accuracy}%, Pima - {pima_accuracy}%")
