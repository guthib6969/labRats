import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# Import Datasets
uci_diabetes = pd.read_csv("/mnt/data/uci_diabetes.csv")
pima_diabetes = pd.read_csv("/mnt/data/pima_diabetes.csv")

# Display Dataset Samples
print("UCI Diabetes Dataset Sample:")
print(uci_diabetes.head())

print("\nPima Indians Diabetes Dataset Sample:")
print(pima_diabetes.head())

# Define Relevant Numerical Columns
numerical_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", 
                     "BMI", "DiabetesPedigreeFunction", "Age"]

# Univariate Analysis Function
def univariate_analysis(df, columns):
    stats = {}
    for col in columns:
        if col in df.columns:
            clean_col = df[col].dropna()  # Drop NaNs for safe statistics
            stats[col] = {
                "Mean": np.mean(clean_col),
                "Median": np.median(clean_col),
                "Mode": clean_col.mode()[0] if not clean_col.mode().empty else np.nan,
                "Variance": np.var(clean_col, ddof=1),
                "Standard Deviation": np.std(clean_col, ddof=1),
                "Skewness": skew(clean_col),
                "Kurtosis": kurtosis(clean_col)
            }
        else:
            stats[col] = "Column Not Found"
    return pd.DataFrame(stats).T

# Perform Univariate Analysis
uci_stats = univariate_analysis(uci_diabetes, numerical_columns)
pima_stats = univariate_analysis(pima_diabetes, numerical_columns)

# Display Results
print("\nUCI Diabetes Dataset Statistics:")
print(uci_stats)

print("\nPima Indians Diabetes Dataset Statistics:")
print(pima_stats)
