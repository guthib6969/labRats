import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Load Dataset
uci_diabetes = pd.read_csv("uci_diabetes (3).csv")

# Plot Normal Curves for Glucose and BMI
plt.figure(figsize=(12, 5))

# Normal Curve for Glucose
plt.subplot(1, 2, 1)
sns.histplot(uci_diabetes["Glucose"], kde=True, stat="density", linewidth=0)
x_glucose = np.linspace(uci_diabetes["Glucose"].min(), uci_diabetes["Glucose"].max(), 100)
plt.plot(x_glucose, norm.pdf(x_glucose, uci_diabetes["Glucose"].mean(), uci_diabetes["Glucose"].std()), 'r')
plt.title("Normal Curve - Glucose")

# Normal Curve for BMI
plt.subplot(1, 2, 2)
sns.histplot(uci_diabetes["BMI"], kde=True, stat="density", linewidth=0)
x_bmi = np.linspace(uci_diabetes["BMI"].min(), uci_diabetes["BMI"].max(), 100)
plt.plot(x_bmi, norm.pdf(x_bmi, uci_diabetes["BMI"].mean(), uci_diabetes["BMI"].std()), 'r')
plt.title("Normal Curve - BMI")

plt.show()
