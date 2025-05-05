import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('iris_dataset(2d).csv')

# Display basic information and summary statistics
print("Basic Information:")
df.info()

print("\nSummary Statistics:")
print(df.describe())

# Univariate analysis - species count
print("\nSpecies Count:")
print(df['species'].value_counts())

# Histogram for feature distributions
df.hist(figsize=(10, 8), edgecolor='black', grid=False)
plt.suptitle('Distribution of Iris Features', fontsize=16)
plt.tight_layout()
plt.show()

# Boxplot for Sepal Length by Species
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='species', y='sepal length (cm)', palette='Set2')
plt.title('Sepal Length Distribution by Species')
plt.ylabel('Sepal Length (cm)')
plt.xlabel('Species')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Pairplot to analyze feature relationships by species
sns.pairplot(df, hue='species', diag_kind='hist', palette='Set1')
plt.suptitle('Pairwise Feature Relationships', y=1.02)
plt.show()
