import pandas as pd

# Load dataset into a DataFrame
df = pd.read_csv('data.csv')

# Display first and last few rows
print("First 5 rows:\n", df.head())
print("Last 5 rows:\n", df.tail())

# Check data types and general info
df.info()

# Summary statistics
print("Summary statistics:\n", df.describe())

# Handle missing values (only for numeric columns)
df.fillna(df.select_dtypes(include='number').mean(), inplace=True)

# Create a new column (make sure 'existing_column' exists)
if 'existing_column' in df.columns:
    df['new_column'] = df['existing_column'] * 2
    # Create a Series and perform operations
    series = df['existing_column']
    print("Series addition:\n", series + 10)
else:
    print("Column 'existing_column' not found.")

# Filter rows based on conditions (check columns)
if {'existing_column', 'another_column'}.issubset(df.columns):
    filtered_df = df[(df['existing_column'] > 50) & (df['another_column'] < 100)]
    print("Filtered DataFrame:\n", filtered_df)

# Grouping and aggregation (check columns)
if {'category_column', 'numeric_column'}.issubset(df.columns):
    grouped = df.groupby('category_column')['numeric_column'].mean()
    print("Grouped mean:\n", grouped)

    # Sorting
    df_sorted = df.sort_values(by='numeric_column', ascending=False)
    print("Sorted DataFrame:\n", df_sorted)

    # Boolean masking
    masked_df = df[df['numeric_column'] > df['numeric_column'].median()]
    print("Masked DataFrame:\n", masked_df)

    # Compute summary statistics
    print("Total sum:", df['numeric_column'].sum())
    print("Mean:", df['numeric_column'].mean())
    print("Standard Deviation:", df['numeric_column'].std())
else:
    print("Required columns for grouping/sorting/statistics not found.")

# Remove duplicates and drop any remaining missing values
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Create a new DataFrame with selected columns (check if columns exist)
required_columns = ['column1', 'column2']
if set(required_columns).issubset(df.columns):
    subset_df = df[required_columns]
    # Save the new DataFrame to a CSV file
    subset_df.to_csv('filtered_data.csv', index=False)
    print("Filtered data saved to 'filtered_data.csv'.")
else:
    print(f"One or more required columns {required_columns} not found.")
