import pandas as pd

# Read local CSV file
try:
    text_df = pd.read_csv('Google_data (2b.c1).csv')
    print("Loaded text_df")
except Exception as e:
    print("Error loading text_df:", e)

# Read Excel file (Sheet1)
try:
    excel_df = pd.read_excel('data (2c2).xlsx', sheet_name='Sheet1')
    print("Loaded excel_df")
except Exception as e:
    print("Error loading excel_df:", e)

# Read CSV from web
try:
    web_df = pd.read_csv('https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv')
    print("Loaded web_df")
except Exception as e:
    print("Error loading web_df:", e)

# Display heads (if loaded successfully)
print("\nText DF Head:\n", text_df.head() if 'text_df' in locals() else "text_df not loaded")
print("\nExcel DF Head:\n", excel_df.head() if 'excel_df' in locals() else "excel_df not loaded")
print("\nWeb DF Head:\n", web_df.head() if 'web_df' in locals() else "web_df not loaded")

# Handle missing values (only if dataframes are loaded)
if 'text_df' in locals():
    text_df.fillna(method='ffill', inplace=True)

if 'excel_df' in locals():
    excel_df.fillna(method='bfill', inplace=True)

if 'web_df' in locals():
    web_df.dropna(inplace=True)

# Save processed data
if 'text_df' in locals():
    text_df.to_csv('processed_text.csv', index=False)
    print("Saved processed_text.csv")

if 'excel_df' in locals():
    excel_df.to_excel('processed_excel.xlsx', index=False)
    print("Saved processed_excel.xlsx")
