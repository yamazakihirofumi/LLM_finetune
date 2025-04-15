# This program is used to convert .xslx file to csv for easier process 


import pandas as pd

# Load the Excel file
# Replace 'your_file.xlsx' with your actual filename
excel_file = 'weibo_data.xlsx'
df = pd.read_excel(excel_file)

# Save as CSV
csv_file = excel_file.replace('.xlsx', '.csv')


df.to_csv(csv_file, index=False,quoting=1)
print(f"Successfully converted {excel_file} to {csv_file}")

# Create a simplified version with only username, comment, and class label
simple_df = pd.DataFrame()
simple_df['username'] = df.iloc[:, 0]  # Username is at column 0
simple_df['comment'] = df.iloc[:, 2]   # Comment is at column 2
simple_df['class'] = -1                # Initialize class label as -1

# Save the shorter CSV
simple_csv = 'simple.csv'
simple_df.to_csv(simple_csv, index=False,quoting=1)

print(f"Created simplified dataset at {simple_csv}")