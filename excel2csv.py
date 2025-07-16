import pandas as pd
import subprocess

# Specify Excel file name
excel_file = 'online_retail_II.xlsx'

# Read each sheet into a separate DataFrame
df_sheet1 = pd.read_excel(excel_file, sheet_name='Year 2009-2010')
df_sheet2 = pd.read_excel(excel_file, sheet_name='Year 2010-2011')

# Combine the DataFrames
# Use ignore_index=True to reset the index of the combined DataFrame
combined_df = pd.concat([df_sheet1, df_sheet2], ignore_index=True)

# Save the combined DataFrame to a CSV file
output_file = 'online_retail_II_combined.csv'
combined_df.to_csv(output_file, index=False, encoding='utf-8')

# Set up Git LFS tracking for both files
try:
    subprocess.run(['git', 'lfs', 'track', '*.xlsx'], check=True)
    subprocess.run(['git', 'lfs', 'track', '*.csv'], check=True)
    print("Git LFS tracking configured for Excel and CSV files")
except subprocess.CalledProcessError as e:
    print(f"Error setting up Git LFS: {e}")

print(f"Sheets combined and saved to {output_file}")