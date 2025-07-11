import pandas as pd

# specify Excel file name
excel_file = 'online_retail_II.xlsx'

# read each sheet into a separate DataFrame
df_sheet1 = pd.read_excel(excel_file, sheet_name='Year 2009-2010')
df_sheet2 = pd.read_excel(excel_file, sheet_name='Year 2010-2011')

# combine the DataFrames
# use ignore_index=True to reset the index of the combined DataFrame
combined_df = pd.concat([df_sheet1, df_sheet2], ignore_index=True)

# save the combined DataFrame to a CSV file
combined_df.to_csv('online_retail_II_combined.csv', index=False, encoding='utf-8')

print("Sheets combined and saved to online_retail_II_combined.csv")