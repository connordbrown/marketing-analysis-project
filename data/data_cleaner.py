import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import subprocess

# Read the combined CSV file
df = pd.read_csv('data/online_retail_II.csv')

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Handle missing Customer IDs by filling with 0
df['Customer ID'] = df['Customer ID'].fillna(0).astype(int)

# Create MinMaxScaler instances for Price and Quantity
price_scaler = MinMaxScaler()
quantity_scaler = MinMaxScaler()

# Reshape the data for scaling (sklearn expects 2D array)
prices = df['Price'].values.reshape(-1, 1)
quantities = df['Quantity'].values.reshape(-1, 1)

# Perform the scaling
normalized_prices = price_scaler.fit_transform(prices)
normalized_quantities = quantity_scaler.fit_transform(quantities)

# Add normalized columns to the DataFrame
df['NormalizedPrice'] = normalized_prices
df['NormalizedQuantity'] = normalized_quantities

# Calculate total value for each transaction
df['TotalValue'] = df['Price'] * df['Quantity']

# Save the cleaned data to a new CSV file
output_file = 'data/online_retail_II_cleaned.csv'
df.to_csv(output_file, index=False)

print()

# Print basic statistics about the cleaned dataset
print("\nDataset Summary:")
print(f"Total number of transactions: {len(df)}")
print(f"Date range: from {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
print(f"Number of unique customers: {df['Customer ID'].nunique()}")
print(f"Percentage of originally missing CustomerIDs: {(df['Customer ID'] == 0).sum() / len(df) * 100:.2f}%")
print()

# Set up Git LFS tracking for the cleaned CSV file
try:
    subprocess.run(['git', 'lfs', 'track', '*.csv'], check=True)
    print("Git LFS tracking configured for cleaned CSV file")
except subprocess.CalledProcessError as e:
    print(f"Error setting up Git LFS: {e}")

print(f"Data cleaning and normalization completed. Cleaned data saved to {output_file}.")
print()