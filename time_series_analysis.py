import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

# Read the cleaned data
df = pd.read_csv('online_retail_II_cleaned.csv')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Create daily sales data
daily_sales = df.groupby('InvoiceDate').agg({
    'TotalValue': 'sum'
}).reset_index()

# Add time-based features
daily_sales['Year'] = daily_sales['InvoiceDate'].dt.year
daily_sales['Month'] = daily_sales['InvoiceDate'].dt.month
daily_sales['DayOfWeek'] = daily_sales['InvoiceDate'].dt.dayofweek
daily_sales['DayOfMonth'] = daily_sales['InvoiceDate'].dt.day

# Create training and testing sets (80-20 split)
train_size = int(len(daily_sales) * 0.8)
train = daily_sales[:train_size]
test = daily_sales[train_size:]

# Prepare features for regression
features = ['Year', 'Month', 'DayOfWeek', 'DayOfMonth']
X_train = train[features]
y_train = train['TotalValue']
X_test = test[features]
y_test = test['TotalValue']

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
train_pred = model.fit(X_train, y_train).predict(X_train)
test_pred = model.predict(X_test)

# Calculate metrics
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
train_mape = calculate_mape(y_train, train_pred)
test_mape = calculate_mape(y_test, test_pred)

# Print results
print("\nModel Performance Metrics:")
print(f"Training RMSE: £{train_rmse:.2f}")
print(f"Testing RMSE: £{test_rmse:.2f}")
print(f"Training MAPE: {train_mape:.2f}%")
print(f"Testing MAPE: {test_mape:.2f}%")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': np.abs(model.coef_)
})
print("\nFeature Importance:")
print(feature_importance.sort_values('Importance', ascending=False))

# Visualizations
plt.figure(figsize=(15, 10))

# Actual vs Predicted (Training)
plt.subplot(2, 1, 1)
plt.plot(train.index, y_train, label='Actual', alpha=0.7)
plt.plot(train.index, train_pred, label='Predicted', alpha=0.7)
plt.title('Training Set: Actual vs Predicted Sales')
plt.xlabel('Time')
plt.ylabel('Sales Value')
plt.legend()

# Actual vs Predicted (Testing)
plt.subplot(2, 1, 2)
plt.plot(test.index, y_test, label='Actual', alpha=0.7)
plt.plot(test.index, test_pred, label='Predicted', alpha=0.7)
plt.title('Testing Set: Actual vs Predicted Sales')
plt.xlabel('Time')
plt.ylabel('Sales Value')
plt.legend()

plt.tight_layout()
plt.savefig('time_series_predictions.png')
plt.close()

# Seasonal Analysis
plt.figure(figsize=(15, 5))

# Monthly seasonality
monthly_avg = df.groupby(df['InvoiceDate'].dt.month)['TotalValue'].mean()
plt.subplot(1, 2, 1)
monthly_avg.plot(kind='bar')
plt.title('Average Sales by Month')
plt.xlabel('Month')
plt.ylabel('Average Sales')

# Day of week seasonality
daily_avg = df.groupby(df['InvoiceDate'].dt.dayofweek)['TotalValue'].mean()
plt.subplot(1, 2, 2)
daily_avg.plot(kind='bar')
plt.title('Average Sales by Day of Week')
plt.xlabel('Day of Week (0=Monday)')
plt.ylabel('Average Sales')

plt.tight_layout()
plt.savefig('seasonal_patterns.png')
plt.close()