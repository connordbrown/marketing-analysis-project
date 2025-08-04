# Marketing Analysis Project

This guide explains how to run the code for my Data Mining project analyzing online retail data.

## System Overview

The project consists of five main components:

1. Data Preparation (`data/excel2csv.py`)
2. Data Cleaning (`data/data_cleaner.py`)
3. Market Basket Analysis (`market_basket/market_basket_analysis.py`)
4. Customer Segmentation (`clustering/cluster_analysis.py`)
5. Time Series Analysis (`time_series/time_series_analysis.py`)

## Dependencies

Install the required dependencies with pip:

```bash
pip install pandas numpy scikit-learn mlxtend seaborn matplotlib
```

## Running the Analysis

### 1. Data Preparation
First, convert the Excel data to CSV and set up Git LFS:

```bash
cd data
python excel2csv.py
```

This script:
- Combines multiple Excel sheets into one CSV file
- Configures Git LFS for handling large files
- Creates `online_retail_II_combined.csv`

### 2. Data Cleaning
Clean the combined dataset:

```bash
python data/data_cleaner.py
```

This script:
- Normalizes prices and quantities
- Handles missing Customer IDs
- Parses dates into programming-friendly format
- Creates `online_retail_II_cleaned.csv`

### 3. Market Basket Analysis
Run the market basket analysis to discover product associations:

```bash
python market_basket/market_basket_analysis.py
```

### 4. Customer Segmentation
Perform RFM analysis and customer clustering:

```bash
python clustering/cluster_analysis.py
```

### 5. Time Series Analysis
Analyze sales trends and seasonality:

```bash
python time_series/time_series_analysis.py
```

## Project Structure

```
marketing-analysis-project/
├── data/
│   ├── excel2csv.py
│   ├── data_cleaner.py
│   ├── online_retail_II.xlsx
│   ├── online_retail_II_combined.csv
│   └── online_retail_II_cleaned.csv
├── market_basket/
│   └── market_basket_analysis.py
├── clustering/
│   └── cluster_analysis.py
├── time_series/
│   └── time_series_analysis.py
└── README.md
```

## Results

The analysis produces:
- Clean, normalized dataset with proper date formatting
- Product association rules with support, confidence, and lift metrics
- Customer segments based on RFM (Recency, Frequency, Monetary) analysis
- Sales forecasting with RMSE and MAPE accuracy metrics
- Seasonal pattern analysis
