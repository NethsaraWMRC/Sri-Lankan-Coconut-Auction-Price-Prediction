"""
Simple Data Preprocessing Methods for Coconut Auction Dataset
"""

import pandas as pd
import numpy as np
import re


def load_data(file_path):
    """Load CSV data"""
    return pd.read_csv(file_path)


def clean_dates(df):
    """Fix and standardize date formats"""
    def fix_date(date_str):
        if pd.isna(date_str):
            return None
        
        date_str = str(date_str).strip()
        
        # Fix missing slash: "28/112024" -> "28/11/2024"
        if re.match(r'\d{2}/\d{6}', date_str):
            date_str = date_str[:3] + date_str[3:5] + '/' + date_str[5:]
        
        # Fix dot: "18.05/2023" -> "18/05/2023"
        date_str = date_str.replace('.', '/')
        
        try:
            return pd.to_datetime(date_str, format='%d/%m/%Y')
        except:
            try:
                return pd.to_datetime(date_str, dayfirst=True)
            except:
                return None
    
    df['Date'] = df['Date'].apply(fix_date)
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df


def clean_numeric_values(df):
    """Fix numeric format issues"""
    def fix_number(value):
        if pd.isna(value):
            return None
        
        value_str = str(value).strip()
        value_str = value_str.replace(',', '')
        value_str = re.sub(r'\.\.', '.', value_str)
        value_str = re.sub(r'[^\d.-]', '', value_str)
        
        try:
            return float(value_str)
        except:
            return None
    
    numeric_cols = ['Offered_Quantity', 'Sold_Quantity', 'Avg_Price_Rs./1000']
    
    for col in numeric_cols:
        df[col] = df[col].apply(fix_number)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def handle_missing_values(df):
    """Remove rows with missing values"""
    return df.dropna().reset_index(drop=True)


def remove_duplicates(df):
    """Remove duplicate rows"""
    return df.drop_duplicates().reset_index(drop=True)


def normalize_data(df, columns):
    """Normalize specified columns to 0-1 range"""
    df_normalized = df.copy()
    
    for col in columns:
        min_val = df[col].min()
        max_val = df[col].max()
        df_normalized[col + '_normalized'] = (df[col] - min_val) / (max_val - min_val)
    
    return df_normalized


def standardize_data(df, columns):
    """Standardize specified columns (mean=0, std=1)"""
    df_standardized = df.copy()
    
    for col in columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        df_standardized[col + '_standardized'] = (df[col] - mean_val) / std_val
    
    return df_standardized


def remove_outliers(df, column, method='iqr'):
    """Remove outliers using IQR method"""
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return df[(df[column] >= lower) & (df[column] <= upper)]
    return df


def add_features(df):
    """Add useful derived features"""
    df['Sales_Ratio'] = df['Sold_Quantity'] / df['Offered_Quantity']
    df['Unsold_Quantity'] = df['Offered_Quantity'] - df['Sold_Quantity']
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    
    return df


def load_exchange_rate_data(exchange_rate_file):
    """Load and clean exchange rate data"""
    print("   Loading exchange rate data...")
    df_rate = pd.read_csv(exchange_rate_file)
    
    # Clean column names (remove quotes and spaces)
    df_rate.columns = df_rate.columns.str.strip().str.replace('"', '').str.replace("'", "")
    
    # Drop empty rows
    df_rate = df_rate.dropna(how='all').reset_index(drop=True)
    
    # Convert date to datetime
    df_rate['Date'] = pd.to_datetime(df_rate['Date'])
    
    # Keep only USD data and relevant columns
    df_rate = df_rate[df_rate['Currency'] == 'USD'].copy()
    df_rate = df_rate[['Date', 'Buy Rate', 'Sell Rate']]
    
    # Rename columns
    df_rate.rename(columns={
        'Buy Rate': 'USD_Buy_Rate',
        'Sell Rate': 'USD_Sell_Rate'
    }, inplace=True)
    
    # Add average rate
    df_rate['USD_Avg_Rate'] = (df_rate['USD_Buy_Rate'] + df_rate['USD_Sell_Rate']) / 2
    
    print(f"   Loaded {len(df_rate)} exchange rate records")
    return df_rate


def merge_with_exchange_rates(df_coconut, exchange_rate_file):
    """Merge coconut data with exchange rate data using Nearest Backward Match"""
    print("   Merging with exchange rate data...")
    
    # Load exchange rate data
    df_rate = load_exchange_rate_data(exchange_rate_file)
    
    # IMPORTANT: Both dataframes MUST be sorted by Date for merge_asof
    df_coconut = df_coconut.sort_values('Date')
    df_rate = df_rate.sort_values('Date')
    
    # Use merge_asof to find the NEAREST previous exchange rate
    # direction='backward' means: "If today's rate is missing, give me yesterday's rate."
    df_merged = pd.merge_asof(
        df_coconut,
        df_rate,
        on='Date',
        direction='backward'
    )
    
    print(f"   Merged successfully! Added exchange rate columns with time-series alignment.")
    return df_merged


def preprocess_pipeline(input_file, output_file, exchange_rate_file=None):
    """Complete preprocessing pipeline"""
    print("Starting preprocessing...")
    
    # Load data
    print("1. Loading data...")
    df = load_data(input_file)
    print(f"   Loaded {len(df)} rows")
    
    # Clean dates
    print("2. Cleaning dates...")
    df = clean_dates(df)
    print(f"   After cleaning: {len(df)} rows")
    
    # Clean numeric values
    print("3. Cleaning numeric values...")
    df = clean_numeric_values(df)
    
    # Handle missing values
    print("4. Handling missing values...")
    df = handle_missing_values(df)
    print(f"   After handling missing: {len(df)} rows")
    
    # Remove duplicates
    print("5. Removing duplicates...")
    df = remove_duplicates(df)
    print(f"   After removing duplicates: {len(df)} rows")
    
    # Add features
    print("6. Adding derived features...")
    df = add_features(df)
    
    # Merge with exchange rates if file provided
    if exchange_rate_file:
        print("7. Merging with exchange rate data...")
        df = merge_with_exchange_rates(df, exchange_rate_file)
    
    # Save cleaned data
    print(f"{'8' if exchange_rate_file else '7'}. Saving to {output_file}...")
    df.to_csv(output_file, index=False)
    
    print("\n✓ Preprocessing complete!")
    print(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    return df
