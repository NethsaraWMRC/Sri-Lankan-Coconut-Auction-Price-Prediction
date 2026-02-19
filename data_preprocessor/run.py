"""
Run Data Preprocessing
Simple script to preprocess coconut auction data
"""

import os
from preprocess import preprocess_pipeline

# Get the directory paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Input and output file paths
INPUT_FILE = os.path.join(parent_dir, 'srilanka_coconut_auction_data.csv')
OUTPUT_FILE = os.path.join(parent_dir, 'srilanka_coconut_auction_data_cleaned.csv')
EXCHANGE_RATE_FILE = os.path.join(parent_dir, 'buy_and_sell_exchange_rates_usd_lkr.csv')

if __name__ == "__main__":
    # Run the preprocessing pipeline with exchange rate data
    cleaned_data = preprocess_pipeline(INPUT_FILE, OUTPUT_FILE, EXCHANGE_RATE_FILE)
    
    # Display summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Input file:  {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Total records: {len(cleaned_data)}")
    print(f"Total columns: {len(cleaned_data.columns)}")
    print(f"Date range: {cleaned_data['Date'].min()} to {cleaned_data['Date'].max()}")
    print(f"Average sales ratio: {cleaned_data['Sales_Ratio'].mean():.2%}")
    
    # Show if exchange rates were added
    if 'USD_Avg_Rate' in cleaned_data.columns:
        print(f"Exchange rate range: {cleaned_data['USD_Avg_Rate'].min():.2f} - {cleaned_data['USD_Avg_Rate'].max():.2f}")
    
    print("\nColumns:", list(cleaned_data.columns))
    print("="*50)
