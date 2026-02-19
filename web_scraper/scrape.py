import requests
import pandas as pd
import io

# 1. Target URL
url = "https://cda.gov.lk/web/index.php?option=com_content&view=article&id=22&Itemid=135&lang=en"

# 2. Define Headers (The Disguise)
# This tells the server we are a real browser, not a bot.
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.google.com/"
}

print("Connecting to CDA website with headers...")

try:
    # Pass the headers into the request
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Check for errors
    print("Connection successful!")

    # 3. Parse HTML tables
    print("Extracting tables...")
    # Use pandas to read the HTML directly from the response text
    tables = pd.read_html(io.StringIO(response.text))

    # 4. Filter and Clean
    cleaned_tables = []
    
    print(f"Found {len(tables)} tables. Processing...")

    for i, df in enumerate(tables):
        # Convert all columns to string to avoid errors during search
        df_str = df.astype(str)
        
        # Check if this table looks like auction data (look for specific keywords in the first few rows)
        # We look for "Date" and "Quantity" in the column headers or first row
        is_auction_table = False
        
        # Flatten the first few rows to check for keywords
        first_rows = df_str.head(5).to_string()
        if "Date" in first_rows and "Quantity" in first_rows:
             is_auction_table = True

        if is_auction_table:
            # Clean column names: The first row is often the header in these scraped tables
            # If the current header is just 0, 1, 2, make the first row the header
            if isinstance(df.columns[0], int):
                df.columns = df.iloc[0]
                df = df[1:]
            
            # Standardize columns
            df.columns = [str(c).replace('\r', ' ').replace('\n', ' ').strip() for c in df.columns]
            
            # Keep only relevant columns
            # We explicitly look for the Date and Price columns
            # Note: The website changes column names slightly over years, so we grab by index if needed
            if len(df.columns) >= 4:
                # Rename columns to standard names
                df.columns.values[0] = "Date"
                df.columns.values[-1] = "Avg_Price"
                
                # Filter out garbage rows (headers repeated in middle of table)
                df = df[df["Date"].str.contains("Date", case=False, na=False) == False]
                df = df[df["Date"].notna()]
                
                cleaned_tables.append(df)

    # 5. Combine and Save
    if cleaned_tables:
        final_df = pd.concat(cleaned_tables, ignore_index=True)
        
        # Save to CSV
        output_filename = "srilanka_coconut_auction_data.csv"
        final_df.to_csv(output_filename, index=False)
        print(f"Success! Collected {len(final_df)} rows of data.")
        print(f"Saved to {output_filename}")
        print("-" * 20)
        print(final_df.head())
    else:
        print("Connected, but no auction tables were recognized. The website format might be complex.")

except requests.exceptions.HTTPError as err:
    print(f"HTTP Error: {err}")
except Exception as e:
    print(f"An error occurred: {e}")