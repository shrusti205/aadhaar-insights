import pandas as pd

def clean_aadhaar_data():
    # Read the original data
    df = pd.read_csv('aadhaar_data.csv')
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date to keep the most recent entry for duplicates
    df = df.sort_values('date', ascending=False)
    
    # Remove duplicates, keeping the first occurrence (most recent date due to sorting)
    df_cleaned = df.drop_duplicates(['date', 'state', 'district'], keep='first')
    
    # Sort back to chronological order
    df_cleaned = df_cleaned.sort_values('date')
    
    # Save cleaned data
    df_cleaned.to_csv('aadhaar_data_cleaned.csv', index=False)
    print(f"Cleaned data saved. Original rows: {len(df)}, Cleaned rows: {len(df_cleaned)}")
    return df_cleaned

if __name__ == "__main__":
    clean_aadhaar_data()
