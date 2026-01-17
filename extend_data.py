import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_extended_data():
    # Load the existing cleaned data
    df = pd.read_csv('aadhaar_data_cleaned.csv')
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Get the last date in the existing data
    last_date = df['date'].max()
    
    # If we already have data through 2024, no need to extend
    if last_date.year >= 2024 and last_date.month == 12:
        print("Data already extends through December 2024.")
        return
    
    # Get unique combinations of state, district, update_type, age_group, gender, center_type
    unique_combinations = df.drop_duplicates(
        ['state', 'district', 'update_type', 'age_group', 'gender', 'center_type']
    )
    
    # Generate dates from the month after the last date to December 2024
    dates = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        end=pd.Timestamp('2024-12-01'),
        freq='MS'
    )
    
    extended_data = []
    
    for date in dates:
        # Get data from the same month in the previous year for seasonality
        prev_year_data = df[df['date'].dt.month == date.month].copy()
        
        # If no data for this month, use the most recent data
        if prev_year_data.empty:
            prev_year_data = df[df['date'] == df['date'].max()].copy()
        
        # For each unique combination, generate new data
        for _, row in unique_combinations.iterrows():
            # Get similar records from previous year
            similar = prev_year_data[
                (prev_year_data['state'] == row['state']) &
                (prev_year_data['district'] == row['district']) &
                (prev_year_data['update_type'] == row['update_type']) &
                (prev_year_data['age_group'] == row['age_group']) &
                (prev_year_data['gender'] == row['gender']) &
                (prev_year_data['center_type'] == row['center_type'])
            ]
            
            # If no similar record, use the most recent for this combination
            if similar.empty:
                similar = prev_year_data[
                    (prev_year_data['state'] == row['state']) &
                    (prev_year_data['district'] == row['district'])
                ].iloc[0:1]
            
            # If still no data, use the row as is
            if similar.empty:
                base_row = row
            else:
                base_row = similar.iloc[0]
            
            # Generate new data with some variation
            growth_rate = 1 + np.random.uniform(0.02, 0.05)  # 2-5% growth
            
            new_enrolments = int(base_row['enrolments'] * growth_rate)
            new_updates = int(base_row['updates'] * growth_rate)
            
            # Add some randomness
            new_enrolments += int(np.random.normal(0, new_enrolments * 0.1))
            new_updates += int(np.random.normal(0, new_updates * 0.1))
            
            # Ensure values are positive
            new_enrolments = max(100, new_enrolments)
            new_updates = max(10, new_updates)
            
            # Create new row
            new_row = {
                'date': date.strftime('%Y-%m-%d'),
                'state': row['state'],
                'district': row['district'],
                'enrolments': new_enrolments,
                'updates': new_updates,
                'update_type': row['update_type'],
                'age_group': row['age_group'],
                'gender': row['gender'],
                'center_type': row['center_type']
            }
            
            extended_data.append(new_row)
    
    # Create DataFrame from extended data
    extended_df = pd.DataFrame(extended_data)
    
    # Combine with original data
    combined_df = pd.concat([df, extended_df], ignore_index=True)
    
    # Sort by date and other columns
    combined_df = combined_df.sort_values(['date', 'state', 'district', 'update_type'])
    
    # Save to new file
    combined_df.to_csv('aadhaar_data_extended.csv', index=False)
    print(f"Extended data saved to 'aadhaar_data_extended.csv' with {len(combined_df)} rows.")
    print(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")

if __name__ == "__main__":
    generate_extended_data()
