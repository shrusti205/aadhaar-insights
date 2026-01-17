import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime

def load_data():
    try:
        df = pd.read_csv("aadhaar_data.csv")
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m', errors='coerce')
        df = df.dropna(subset=['date'])
        return df
    except Exception as e:
        print(f"Error loading main data: {e}")
        return pd.DataFrame()

def load_detailed_data():
    detailed_data = {}
    def load_folder(folder_name):
        path = os.path.join(os.getcwd(), folder_name, "*.csv")
        files = glob.glob(path)
        if not files: return pd.DataFrame()
        dfs = []
        for f in files:
            try: dfs.append(pd.read_csv(f))
            except Exception as e: print(f"Error {f}: {e}")
        if not dfs: return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)
    
    try:
        enrol_df = load_folder("api_data_aadhar_enrolment")
        if not enrol_df.empty:
            enrol_df['date'] = pd.to_datetime(enrol_df['date'], format='%d-%m-%Y', errors='coerce')
            enrol_df.rename(columns={'age_18_greater': 'age_18_plus'}, inplace=True)
            detailed_data['enrolment'] = enrol_df
    except Exception as e:
        print(f"Error loading detailed: {e}")
    return detailed_data

print("--- Testing Data Loading ---")
df = load_data()
if df.empty:
    print("Main DF is empty!")
else:
    print(f"Main DF loaded: {len(df)} rows")

detailed = load_detailed_data()
print(f"Detailed keys: {detailed.keys()}")

# Filter logic simulation
filtered_df = df.copy()
print(f"Filtered DF: {len(filtered_df)} rows")

print("--- Testing Tab 3 Logic ---")
try:
    dist_agg = filtered_df.groupby(['state', 'district'])[['enrolments', 'updates']].sum().reset_index()
    print(f"Dist Agg: {len(dist_agg)} rows")
    corr = filtered_df[['enrolments', 'updates']].corr()
    print(f"Correlation: {corr}")
    
    # Anomaly detection
    filtered_df['z_score'] = (filtered_df['updates'] - filtered_df['updates'].mean()) / filtered_df['updates'].std()
    anomalies = filtered_df[filtered_df['z_score'].abs() > 3]
    print(f"Anomalies: {len(anomalies)}")
except Exception as e:
    print(f"ERROR in Tab 3 Logic: {e}")

print("--- Testing Tab 4 Logic ---")
try:
    monthly_trend = filtered_df.groupby('date')['updates'].sum()
    print(f"Monthly Trend: {len(monthly_trend)} points")
    if len(monthly_trend) > 3:
        last_3_avg = monthly_trend.rolling(window=3).mean().iloc[-1]
        print(f"Last 3 avg: {last_3_avg}")
    
    total_recs = len(filtered_df)
    high_activity = len(filtered_df[filtered_df['updates'] > filtered_df['updates'].median()])
    inclusion_score = min(100, (high_activity / total_recs) * 100 + 50) if total_recs > 0 else 0
    print(f"Inclusion Score: {inclusion_score}")
except Exception as e:
    print(f"ERROR in Tab 4 Logic: {e}")

print("--- Finished Diagnostic ---")
