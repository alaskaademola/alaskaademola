import pandas as pd
from sklearn.ensemble import IsolationForest
import os

# Finds your file in the data folder
base_path = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(base_path, "..", "data", "ecowas_trade.csv")

def run_ai_audit():
    try:
        # Load data with the fix for that 'utf-8' error
        df = pd.read_csv(FILE_PATH, encoding='latin1', on_bad_lines='skip')
        
        # Use the columns we found in your terminal: primaryValue and qty
        # We fill '0' for any missing numbers so the AI doesn't crash
        analysis_data = df[['primaryValue', 'qty']].fillna(0)

        # Initialize the AI (Looking for the 5% most 'weird' trades)
        model = IsolationForest(contamination=0.05, random_state=42)
        df['is_anomaly'] = model.fit_predict(analysis_data)

        # -1 = Red Flag (Anomaly)
        red_flags = df[df['is_anomaly'] == -1]

        print(f"✅ Scan Complete!")
        print(f"Total Trades Analyzed: {len(df)}")
        print(f"⚠️ High-Risk Red Flags Detected: {len(red_flags)}")
        
        if not red_flags.empty:
            print("\n--- Top Suspicious Transactions ---")
            print(red_flags[['reporterDesc', 'partnerDesc', 'primaryValue', 'qty']].head())

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_ai_audit()