import pandas as pd
from sklearn.ensemble import IsolationForest
import os
import re

base_path = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(base_path, "..", "data", "ecowas_trade.csv")

def clean_to_float(value):
    if pd.isna(value): return 0.0
    clean_val = re.sub(r'[^0-9.]', '', str(value))
    try:
        return float(clean_val)
    except:
        return 0.0

def run_final_red_flag_audit():
    try:
        print("--- 🛡️ ECO-Trade Sentinel: Analyzing Anomalies ---")
        df = pd.read_csv(FILE_PATH, encoding='latin1', on_bad_lines='skip')

        # 1. Column Mapping (Finding Value and Weight)
        val_col = next((c for c in df.columns if 'Value' in c or 'primaryValue' in c), None)
        qty_col = next((c for c in df.columns if 'Weight' in c or 'qty' in c), None)

        # 2. Deep Cleaning
        df[val_col] = df[val_col].apply(clean_to_float)
        df[qty_col] = df[qty_col].apply(clean_to_float)
        df = df[(df[val_col] > 0) & (df[qty_col] > 0)].copy()
        
        # 3. Feature Engineering
        df['unit_price'] = df[val_col] / df[qty_col]

        # 4. AI Detection
        model = IsolationForest(contamination=0.1, random_state=42)
        df['is_anomaly'] = model.fit_predict(df[[val_col, 'unit_price']])
        
        # Capture the Red Flags
        red_flags_df = df[df['is_anomaly'] == -1].copy()

        print(f"✅ ANALYSIS COMPLETE")
        print(f"Total Transactions Scanned: {len(df)}")
        print(f"⚠️ High-Risk Anomalies Found: {len(red_flags_df)}")

        # 5. PRINT THE 22 FLAGS (The part you want to see!)
        if not red_flags_df.empty:
            print("\n" + "="*80)
            print(f"{'REPORTER':<20} | {'COMMODITY':<15} | {'VALUE ($)':<15} | {'UNIT PRICE'}")
            print("-" * 80)
            
            # Sort by Unit Price so the "worst" ones are at the top
            top_flags = red_flags_df.sort_values(by='unit_price', ascending=False)
            
            for index, row in top_flags.iterrows():
                reporter = str(row.get('reporterDesc', 'Unknown'))[:18]
                cmd = str(row.get('cmdCode', 'N/A'))[:13]
                val = f"{row[val_col]:,.2f}"
                u_price = f"{row['unit_price']:.4f}"
                print(f"{reporter:<20} | {cmd:<15} | {val:<15} | {u_price}")
            print("="*80)
        
    except Exception as e:
        print(f"Technical Error: {e}")

if __name__ == "__main__":
    run_final_red_flag_audit()