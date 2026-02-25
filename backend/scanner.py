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

def run_greedy_audit():
    try:
        print("--- 🛡️ ECO-Trade Sentinel: Greedy Forensic Audit ---")
        df = pd.read_csv(FILE_PATH, encoding='latin1', on_bad_lines='skip')

        # 1. Identify and Clean Columns
        val_col = next((c for c in df.columns if 'Value' in c or 'primaryValue' in c), None)
        qty_col = next((c for c in df.columns if 'Weight' in c or 'qty' in c), None)
        df[val_col] = df[val_col].apply(clean_to_float)
        df[qty_col] = df[qty_col].apply(clean_to_float)
        df = df[(df[val_col] > 0) & (df[qty_col] > 0)].copy()
        
        # 2. AI Anomaly Detection
        df['unit_price'] = df[val_col] / df[qty_col]
        model = IsolationForest(contamination=0.1, random_state=42)
        df['is_anomaly'] = model.fit_predict(df[[val_col, 'unit_price']])
        red_flags_df = df[df['is_anomaly'] == -1].copy()

        print(f"✅ CLEAN SUCCESSFUL! Scanned: {len(df)} transactions.")
        print(f"⚠️ Red Flags Found: {len(red_flags_df)}")

        # 3. GREEDY STEP: Sector-Risk Ranking
        # This uses 'cmdCode' or 'cmdDesc' to see where the most flags are
        cmd_col = 'cmdCode' if 'cmdCode' in df.columns else 'cmdDesc'
        print("\n--- 📊 SECTOR RISK RANKING ---")
        sector_risk = red_flags_df.groupby(cmd_col).agg({
            'unit_price': 'mean',
            'is_anomaly': 'count'
        }).rename(columns={'is_anomaly': 'flag_count'}).sort_values(by='flag_count', ascending=False)
        print(sector_risk)

        # 4. GREEDY STEP: Detecting "Mirror Trades"
        # Flags exact duplicates in Value and Weight
        duplicates = df[df.duplicated(subset=[val_col, qty_col], keep=False)]
        print(f"\n⚠️ Mirror Trade Alert: Found {len(duplicates)} transactions with identical Value/Weight.")
        if not duplicates.empty:
            print(duplicates[[val_col, qty_col, cmd_col]].head(10))
        
    except Exception as e:
        print(f"Technical Error: {e}")

if __name__ == "__main__":
    run_greedy_audit()