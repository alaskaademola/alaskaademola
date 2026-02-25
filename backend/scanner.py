import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import os
import re

base_path = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(base_path, "..", "data", "ecowas_trade.csv")

def clean_to_float(value):
    """Forcefully convert any string with symbols/commas into a clean number."""
    if pd.isna(value): return 0.0
    # Remove everything except numbers and the decimal point
    clean_val = re.sub(r'[^0-9.]', '', str(value))
    try:
        return float(clean_val)
    except:
        return 0.0

def run_ecowas_sentinel():
    try:
        print("--- 🛡️ ECO-Trade Sentinel: Initializing Deep Clean ---")
        df = pd.read_csv(FILE_PATH, encoding='latin1', on_bad_lines='skip')

        # 1. IDENTIFY COLUMNS (Matching your UN Comtrade Search)
        val_col = next((c for c in df.columns if 'Value' in c or 'primaryValue' in c), None)
        qty_col = next((c for c in df.columns if 'Weight' in c or 'qty' in c), None)

        if not val_col or not qty_col:
            print(f"❌ Error: Could not find Value or Quantity. Found: {df.columns.tolist()[:3]}")
            return

        # 2. THE SLEDGEHAMMER CLEAN
        # This converts "2,500.00 US$" into 2500.00
        df[val_col] = df[val_col].apply(clean_to_float)
        df[qty_col] = df[qty_col].apply(clean_to_float)

        # 3. FILTER & ANALYZE
        # Keep only rows that actually have data
        df = df[(df[val_col] > 0) & (df[qty_col] > 0)].copy()
        
        if df.empty:
            print("❌ Error: No valid numeric data found. Check if your CSV has numbers in the Value column!")
            return

        df['unit_price'] = df[val_col] / df[qty_col]

        # 4. AI AUDIT (Isolation Forest)
        model = IsolationForest(contamination=0.1, random_state=42)
        df['is_anomaly'] = model.fit_predict(df[[val_col, 'unit_price']])
        red_flags = df[df['is_anomaly'] == -1]

        print(f"✅ CLEAN SUCCESSFUL! Scanned: {len(df)} transactions.")
        print(f"⚠️ Red Flags Found: {len(red_flags)}")
        
        # 5. SAVE EVIDENCE CHART
        plt.figure(figsize=(10, 6))
        plt.scatter(df[qty_col], df[val_col], c=df['is_anomaly'], cmap='coolwarm', alpha=0.5)
        plt.title('Trade Integrity Audit: Value vs Quantity')
        plt.xlabel('Quantity / Weight')
        plt.ylabel('Trade Value (US$)')
        plt.savefig(os.path.join(base_path, "research_evidence.png"))
        print("📈 Chart saved as 'research_evidence.png'!")

    except Exception as e:
        print(f"Technical Error: {e}")

if __name__ == "__main__":
    run_ecowas_sentinel()