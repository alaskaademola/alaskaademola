import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import os

base_path = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(base_path, "..", "data", "ecowas_trade.csv") # Make sure this matches your file name

def run_final_audit():
    try:
        # 1. Load data with encoding fix
        df = pd.read_csv(FILE_PATH, encoding='latin1', on_bad_lines='skip')
        
        # 2. FILTER: Ignore $0 values and very small quantities
        # This removes the "noise" seen in your last screenshot
        df = df[(df['primaryValue'] > 100) & (df['qty'] > 0)]
        
        # 3. FEATURE ENGINEERING: Calculate Unit Price
        # Over-invoicing is hidden here! (Price / Quantity)
        df['unit_price'] = df['primaryValue'] / df['qty']

        # 4. AI SCAN: Focus on Value and Unit Price
        model = IsolationForest(contamination=0.1, random_state=42)
        df['risk_score'] = model.fit_predict(df[['primaryValue', 'unit_price']])

        red_flags = df[df['risk_score'] == -1]

        print(f"✅ Final Audit Complete!")
        print(f"Cleaned Transactions Scanned: {len(df)}")
        print(f"⚠️ High-Value Red Flags: {len(red_flags)}")
        
        # 5. Save the Presentation Chart
        plt.figure(figsize=(12, 7))
        plt.scatter(df['qty'], df['primaryValue'], c=df['risk_score'], cmap='hot', alpha=0.5)
        plt.title('ECOWAS Trade Integrity: Value vs Quantity Anomalies')
        plt.xlabel('Quantity')
        plt.ylabel('Trade Value ($)')
        plt.savefig(os.path.join(base_path, "final_trade_report.png"))
        
        # 6. Show the "Real" Suspicious Trades
        if not red_flags.empty:
            print("\n--- Top High-Risk Trades (Excluding $0) ---")
            print(red_flags[['reporterDesc', 'cmdDesc', 'primaryValue', 'unit_price']].head(10))

    except Exception as e:
        print(f"Technical Error: {e}")

if __name__ == "__main__":
    run_final_audit()