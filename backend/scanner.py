import pandas as pd
from sklearn.ensemble import IsolationForest

# 1. This tells the computer where to find your trade file
# Change "your_file_name.csv" to the actual name of the file you downloaded
FILE_PATH = "data/your_file_name.csv" 

def find_corruption_risk():
    try:
        # Load the data
        df = pd.read_csv(FILE_PATH)
        
        # We look at 'TradeValue' and 'Qty' to find weird patterns
        # Note: Make sure these column names match your CSV headers!
        data_to_check = df[['TradeValue', 'Qty']]

        # The AI: Isolation Forest 'isolates' the weird trades
        model = IsolationForest(contamination=0.05) 
        df['risk_level'] = model.fit_predict(data_to_check)

        # -1 means the AI found a Red Flag (anomaly)
        red_flags = df[df['risk_level'] == -1]
        
        print(f"--- ECOWAS RISK REPORT ---")
        print(f"Total Transactions Scanned: {len(df)}")
        print(f"High-Risk Red Flags Found: {len(red_flags)}")
        print(red_flags.head()) # Shows the top 5 suspicious trades
        
    except Exception as e:
        print(f"Error: {e}. Make sure the file name and columns are correct!")

if __name__ == "__main__":
    find_corruption_risk()