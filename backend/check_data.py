import pandas as pd
import os

# 1. This finds your file correctly
base_path = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(base_path, "..", "data", "ecowas_trade.csv")

def check_my_data():
    try:
        print(f"Opening file at: {FILE_PATH}")
        
        # 2. 'latin1' encoding fixes the 'utf-8' error you saw in your terminal
        df = pd.read_csv(FILE_PATH, encoding='latin1', on_bad_lines='skip')
        
        print("\n--- SUCCESS! Your CSV Columns are: ---")
        print(df.columns.tolist())
        
        print("\n--- First 3 rows of data: ---")
        print(df.head(3))
        
    except Exception as e:
        # This will catch any remaining errors and tell us why
        print(f"Technical Error: {e}")

if __name__ == "__main__":
    check_my_data()