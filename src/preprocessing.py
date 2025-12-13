import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

def clean_data():
    print("Starting data cleaning...")
    

    raw_path = 'data/raw/student_performance_full.csv'
    if not os.path.exists(raw_path):
        print(f"Error: The file {raw_path} does not exist. Run get_data.py first.")
        return

    df = pd.read_csv(raw_path, sep=';')
    

    df = df.dropna()

    
    
    df['success_status'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
    print(f"   -> Distribution of successes (1) / failures (0) : \n{df['success_status'].value_counts()}")


    le = LabelEncoder()
    object_cols = df.select_dtypes(include=['object']).columns
    
    print(f"   -> Encoding columns: {list(object_cols)}")
    
    for col in object_cols:
        df[col] = le.fit_transform(df[col])

    
    processed_dir = 'data/processed'
    os.makedirs(processed_dir, exist_ok=True)
    
    output_path = f'{processed_dir}/student_data_cleaned.csv'
    df.to_csv(output_path, index=False)
    
    print(f"SUCCESS: Cleaned data saved to '{output_path}'")
    print(f"Ready for Machine Learning!")

if __name__ == "__main__":
    clean_data()