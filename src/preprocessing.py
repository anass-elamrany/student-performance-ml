import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

def clean_data():
    print("🧹 [2/3] Starting data cleaning & feature engineering...")

    raw_path = 'data/raw/student_performance_full.csv'
    if not os.path.exists(raw_path):
        print(f"❌ Error: The file {raw_path} does not exist. Run get_data.py first.")
        return

    df = pd.read_csv(raw_path, sep=';')
    df = df.dropna()

    # --- MISE À JOUR : CRÉATION DES 3 CLASSES (Cahier des charges) ---
    # 0 = À risque (< 10)
    # 1 = Moyenne performance (10 <= note < 14)
    # 2 = Bon performeur (>= 14)
    
    def create_category(grade):
        if grade < 10:
            return 0  
        elif 10 <= grade < 14:
            return 1  
        else:
            return 2  

    
    df['grade_category'] = df['G3'].apply(create_category)
    
    print(f"   -> Distribution des classes (0=Risque, 1=Moyen, 2=Bon) :")
    print(df['grade_category'].value_counts().sort_index())
    # ---------------------------------------------------------------

    # Encodage des variables texte (Sex, School, etc.)
    le = LabelEncoder()
    object_cols = df.select_dtypes(include=['object']).columns
    
    print(f"   -> Encoding columns: {list(object_cols)}")
    for col in object_cols:
        df[col] = le.fit_transform(df[col])

    # Sauvegarde
    processed_dir = 'data/processed'
    os.makedirs(processed_dir, exist_ok=True)
    
    output_path = f'{processed_dir}/student_data_cleaned.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✅ SUCCESS: Cleaned data saved to '{output_path}'")

if __name__ == "__main__":
    clean_data()