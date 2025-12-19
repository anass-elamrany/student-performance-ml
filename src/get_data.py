import pandas as pd
from ucimlrepo import fetch_ucirepo
import os

def ingest_data():
    print("🚀 [1/3] Ingesting data from UCI Machine Learning Repository...")

    raw_data_dir = 'data/raw'
    os.makedirs(raw_data_dir, exist_ok=True)

    try:
        print("   Downloading from UCI ML Repository (ID: 320)...")
        student_performance = fetch_ucirepo(id=320) 

        # data (as pandas dataframes) 
        X = student_performance.data.features 
        y = student_performance.data.targets 
        
        print("   Combining features and target into a single DataFrame...")
        # On combine tout pour avoir un CSV brut complet
        df = pd.concat([X, y], axis=1)

        output_path = f'{raw_data_dir}/student_performance_full.csv'
        # Important : On utilise le séparateur ; pour éviter les bugs avec les CSV Excel
        df.to_csv(output_path, index=False, sep=';')

        print(f"✅ SUCCÈS : Données brutes sauvegardées dans '{output_path}'")
        print(f"   Dimensions : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    except Exception as e:
        print(f"❌ ERREUR LORS DU TÉLÉCHARGEMENT : \n{e}")

if __name__ == "__main__":
    ingest_data()