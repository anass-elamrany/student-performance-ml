import pandas as pd
from ucimlrepo import fetch_ucirepo
import os

def ingest_data():
    print("Ingesting data from UCI Machine Learning Repository...")

    raw_data_dir = 'data/raw'
    os.makedirs(raw_data_dir, exist_ok=True)

    try:
        print("Downloading from UCI ML Repository...")
        student_performance = fetch_ucirepo(id=320) 

        # data (as pandas dataframes) 
        X = student_performance.data.features 
        y = student_performance.data.targets 
        
        print("combining features and target into a single DataFrame...")
        df = pd.concat([X, y], axis=1)

        output_path = f'{raw_data_dir}/student_performance_full.csv'
        df.to_csv(output_path, index=False , sep=';')

        print(f"SUCCÈS : Données sauvegardées dans '{output_path}'")
        print(f" Dimensions : {df.shape[0]} lignes, {df.shape[1]} colonnes")

        print("Aperçu des données :")
        print(df.head())

    except Exception as e:
            print(f" ERREUR  : \n{e}")

if __name__ == "__main__":
    ingest_data()   