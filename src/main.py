import get_data
import preprocessing
import train_model
import time

def main():
    print("========================================")
    print("🎓 STUDENT PERFORMANCE PIPELINE START")
    print("========================================")
    
    start_time = time.time()

    # Etape 1 : Téléchargement
    get_data.ingest_data()
    
    # Etape 2 : Nettoyage et Création des 3 Classes
    preprocessing.clean_data()
    
    # Etape 3 : Entraînement et Sauvegarde des Modèles
    train_model.train()
    
    duration = time.time() - start_time
    print("\n========================================")
    print(f"✅ PIPELINE COMPLETED in {duration:.2f} seconds")
    print("========================================")

if __name__ == "__main__":
    main()