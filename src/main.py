import get_data
import preprocessing
import train_model
import time

def main():
    print("========================================")
    print("Starting MACHINE LEARNING")
    print("========================================")
    
    start_time = time.time()

    
    print("\n[ 1/3] Get Data")
    get_data.ingest_data()
    
    
    print("\n[ 2/3] Preprocessing")
    preprocessing.clean_data()
    
    
    print("\n[ 3/3] Train Model")
    train_model.train()
    
    duration = time.time() - start_time
    print("\n========================================")
    print(f"✅ PIPELINE finish in {duration:.2f} secondes")
    print("========================================")

if __name__ == "__main__":
    main()  