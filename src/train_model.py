import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
import joblib
import os

def train():
    print("Starting model training...")
    
    # 1. Load cleaned data
    data_path = 'data/processed/student_data_cleaned.csv'
    if not os.path.exists(data_path):
        print("❌ Error: Data not found. Run preprocessing.py first.")
        return
        
    df = pd.read_csv(data_path)
    
    # 2. Separation of Features (X) and Targets (y)
    # Remove answers (G3 and statut_reussite) from input data
    X = df.drop(['G3', 'statut_reussite'], axis=1)
    
    # Target 1: Classification (0 = Failure, 1 = Success)
    y_class = df['statut_reussite']
    
    # Target 2: Regression (The exact grade G3)
    y_reg = df['G3']
    
    # 3. Train/Test split (80% training, 20% test)
    # Note: We use the same split for both tasks for consistency
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X, y_class, y_reg, test_size=0.2, random_state=42
    )
    
    # --- TASK A: CLASSIFICATION (Risk of failure) ---
    print("\n🔹 Training Classification model (Random Forest)...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_class_train)
    
    # Evaluation
    y_class_pred = clf.predict(X_test)
    acc = accuracy_score(y_class_test, y_class_pred)
    print(f"   -> Model Accuracy: {acc:.2%}")
    # Display details (Recall, Precision per class)
    print("   -> Detailed Report:")
    print(classification_report(y_class_test, y_class_pred))

    # --- TASK B: REGRESSION (Grade Prediction) ---
    print("\n🔹 Training Regression model...")
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, y_reg_train)
    
    # Evaluation
    y_reg_pred = reg.predict(X_test)
    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    print(f"   -> Mean Absolute Error (MAE): {mae:.2f} points")
    print(f"      (The model has an average error of {mae:.2f} on the final grade)")

    # 4. Save Results for POWER BI
    print("\n💾 Generating Power BI file...")
    
    # Take entire dataset and add predictions
    df['Prediction_Risk'] = clf.predict(X)  # 0 or 1
    df['Prediction_Grade'] = reg.predict(X)  # Estimated grade
    
    # Add readable "Alert" column
    df['Alert_Type'] = df.apply(lambda row: 'CRITICAL' if row['Prediction_Risk'] == 0 else 'Normal', axis=1)

    output_path = 'data/processed/predictions_final.csv'
    df.to_csv(output_path, index=False)
    
    # 5. Save models (if we want to reuse them later in an API)
    os.makedirs('models', exist_ok=True)
    joblib.dump(clf, 'models/model_classification.pkl')
    joblib.dump(reg, 'models/model_regression.pkl')
    
    print(f"✅ Done! File generated: {output_path}")

if __name__ == "__main__":
    train()