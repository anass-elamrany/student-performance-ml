import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report, confusion_matrix
import joblib
import os
import time

def train():
    print("========================================")
    print("🤖 [3/3] TRAIN MODEL")
    print("========================================")
    
    #Chargement des données
    data_path = 'data/processed/student_data_cleaned.csv'
    if not os.path.exists(data_path):
        print("❌ Error: Data not found. Run preprocessing.py first.")
        return
        
    df = pd.read_csv(data_path)
    print(f"Données chargées : {df.shape[0]} étudiants")
    
    # Séparation Features / Target
    # On enlève les réponses (G3 et la catégorie)
    X = df.drop(['G3', 'grade_category'], axis=1)
    y_class = df['grade_category'] 
    y_reg = df['G3']               
    
    # Split Data (80% Train, 20% Test)
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X, y_class, y_reg, test_size=0.2, random_state=42
    )
    
    # ==========================================
    # COMPARAISON CLASSIFICATION
    # (Objectif : Prédire la catégorie Bon/Moyen/Risque)
    # ==========================================
    print("\n COMPARATIF CLASSIFICATION (Accuracy)")
    print(f"{'Modèle':<25} | {'Score':<10} | {'Note'}")
    print("-" * 55)

    classifiers = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM (Support Vector)": SVC(kernel='linear', random_state=42),  
        "KNN (k-Nearest)": KNeighborsClassifier(n_neighbors=5),         
        "GradientBoosting 🚀": GradientBoostingClassifier(random_state=42), 
        "LogisticRegression": LogisticRegression(max_iter=1000)         
    }

    best_clf_name = ""
    best_clf_score = 0
    best_clf_model = None

    for name, model in classifiers.items():
        model.fit(X_train, y_class_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_class_test, pred)
        
        # Petit commentaire auto
        note = "Top!" if acc > 0.85 else "Moyen"
        print(f"{name:<25} | {acc:.2%}    | {note}")
        
        if acc > best_clf_score:
            best_clf_score = acc
            best_clf_model = model
            best_clf_name = name

    print("-" * 55)
    print(f"🏆 VAINQUEUR CLASSIF : {best_clf_name} (Précision : {best_clf_score:.2%})")

    # ==========================================
    # COMPARAISON RÉGRESSION
    # (Objectif : Prédire la note exacte sur 20)
    # ==========================================
    print("\n  COMPARATIF RÉGRESSION (Erreur Moyenne MAE)")
    print(f"{'Modèle':<25} | {'Erreur':<10} | {'Note'}")
    print("-" * 55)
    
    regressors = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "LinearRegression": LinearRegression(),                         
        "KNN Regressor": KNeighborsRegressor(n_neighbors=5),            
        "GradientBoosting 🚀": GradientBoostingRegressor(random_state=42) 
    }

    best_reg_name = ""
    best_reg_score = float('inf') 
    best_reg_model = None

    for name, model in regressors.items():
        model.fit(X_train, y_reg_train)
        pred = model.predict(X_test)
        mae = mean_absolute_error(y_reg_test, pred)
        
        note = "Précis" if mae < 1.0 else "Moyen"
        print(f"{name:<25} | +/- {mae:.2f}   | {note}")
        
        if mae < best_reg_score:
            best_reg_score = mae
            best_reg_model = model
            best_reg_name = name

    print("-" * 55)
    print(f"🏆 VAINQUEUR REGRESSION : {best_reg_name} (Erreur Moyenne : {best_reg_score:.2f} pts)")

    # ==========================================
    #  SAUVEGARDE & PRODUCTION
    # ==========================================
    print("\n💾 Génération des fichiers finaux...")
    
    # On utilise les CHAMPIONS pour générer les données finales
    df['Prediction_Category'] = best_clf_model.predict(X)
    df['Prediction_Grade'] = best_reg_model.predict(X)
    
    # Système d'alerte basé sur la classification championne
    def generate_alert(row):
        cat = row['Prediction_Category']
        if cat == 0:
            return '🔴 ALERTE: Risque Élevé'
        elif cat == 1:
            return '🟠 Surveillance: Moyen'
        else:
            return '🟢 Performance: Bon'

    df['Alert_Status'] = df.apply(generate_alert, axis=1)
    
    
    df['Model_Classif_Used'] = best_clf_name
    df['Model_Reg_Used'] = best_reg_name

    output_path = 'data/processed/predictions_final.csv'
    df.to_csv(output_path, index=False)
    
    # Sauvegarde des modèles (.pkl)
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_clf_model, 'models/model_classification.pkl')
    joblib.dump(best_reg_model, 'models/model_regression.pkl')
    
    # Génération du rapport texte pour ton PFE
    report_path = 'models/rapport_performance.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("RAPPORT AUTOMATIQUE DE PERFORMANCE ML\n")
        f.write("=====================================\n\n")
        f.write(f"Date du test : {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("CHAMPION CLASSIFICATION (Prédiction du Niveau):\n")
        f.write(f"- Modèle : {best_clf_name}\n")
        f.write(f"- Accuracy : {best_clf_score:.2%}\n")
        f.write("- Explication : Ce modèle est le meilleur pour distinguer les bons élèves des élèves à risque.\n\n")
        f.write("CHAMPION REGRESSION (Prédiction de la Note):\n")
        f.write(f"- Modèle : {best_reg_name}\n")
        f.write(f"- Erreur Moyenne (MAE) : {best_reg_score:.2f} points\n")
        f.write("- Explication : En moyenne, ce modèle se trompe de moins de 1 point sur la note finale.\n")

    print(f"✅ Terminé ! Rapport généré : {report_path}")
    print(f"✅ Modèles sauvegardés dans 'models/' (Prêts pour Django)")

if __name__ == "__main__":
    train()