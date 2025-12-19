# 🎓 Student Performance Prediction

A complete Machine Learning pipeline to predict student academic performance .

![Python](https://img.shields.io/badge/Python-3.9-blue?style=for-the-badge&logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Container-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit_Learn-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

## 📄 About the Project

This project is an **End-to-End Machine Learning Pipeline** designed to predict student academic performance and identify at-risk students early in the year.

Unlike simple prediction tools, this system performs a **"Battle Royale" comparison** between multiple algorithms (Random Forest, SVM, KNN, Gradient Boosting) to automatically select the most accurate model for the dataset.

**The system predicts two key metrics:**
1.  **Academic Status (Classification):** Classifies students into 3 categories: *At Risk*, *Average*, *Good Performer*.
2.  **Final Grade (Regression):** Predicts the exact final score (0-20 scale).

Data is sourced automatically from the [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance).

**The pipeline performs 3 main steps:**
1.  **Ingestion:** Automatically downloads fresh data.
2.  **Processing:** Cleans and formats the data.
3.  **Training:** Trains Random Forest models to predict results.

## 🛠️ Tech Stack

* **Language:** Python 3.9
* **Libraries:** Pandas, Numpy
* **Machine Learning:** Scikit-Learn (RandomForest, SVM, KNN, GradientBoosting, LinearRegression)
* **Visualization:** Matplotlib, Seaborn (in Notebooks)
* **Containerization:** Docker

## 📂 Project Structure

```text
student-performance-ml/
│
├── data/
│   ├── raw/               # Fresh data downloaded from UCI
│   └── processed/         # Cleaned data & Final Predictions CSV
│
├── src/
│   ├── get_data.py        # ETL: Ingestion script
│   ├── preprocessing.py   # ETL: Cleaning & Feature Engineering
│   ├── train_model.py     # ML: Training & Model Comparison (Battle Royale)
│   └── main.py            # Pipeline Orchestrator
│
├── models/                # Trained models (.pkl) & Performance Reports
├── notebooks/             # Jupyter Notebooks for Deep Analysis & Charts
├── Dockerfile             # Production-ready Docker image
└── requirements.txt       # Python dependencies
```

## 🚀 How to Run

### Option 1: Using Docker (Recommended)
This method isolates the environment. You don't need to install Python libraries manually.

1.  **Build the Docker image:**
    ```bash
    docker build -t student-ml .
    ```

2.  **Run the pipeline:**
    *We use `-v` to map your local folder to the container so the generated CSV files are saved on your computer, not lost inside the container.*
    ```bash
    # Linux / Mac
    docker run -v $(pwd)/data:/app/data student-ml
    
    # Windows (Command Prompt)
    docker run -v %cd%/data:/app/data student-ml
    
    # Windows (PowerShell)
    docker run -v ${PWD}/data:/app/data student-ml
    ```

### Option 2: Running Locally (Python)
If you prefer running it on your machine:

```
# 1. Install dependencies
pip install -r requirements.txt
```

```
# 2. Run the main script
python src/main.py
```

## 📊 Model Performance

The system evaluates multiple models and selects the champion. Current benchmarks on the test set:

### 🏆 Classification (Risk Detection)

*Goal: Predict if a student is "At Risk" (0-9), "Average" (10-13), or "Good" (14+).*

| Model | Accuracy | Status |
| :--- | :--- | :--- |
| **Random Forest** | **86.15%** | 🏆 **Winner** |
| Gradient Boosting | 85.38% | 🥈 Runner-up |
| Logistic Regression| 85.38% | |
| SVM (Linear) | 84.62% | |
| KNN (k=5) | 82.31% | |

### 📈 Regression (Grade Prediction)

*Goal: Predict the exact final grade (G3).*

* **Best Model:** Random Forest Regressor
* **Mean Absolute Error (MAE):** **0.75** points.
* *Interpretation: The model's prediction is accurate within a margin of less than 1 point out of 20.*

## License

[MIT](https://choosealicense.com/licenses/mit/)
