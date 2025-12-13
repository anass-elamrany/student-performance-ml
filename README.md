# 🎓 Student Performance Prediction

A complete Machine Learning pipeline to predict student academic performance (Pass/Fail and Final Grade) ..

## 📄 About the Project
This project automates the process of predicting student success based on demographic and social data (from the [UCI Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)). It is designed to help educators identify at-risk students early.

**The pipeline performs 3 main steps:**
1.  **Ingestion:** Automatically downloads fresh data.
2.  **Processing:** Cleans and formats the data.
3.  **Training:** Trains Random Forest models to predict results.

## 🛠️ Tech Stack

* **Language:** Python 3.9
* **Libraries:** Pandas, Scikit-Learn, Numpy
* **Container:** Docker


## 🚀 How to Run

Option 1: Using Docker (Recommended)
You don't need to install Python. Just run these two commands:

```bash
# 1. Build the image
docker build -t student-ml .

# 2. Run the pipeline
docker run -v $(pwd)/data:/app/data student-ml
```

Option 2: Running Locally (Python)
If you prefer running it on your machine:
```
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the main script
python src/main.py
```
## 📊 Results
The models were trained on the UCI Student Performance dataset.

* **Classification (Pass/Fail):** 92% Accuracy

* **Regression (Predict Grade):** Error margin of ±0.75 points (out of 20)

## License

[MIT](https://choosealicense.com/licenses/mit/)