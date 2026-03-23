# 🧠 Customer Churn Prediction

A machine learning project to predict customer churn using the Telco dataset, 
with an interactive Streamlit dashboard.

## 🚀 Live Demo
[Click here to view the app](#) ← replace with your Streamlit link

## 📊 Project Overview
- Cleaned and preprocessed Telco Customer Churn dataset
- Handled class imbalance using **SMOTE**
- Trained and compared **6 ML models**
- Best model: **Random Forest (RandomizedSearchCV)** → 79.87% accuracy

## 🛠️ Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, XGBoost, Imbalanced-learn
- Streamlit, Plotly

## 📁 Files
| File | Description |
|------|-------------|
| `app.py` | Streamlit dashboard |
| `project.ipynb` | Model training notebook |
| `Random_Forest_Tuned.pkl` | Saved best model |
| `encoders.pkl` | Label encoders |
| `datachurn.csv` | Dataset |
| `requirement.txt` | Dependencies |

## ▶️ Run Locally
pip install -r requirement.txt
streamlit run app.py

## 📬 Contact
Connect with me on LinkedIn: https://www.linkedin.com/in/siddharth-tiwari-85689a243/