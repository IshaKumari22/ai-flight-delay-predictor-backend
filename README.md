#  AI Flight Delay Predictor  
A full-stack + machine learning project that predicts whether a flight will be delayed based on real aviation data, weather, distance, airline, and more.

This project is currently in **Phase 1: Data Engineering & Cleaning**.

---

##  Project Status  
**Completed so far:**
- Project setup & folder structure  
- Kaggle API configuration  
- Automated real dataset downloading  
- Data cleaning & preprocessing pipeline for ML  

**Upcoming next steps:**
- Feature engineering  
- ML model training (XGBoost)  
- FastAPI backend  
- React dashboard frontend  

---

## 🗂️ Project Structure
ai-flight-delay-predictor/
│
├── data/
│ ├── raw/ # Raw datasets downloaded from Kaggle (ignored in git)
│ └── processed/ # Cleaned datasets for training (ignored in git)
│
├── src/
│ ├── data_prep.py # Data cleaning pipeline
│ ├── download_kaggle_flights.py # Real dataset download script
│ └── train_model.py # (Will be added next)
│
├── models/
│ └── model.pkl # Trained ML model (ignored in git)
│
├── requirements.txt
├── README.md
└── .gitignore


---

## 📥 Data Source  
Real flight delay data is downloaded automatically using this Kaggle dataset:

**Dataset:**  
[`patrickzel/flight-delay-and-cancellation-dataset-2019-2023`](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023)

**File:**  
`s
rc/download_kaggle_flights.py`

---

## 🧹 Data Cleaning (Current Stage)

The script:


performs:

- Selecting important aviation columns  
- Converting `FL_DATE` to datetime  
- Filling missing values  
- Creating target label `is_delayed`  
- Saving cleaned dataset to `data/processed/train_clean.csv`

---

## 🧠 Upcoming (Model Training)

Next, we will:

- Train XGBoost model  
- Save model as `model.pkl`  
- Test accuracy, ROC-AUC  
- Prepare model input format  

---

## 🖥️ Backend (Coming Soon)

Using **FastAPI**:
- `/predict` API endpoint  
- Returns delay probability & prediction  
- Connects with model.pkl  

---

## 🎨 Frontend (Coming Soon)

Using **React**:
- Flight form  
- Prediction results  
- Charts / analytics  
- Airport route visualisation  

---


