#  AI Flight Delay Predictor  
A full-stack + machine learning system that predicts whether a flight will be delayed using real aviation datasets and ML models.



---

##  Project Status  
**Completed :**
- Dataset download automation (Kaggle API)
- Data cleaning pipeline (train_clean.csv)
- Feature transformation & label creation (is_delayed)
- ML model training (XGBoost)
- API development using FastAPI
- React UI for taking flight inputs and visualizing delay prediction with probability meter



---



## Data Source  
Real flight delay data is downloaded automatically using this Kaggle dataset:

**Dataset:**  
[`patrickzel/flight-delay-and-cancellation-dataset-2019-2023`](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023)

**File:**  
`src/download_kaggle_flights.py`

---

##  Data Cleaning 

The script:


performs:

- Selecting important aviation columns  
- Converting `FL_DATE` to datetime  
- Filling missing values  
- Creating target label `is_delayed`  
- Saving cleaned dataset to `data/processed/train_clean.csv`

---
 Features

- Flight delay probability prediction
- Dynamic UI with probability meter
- Auto-distance filling from route mapping
- Airport + city dropdown selection
- Cancelled/Diverted logic handling
- Smooth UI animations & result scrolling

---

## 🖥️ Backend

Using **FastAPI**:
- `/predict` API endpoint  
- Returns delay probability & prediction  
- Connects with model.pkl  

---

## 🎨 Frontend 

Using **React**:
- Flight form  
- Prediction results  


---
Machine Learning Model

- XGBoost (best performance for tabular prediction)

Output

-Binary target: delayed or not
- Probability score 0–1

