import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path
import joblib

Base = Path(__file__).resolve().parents[1]

raw_data = Base / "data" / "raw" / "flights_sample_3m.csv"
processed_data = Base / "data" / "processed" / "train_clean_city.csv"
model_file = Base / "models" / "model.pkl"
encoder_file = Base / "models" / "encoders.pkl"

print("Loading raw dataset...")
df = pd.read_csv(raw_data)

required_cols = [
    "FL_DATE","AIRLINE","ORIGIN","ORIGIN_CITY",
    "DEST","DEST_CITY","DEP_DELAY",
    "ARR_DELAY","CANCELLED","DIVERTED","DISTANCE"
]

df = df[required_cols].copy()
df["is_delayed"] = (df["ARR_DELAY"] > 0).astype(int)

df.to_csv(processed_data, index=False)
print("✔ Saved processed dataset:", processed_data)

label_cols = ["AIRLINE", "ORIGIN", "DEST", "ORIGIN_CITY", "DEST_CITY"]
encoders = {}

for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

joblib.dump(encoders, encoder_file)
print("✔ Encoders saved:", encoder_file)

X = df[[ "AIRLINE","ORIGIN","ORIGIN_CITY",
         "DEST","DEST_CITY","DEP_DELAY",
         "CANCELLED","DIVERTED","DISTANCE" ]]

y = df["is_delayed"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(" Training XGBoost model...")
model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✔ Model Accuracy: {acc:.3f}")

joblib.dump(model, model_file)
print("✔ Model saved:", model_file)
