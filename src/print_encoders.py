import joblib
from pathlib import Path

Base=Path(__file__).resolve().parents[1]
ENC_PATH=Base/"models"/"encoders.pkl"

encoders=joblib.load(ENC_PATH)
print("AIRLINE VALUES:")
print(encoders["AIRLINE"].classes_)
print("ORIGIN VALUES:")
print(encoders["ORIGIN"].classes_)
print("DEST VALUES:")
print(encoders["DEST"].classes_)