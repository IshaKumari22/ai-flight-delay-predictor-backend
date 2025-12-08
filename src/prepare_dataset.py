import pandas as pd
from pathlib import Path

# 🔹 Define project base folder
BASE = Path(__file__).resolve().parents[1]

RAW_PATH = BASE / "data" / "raw" / "flights_sample_3m.csv"   # change file name if needed
PROCESSED_PATH = BASE / "data" / "processed" / "train_clean_city.csv"

print("📌 Loading raw dataset...")
df = pd.read_csv(RAW_PATH)

# 🔹 Keep required columns only
required_cols = [
    "FL_DATE",
    "AIRLINE",
    "ORIGIN",
    "ORIGIN_CITY",
    "DEST",
    "DEST_CITY",
    "DEP_DELAY",
    "ARR_DELAY",
    "CANCELLED",
    "DIVERTED",
    "DISTANCE"
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise Exception(f"❌ Columns missing in raw file: {missing}\n"
                    "➡ Your raw CSV must include city columns")

df = df[required_cols]

# 🔹 Convert ARR_DELAY into binary target
print("📌 Creating target label is_delayed...")
df["is_delayed"] = df["ARR_DELAY"].apply(lambda x: 1 if x > 15 else 0)

# 🔹 Fill missing numeric values
numeric_cols = ["DEP_DELAY", "DISTANCE"]
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# 🔹 Fill missing flags
df["CANCELLED"] = df["CANCELLED"].fillna(0)
df["DIVERTED"] = df["DIVERTED"].fillna(0)

# 🔹 Drop rows where city names missing
df = df.dropna(subset=["AIRLINE", "ORIGIN_CITY", "DEST_CITY"])

print("📌 Saving processed dataset:", PROCESSED_PATH)
df.to_csv(PROCESSED_PATH, index=False)

print("✅ Dataset prepared successfully!")
print("➡ You can now train your model on train_clean_city.csv")

