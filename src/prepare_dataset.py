import pandas as pd
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]

RAW_PATH = BASE / "data" / "raw" / "flights_sample_3m.csv"  
PROCESSED_PATH = BASE / "data" / "processed" / "train_clean_city.csv"

print("Loading raw dataset...")
df = pd.read_csv(RAW_PATH)

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
    raise Exception(f" Columns missing in raw file: {missing}\n"
                    "Your raw CSV must include city columns")

df = df[required_cols]


print(" Creating target label is_delayed...")
df["is_delayed"] = df["ARR_DELAY"].apply(lambda x: 1 if x > 15 else 0)


numeric_cols = ["DEP_DELAY", "DISTANCE"]
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())


df["CANCELLED"] = df["CANCELLED"].fillna(0)
df["DIVERTED"] = df["DIVERTED"].fillna(0)


df = df.dropna(subset=["AIRLINE", "ORIGIN_CITY", "DEST_CITY"])

print(" Saving processed dataset:", PROCESSED_PATH)
df.to_csv(PROCESSED_PATH, index=False)

print(" Dataset prepared successfully!")
print("You can now train your model on train_clean_city.csv")

