import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


BIG = BASE / "data" / "processed" / "train_clean_city.csv"


OUT = BASE / "data" / "processed" / "metadata_small.csv"

print("Loading big CSV...")
df = pd.read_csv(
    BIG,
    usecols=[
        "AIRLINE",
        "ORIGIN",
        "ORIGIN_CITY",
        "DEST",
        "DEST_CITY",
        "DISTANCE",
    ],
)

routes = (
    df
    .drop_duplicates(subset=["ORIGIN", "DEST"])
    .reset_index(drop=True)
)

print("Original rows:", len(df))
print("Unique routes:", len(routes))

routes.to_csv(OUT, index=False)
print("Saved small metadata to:", OUT)
