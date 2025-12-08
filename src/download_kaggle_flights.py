from kaggle.api.kaggle_api_extended import KaggleApi
import os

api = KaggleApi()
api.authenticate()

dataset = "patrickzel/flight-delay-and-cancellation-dataset-2019-2023"

output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

print("Downloading real flight delay dataset from Kaggle...")
api.dataset_download_files(dataset, path=output_dir, unzip=True)
print("Download complete! Files saved in data/raw/")
