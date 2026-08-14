import kagglehub
from dotenv import load_dotenv
import os
load_dotenv()
print("Token loaded:", bool(os.getenv("KAGGLE_API_TOKEN")))
path = kagglehub.competition_download('filament-segmentation-2026')
print("Path to competition files:", path)