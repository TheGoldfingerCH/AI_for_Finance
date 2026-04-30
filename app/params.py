import os
from pathlib import Path

# DEPLOYMENT
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")

# DATA
BASE_DIR = Path(__file__).resolve().parents[1]
LOCAL_DATA_PATH = os.environ.get(
    "LOCAL_DATA_PATH",
    str(BASE_DIR / "data_folder"),
)