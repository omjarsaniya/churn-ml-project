"""
Central path definitions.

Every path is resolved relative to this file's location, not the current
working directory — so `python src/train.py` works the same whether you run
it from the project root, from inside src/, or from Docker's WORKDIR.
"""

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_DATA_PATH = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PIPELINE_PATH = MODELS_DIR / "pipeline.pkl"

# Make sure output directories exist so scripts never fail on a missing folder
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
