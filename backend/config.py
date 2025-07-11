# backend/config.py
import os

# Get the base directory of the project (one level up from backend/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to the trained machine learning model
MODEL_PATH = os.path.join(BASE_DIR, 'data', 'models', 'fraud_detection_model.joblib')

# You can add other configuration variables here, e.g.,
# API_KEY = os.getenv("API_KEY", "your_default_api_key")
# LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
