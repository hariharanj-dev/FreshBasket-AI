"""
Configuration settings and directory paths for FreshBasket AI.

This module centralizes path definitions and high-level project parameters
to keep the codebase modular, clean, and easy to maintain.
"""

from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FORECASTS_DIR = DATA_DIR / "forecasts"

# Models and artifacts
MODELS_DIR = BASE_DIR / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"

# Reports and figures
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Forecasting Configuration
FORECAST_HORIZON_WEEKS = 13
RANDOM_STATE = 42
