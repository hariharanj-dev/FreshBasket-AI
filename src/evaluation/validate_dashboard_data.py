"""FreshBasket AI - Dashboard Data Validation Runner (Evaluation Module Alias).

Allows running validation via either:
    python -m src.data.validate_dashboard_data
or:
    python -m src.evaluation.validate_dashboard_data
"""

import sys
from src.data.validate_dashboard_data import main, validate_dashboard_datasets

if __name__ == "__main__":
    main()
