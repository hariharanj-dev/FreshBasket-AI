"""
FreshBasket AI - Demo Inventory Data Generator
==============================================
Generates baseline development inventory balances and supplier lead times for
all 1,782 store-family time series (54 stores x 33 product families).

IMPORTANT DISCLAIMER:
This dataset contains synthetic DEMO inventory balances for development,
prototyping, and dashboard demonstration. It does not represent actual physical
store stock levels from the retailer. The dashboard allows users to upload
their own real-world inventory CSV files.
"""

from pathlib import Path
import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
INVENTORY_DIR = WORKSPACE_ROOT / "data" / "inventory"
FORECAST_FILE = WORKSPACE_ROOT / "data" / "dashboard" / "forecast_data.csv"

# Category supplier and lead-time mapping
FAMILY_SUPPLIERS = {
    "AUTOMOTIVE": ("Ecuador Motor Parts Ltd.", 14),
    "BABY CARE": ("Andina Infant Care S.A.", 8),
    "BEAUTY": ("Quito Cosmetics & Personal", 10),
    "BEVERAGES": ("Embotelladora Andina S.A.", 5),
    "BOOKS": ("Editorial Panamericana", 18),
    "BREAD/BAKERY": ("Panaderia Nacional Central", 3),
    "CELEBRATION": ("Fiesta & Articulos S.A.", 12),
    "CLEANING": ("Soluciones de Limpieza Quimica", 7),
    "DAIRY": ("Lacteos Cotopaxi C.A.", 3),
    "DELI": ("Embutidos Pichincha S.A.", 4),
    "EGGS": ("Granja Avicola San Jose", 3),
    "FROZEN FOODS": ("Congelados del Pacifico", 6),
    "GROCERY I": ("Distribuidora Central de Abarrotes", 5),
    "GROCERY II": ("Importadora Alimentos Andinos", 6),
    "HARDWARE": ("Ferreteria Industrial del Norte", 15),
    "HOME AND KITCHEN I": ("Hogar & Menaje Ecuador", 14),
    "HOME AND KITCHEN II": ("Electrodomesticos Andinos", 14),
    "HOME APPLIANCES": ("Tecnologia Domestica S.A.", 21),
    "HOME CARE": ("Cuidado del Hogar Continental", 8),
    "LADIESWEAR": ("Textiles & Confecciones Quito", 21),
    "LAWN AND GARDEN": ("Jardineria & Campo S.A.", 14),
    "LINGERIE": ("Textiles & Confecciones Quito", 21),
    "LIQUOR,WINE,BEER": ("Cerveceria & Licores Nacional", 7),
    "MAGAZINES": ("Distribuidora Prensa Andina", 14),
    "MEATS": ("Carnes del Valle C.A.", 2),
    "PERSONAL CARE": ("Higiene & Bienestar S.A.", 7),
    "PET SUPPLIES": ("Nutricion Animal Andina", 9),
    "PLAYERS AND ELECTRONICS": ("Audio & Video Import S.A.", 21),
    "POULTRY": ("Avicola del Sol C.A.", 2),
    "PREPARED FOODS": ("Cocina Lista Industrial", 2),
    "PRODUCE": ("Agricola Sierra y Costa S.A.", 2),
    "SCHOOL AND OFFICE SUPPLIES": ("Papeleria & Utiles del Norte", 15),
    "SEAFOOD": ("Pescaderia Mar del Sur", 2),
}


def generate_demo_inventory_data():
    """Generates inventory.csv and supplier_lead_times.csv."""
    INVENTORY_DIR.mkdir(parents=True, exist_ok=True)

    # Read forecast Week 1 to calibrate realistic stock balances
    df_fc = pd.read_csv(FORECAST_FILE)
    w1_fc = df_fc[df_fc["forecast_horizon_week"] == 1].copy()

    records_inventory = []
    records_suppliers = []

    np.random.seed(42)  # Deterministic seed for reproducible testing

    for _, row in w1_fc.iterrows():
        store_nbr = int(row["store_nbr"])
        family = str(row["family"])
        w1_demand = float(row["lightgbm_forecast"])

        supplier_name, base_lead_time = FAMILY_SUPPLIERS.get(family, ("Proveedor General S.A.", 7))

        # Add small deterministic jitter to lead times by store to simulate geography
        lead_time_days = max(1, base_lead_time + (store_nbr % 3) - 1)

        # Estimate expected daily demand
        daily_demand = max(1.0, w1_demand / 7.0)
        expected_lt_demand = daily_demand * lead_time_days

        # Safety stock: 2 to 4 days of sales buffer
        safety_buffer_days = max(1.0, min(5.0, lead_time_days * 0.4))
        safety_stock = round(daily_demand * safety_buffer_days, 1)

        reorder_point = expected_lt_demand + safety_stock

        # Create realistic risk tiers:
        # ~18% HIGH risk (current stock < lead time demand)
        # ~30% MEDIUM risk (lead time demand <= current stock < reorder point)
        # ~52% LOW risk (current stock >= reorder point)
        rand_val = np.random.uniform(0, 1)

        if rand_val < 0.18:
            # High risk
            current_stock = round(np.random.uniform(0.1, 0.8) * expected_lt_demand, 1)
        elif rand_val < 0.48:
            # Medium risk
            current_stock = round(
                expected_lt_demand + np.random.uniform(0.05, 0.85) * safety_stock, 1
            )
        else:
            # Low risk
            current_stock = round(
                reorder_point + np.random.uniform(0.2, 1.8) * expected_lt_demand, 1
            )

        current_stock = max(0.0, current_stock)

        records_inventory.append({
            "store_nbr": store_nbr,
            "family": family,
            "current_stock": current_stock,
            "safety_stock": safety_stock,
            "supplier_lead_time_days": lead_time_days,
            "last_updated": "2017-08-13",
        })

        records_suppliers.append({
            "store_nbr": store_nbr,
            "family": family,
            "supplier_lead_time_days": lead_time_days,
            "supplier_name": supplier_name,
        })

    df_inv = pd.DataFrame(records_inventory)
    df_sup = pd.DataFrame(records_suppliers)

    # Save CSV files
    inv_path = INVENTORY_DIR / "inventory.csv"
    sup_path = INVENTORY_DIR / "supplier_lead_times.csv"

    df_inv.to_csv(inv_path, index=False)
    df_sup.to_csv(sup_path, index=False)

    print(f"[+] Saved {len(df_inv)} demo inventory records to {inv_path}")
    print(f"[+] Saved {len(df_sup)} supplier records to {sup_path}")

    return df_inv, df_sup


if __name__ == "__main__":
    generate_demo_inventory_data()
