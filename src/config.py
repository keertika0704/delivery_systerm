"""Central configuration for DeliveryIQ.

Single source of truth for paths, file names, column groups (including the
leakage allow/deny lists), palette and default thresholds.  Every other module
imports from here so that the dashboard, the training script and the tests can
never drift apart.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
GEO_DIR = ROOT / "data" / "geo"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Raw Olist files (Kaggle: olistbr/brazilian-ecommerce)
REQUIRED_RAW_FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "products": "olist_products_dataset.csv",
}
OPTIONAL_RAW_FILES = {
    "reviews": "olist_order_reviews_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}
KAGGLE_URL = "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"

# Processed artefacts (written by `python -m src.preprocessing`)
ORDERS_ALL_PATH = PROCESSED_DIR / "orders_all.parquet"          # every order, 1 row each
ORDERS_DELIVERED_PATH = PROCESSED_DIR / "orders_delivered.parquet"  # analysis set
ORDERS_DELIVERED_CSV = PROCESSED_DIR / "orders_delivered.csv"
QUALITY_REPORT_PATH = PROCESSED_DIR / "data_quality_report.json"

# Model artefacts (written by `python -m src.train_model`)
MODEL_PATH = MODELS_DIR / "delay_classifier.joblib"
MODEL_METRICS_PATH = MODELS_DIR / "model_metrics.json"

# --------------------------------------------------------------------------- #
# Reproducibility / defaults
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
DEFAULT_MIN_ORDERS = 100        # minimum sample before a segment is called "reliable"
DEFAULT_ALPHA = 0.05

# --------------------------------------------------------------------------- #
# Column groups
# --------------------------------------------------------------------------- #
TIMESTAMP_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

TARGET_COL = "is_late"  # 0 = on time / early, 1 = delivered after estimated date

# Candidate predictors that are known AT PURCHASE TIME.  The trained model may
# use any subset of these and nothing else (enforced in train_model + tests).
PURCHASE_TIME_NUMERIC = [
    "n_items",
    "n_sellers",
    "total_price",
    "total_freight",
    "freight_ratio",
    "total_weight_g",
    "total_volume_cm3",
    "max_product_length_cm",
    "max_product_height_cm",
    "max_product_width_cm",
    "estimated_delivery_days",   # promise made at checkout: estimated date - purchase date
    "purchase_month",
    "purchase_dow",
    "purchase_quarter",
]
PURCHASE_TIME_CATEGORICAL = [
    "customer_state",
    "seller_state",
    "product_category",
    "same_state",
]
PURCHASE_TIME_FEATURES = PURCHASE_TIME_NUMERIC + PURCHASE_TIME_CATEGORICAL

# Columns that must NEVER be model inputs (they encode the outcome or future info).
LEAKAGE_COLUMNS = [
    "order_status",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "delivery_duration_days",
    "delivery_delay_days",
    "processing_days",
    "approval_days",
    "carrier_to_customer_days",
    "is_late",
    "review_score",
    "analysis_eligible",
    "exclusion_reason",
    # flags derived from post-purchase timestamps (only knowable after the event)
    "carrier_date_invalid",
    "approval_missing",
]

# Processed columns that are deliberately NOT predictors although they are not leaky:
# keys, raw purchase/estimate timestamps (used for the chronological split) and
# purchase-date columns that would only act as a time-drift proxy.
NON_FEATURE_COLUMNS = [
    "order_id",
    "order_purchase_timestamp",
    "order_estimated_delivery_date",
    "purchase_date",
    "purchase_year",
    "purchase_year_month",
]

# Known at purchase time but intentionally outside the model allow-list (high-cardinality
# geography, data-quality flags and secondary aggregates kept for analytics only).
PURCHASE_TIME_UNUSED = [
    "customer_city",
    "seller_city",
    "state_pair",
    "n_products",
    "n_categories",
    "category_missing",
    "weight_missing",
]

# --------------------------------------------------------------------------- #
# Brazil: state codes (UF) -> names
# --------------------------------------------------------------------------- #
STATE_NAMES = {
    "AC": "Acre", "AL": "Alagoas", "AP": "Amapá", "AM": "Amazonas", "BA": "Bahia",
    "CE": "Ceará", "DF": "Distrito Federal", "ES": "Espírito Santo", "GO": "Goiás",
    "MA": "Maranhão", "MT": "Mato Grosso", "MS": "Mato Grosso do Sul",
    "MG": "Minas Gerais", "PA": "Pará", "PB": "Paraíba", "PR": "Paraná",
    "PE": "Pernambuco", "PI": "Piauí", "RJ": "Rio de Janeiro",
    "RN": "Rio Grande do Norte", "RS": "Rio Grande do Sul", "RO": "Rondônia",
    "RR": "Roraima", "SC": "Santa Catarina", "SP": "São Paulo", "SE": "Sergipe",
    "TO": "Tocantins",
}

# --------------------------------------------------------------------------- #
# Branding / palette (used by CSS, Plotly template and Matplotlib figures)
# --------------------------------------------------------------------------- #
APP_NAME = "DeliveryIQ"
APP_TAGLINE = "Intelligent Shipment Analytics & Delay Prediction"
PALETTE = {
    "navy": "#0F172A",
    "slate": "#334155",
    "slate_light": "#64748B",
    "border": "#E2E8F0",
    "background": "#F8FAFC",
    "card": "#FFFFFF",
    "blue": "#2563EB",      # primary analytical colour
    "blue_light": "#93C5FD",
    "green": "#16A34A",     # on time
    "amber": "#F59E0B",     # moderate risk
    "red": "#DC2626",       # late / high risk
}
