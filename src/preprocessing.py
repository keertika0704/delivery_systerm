"""Raw Olist CSVs -> clean, order-level processed tables + a data-quality report.

Run with ``.venv/bin/python -m src.preprocessing``.

Cleaning policy (conservative, every decision is logged in ``data_quality_report.json``):

* **No silent drops.**  Only *exact* duplicate rows and repeated primary keys are
  removed, and each removal is counted in the report.  Orders are never removed:
  ``orders_all`` has exactly one row per raw order.
* **Timestamps** are parsed with ``errors="coerce"``; the number of non-null raw values
  that failed to parse is reported (0 in the real data).
* **Non-delivered orders are never labelled on time** -- they get ``is_late = <NA>`` and
  an ``exclusion_reason``.  Delivered orders without a customer-delivery date, or with
  one earlier than purchase, are excluded the same way.
* **Illogical timestamp orders** (carrier before purchase, customer delivery before carrier,
  approval before purchase) are counted.  An order with an invalid carrier date stays in
  the analysis set; only ``processing_days`` becomes NaN and ``carrier_date_invalid = 1``.
* **Missing product weights/dimensions** stay NaN with ``weight_missing = 1``; imputation
  belongs inside the model pipeline (fitted on the training split only).  A weight or
  dimension of zero (or less) is physically implausible (4 products in the real data have
  0 g) and is treated as missing; the affected products / orders are counted in the
  report under ``product_quality``.
* **Orders without any item rows** (~775 in the real data) stay in ``orders_all`` with
  ``n_items = 0`` / ``n_sellers = 0`` / ``total_price = 0`` / ``total_freight = 0``, NaN
  weight / dimensions / freight_ratio / seller geography, ``weight_missing = 1`` and
  ``category_missing = 1`` (``product_category = "unknown"``).  In the real data every one
  of them is unavailable / cancelled / created / invoiced / shipped, hence ineligible; as a
  safeguard a *delivered* order with no items would be excluded as ``delivered_no_items``
  so item-less rows can never leak into the analysis set.
* **Untranslated categories** (two Portuguese names missing from the translation file)
  are mapped through a small documented supplement; any still-unmapped name keeps its
  Portuguese name.  A truly missing category is ``"unknown"``.
* Customer / seller identifiers are dropped from the processed tables (only ``order_id``).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    KAGGLE_URL,
    OPTIONAL_RAW_FILES,
    ORDERS_ALL_PATH,
    ORDERS_DELIVERED_CSV,
    ORDERS_DELIVERED_PATH,
    PROCESSED_DIR,
    QUALITY_REPORT_PATH,
    RAW_DIR,
    REQUIRED_RAW_FILES,
    TARGET_COL,
    TIMESTAMP_COLS,
)
from .feature_engineering import (
    PROCESSED_COLUMNS,
    UNKNOWN_CATEGORY,
    add_delivery_targets,
    add_geo_features,
    add_time_features,
    aggregate_items,
    validate_processed,
)

#: Source categories absent from the official translation file (documented supplement).
SUPPLEMENTAL_TRANSLATIONS = {
    "pc_gamer": "pc_gamer",
    "portateis_cozinha_e_preparadores_de_alimentos": "portable_kitchen_food_preparers",
}

#: Primary key(s) used for the duplicate-key check of each raw table.
_KEYS = {
    "orders": ["order_id"],
    "items": ["order_id", "order_item_id"],
    "customers": ["customer_id"],
    "sellers": ["seller_id"],
    "products": ["product_id"],
    "reviews": ["review_id", "order_id"],
    "category_translation": ["product_category_name"],
}
_TS_FORMAT = "%Y-%m-%d %H:%M:%S"


class MissingDatasetError(FileNotFoundError):
    """Raised when required raw Olist CSV files are not found."""


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_raw(raw_dir: Path | str = RAW_DIR) -> dict[str, pd.DataFrame]:
    """Read the raw Olist CSVs into a dict keyed ``orders, items, customers, sellers, products``
    (+ ``reviews``, ``category_translation`` when present).

    All columns are read as strings (``dtype=str``) so identifiers keep their exact text and
    parsing / coercion is explicit and counted later.  Raises :class:`MissingDatasetError`
    with placement and download instructions when a *required* file is absent.
    """
    raw_dir = Path(raw_dir)
    missing = [fn for fn in REQUIRED_RAW_FILES.values() if not (raw_dir / fn).is_file()]
    if missing:
        raise MissingDatasetError(
            "Required Olist raw file(s) not found in "
            f"{raw_dir}:\n  - " + "\n  - ".join(missing) + "\n\n"
            f"Download the 'Brazilian E-Commerce Public Dataset by Olist' from {KAGGLE_URL} "
            f"(free Kaggle account), unzip it, and place the CSV files in:\n  {raw_dir}\n"
            "Required: " + ", ".join(REQUIRED_RAW_FILES.values()) + ".\n"
            "Optional: " + ", ".join(OPTIONAL_RAW_FILES.values()) + ".\n"
            "Then run:  .venv/bin/python -m src.preprocessing"
        )
    raw: dict[str, pd.DataFrame] = {}
    for name, fn in {**REQUIRED_RAW_FILES, **OPTIONAL_RAW_FILES}.items():
        path = raw_dir / fn
        if path.is_file():
            raw[name] = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=True)
    return raw


# --------------------------------------------------------------------------- #
# Cleaning helpers
# --------------------------------------------------------------------------- #
def _py(x: Any) -> Any:
    """Make numpy / pandas scalars JSON serialisable."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if x is pd.NaT:  # must precede the datetime check: NaT is a datetime instance
        return None
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    if isinstance(x, dict):
        return {str(k): _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    return x


def _observed_counts(s: pd.Series) -> dict[str, int]:
    """``value_counts`` restricted to observed values (no zero-count categorical levels)."""
    vc = s.value_counts()
    return {str(k): int(v) for k, v in vc[vc > 0].items()}


def _drop_unused_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Remove categorical levels that have no rows (used after cutting a subset)."""
    out = df.copy()
    for c in out.columns[out.dtypes.map(lambda t: isinstance(t, pd.CategoricalDtype))]:
        out[c] = out[c].cat.remove_unused_categories()
    return out


def _inspect_and_dedupe(name: str, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Inspect one raw table and remove exact duplicates / repeated keys (counted, never silent).

    Handling: (1) exact duplicate rows -> keep first; (2) remaining rows that repeat the
    primary key with *different* content -> keep first (documented; 0 in the real data
    except reviews, which are reduced to one per order later and are NOT dropped here).
    """
    key = _KEYS[name]
    info: dict[str, Any] = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "dtypes_as_read": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {c: int(v) for c, v in df.isna().sum().items() if v},
    }
    exact = int(df.duplicated().sum())
    df = df.drop_duplicates()
    key_dups = int(df.duplicated(key).sum())
    info["duplicates"] = {
        "key": key,
        "exact_duplicate_rows": exact,
        "exact_duplicates_removed": exact,
        "duplicate_key_rows_after_exact_removal": key_dups,
    }
    if name == "reviews":
        # review_id repeats across orders in the real data and an order may have several
        # reviews; both are legitimate and handled by reducing to one review per order.
        info["duplicates"]["duplicate_order_id_rows"] = int(df.duplicated("order_id").sum())
        info["duplicates"]["duplicate_review_id_rows"] = int(df.duplicated("review_id").sum())
        info["duplicates"]["handling"] = "kept; reduced to latest review per order"
        info["duplicates"]["duplicate_key_removed"] = 0
    else:
        info["duplicates"]["duplicate_key_removed"] = key_dups
        info["duplicates"]["handling"] = "exact duplicates dropped; repeated keys keep first"
        if key_dups:
            df = df.drop_duplicates(key, keep="first")
    info["rows_after_cleaning"] = int(len(df))
    return df.reset_index(drop=True), info


def _parse_timestamps(orders: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Parse the five order timestamps (coerce) and report unparseable non-null values."""
    out = orders.copy()
    report = {}
    for col in TIMESTAMP_COLS:
        raw = out[col]
        parsed = pd.to_datetime(raw, format=_TS_FORMAT, errors="coerce").astype("datetime64[ns]")
        report[col] = {
            "missing_in_raw": int(raw.isna().sum()),
            "unparseable": int((raw.notna() & parsed.isna()).sum()),
        }
        out[col] = parsed
    return out, report


def _anomaly_counts(df: pd.DataFrame, mask: pd.Series | None = None) -> dict:
    """Counts of illogical timestamp orderings (optionally within a row mask)."""
    d = df if mask is None else df[mask]
    p, a = d["order_purchase_timestamp"], d["order_approved_at"]
    c, u = d["order_delivered_carrier_date"], d["order_delivered_customer_date"]
    return {
        "approved_before_purchase": int((a < p).sum()),
        "carrier_before_purchase": int((c < p).sum()),
        "customer_delivery_before_carrier": int((u < c).sum()),
        "customer_delivery_before_purchase": int((u < p).sum()),
        "carrier_before_approval": int((c < a).sum()),
    }


def _prepare_translation(translation: pd.DataFrame | None, products: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Translation table + documented supplement for source categories it lacks."""
    if translation is None:
        base = pd.DataFrame(columns=["product_category_name", "product_category_name_english"])
        note = "translation file absent; Portuguese category names are kept"
    else:
        base, note = translation.copy(), "official translation file used"
    source = set(products["product_category_name"].dropna().unique())
    absent = sorted(source - set(base["product_category_name"]))
    sup = {k: v for k, v in SUPPLEMENTAL_TRANSLATIONS.items() if k in absent}
    if sup:
        base = pd.concat([base, pd.DataFrame({"product_category_name": list(sup),
                                              "product_category_name_english": list(sup.values())})],
                         ignore_index=True)
    still = [c for c in absent if c not in sup]
    return base, {"note": note, "categories_missing_from_translation": absent,
                  "supplemental_translations_applied": sup, "still_untranslated": still}


def _product_quality(products: pd.DataFrame, items: pd.DataFrame) -> dict:
    """Counts of missing / non-positive product weight and dimensions (kept as NaN downstream)."""
    num = {c: pd.to_numeric(products[c], errors="coerce") for c in
           ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]}
    dims_bad = pd.concat([num[c].le(0) for c in num if c != "product_weight_g"], axis=1).any(axis=1)
    zero_w = num["product_weight_g"].le(0)
    bad_ids = set(products.loc[zero_w | dims_bad, "product_id"])
    return {
        "products": int(len(products)),
        "products_missing_weight": int(num["product_weight_g"].isna().sum()),
        "zero_weight_products": int(zero_w.sum()),
        "nonpositive_dimension_products": int(dims_bad.sum()),
        "item_rows_with_nonpositive_attribute": int(items["product_id"].isin(bad_ids).sum()),
        "orders_with_nonpositive_attribute": int(items.loc[items["product_id"].isin(bad_ids), "order_id"].nunique()),
        "handling": "weight/dimension <= 0 treated as missing (NaN, weight_missing=1)",
    }


def _reduce_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """One review per order: latest ``review_creation_date`` (ties -> latest answer timestamp,
    then highest review_id for determinism).  Keeps only ``review_score``."""
    r = reviews[["review_id", "order_id", "review_score", "review_creation_date",
                 "review_answer_timestamp"]].copy()
    r["review_score"] = pd.to_numeric(r["review_score"], errors="coerce")
    r["_created"] = pd.to_datetime(r["review_creation_date"], errors="coerce")
    r["_answered"] = pd.to_datetime(r["review_answer_timestamp"], errors="coerce")
    r = r.sort_values(["order_id", "_created", "_answered", "review_id"],
                      ascending=[True, True, True, True], na_position="first")
    return r.drop_duplicates("order_id", keep="last")[["order_id", "review_score"]]


# --------------------------------------------------------------------------- #
# Main build
# --------------------------------------------------------------------------- #
def build_orders_table(raw: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """Build the one-row-per-order processed frame and the data-quality report.

    Returns ``(orders_all, report)``.  ``orders_all`` follows ``PROCESSED_COLUMNS`` exactly.
    """
    report: dict[str, Any] = {"raw_files": {}}
    clean: dict[str, pd.DataFrame] = {}
    for name, df in raw.items():
        clean[name], report["raw_files"][name] = _inspect_and_dedupe(name, df)

    orders, ts_report = _parse_timestamps(clean["orders"])
    report["timestamp_parsing"] = ts_report
    n_orders_raw = int(len(raw["orders"]))

    # --- customers (1:1 on customer_id) -------------------------------------------------
    cust = clean["customers"][["customer_id", "customer_state", "customer_city"]]
    orders = orders.merge(cust, on="customer_id", how="left", validate="many_to_one")
    report["customer_join"] = {
        "orders_without_matching_customer": int(orders["customer_state"].isna().sum()),
        "orders_sharing_customer_id": int(orders["customer_id"].duplicated().sum()),
    }

    # --- items -> order aggregates -------------------------------------------------------
    translation, tr_report = _prepare_translation(clean.get("category_translation"), clean["products"])
    report["category_translation"] = tr_report
    items = clean["items"]
    item_orders_unknown = int((~items["order_id"].isin(orders["order_id"])).sum())
    items_product_missing = int((~items["product_id"].isin(clean["products"]["product_id"])).sum())
    items_seller_missing = int((~items["seller_id"].isin(clean["sellers"]["seller_id"])).sum())
    agg = aggregate_items(items, clean["products"], clean["sellers"], translation)
    product_quality = _product_quality(clean["products"], items)
    orders = orders.merge(agg, on="order_id", how="left", validate="one_to_one")

    for c in ["n_items", "n_products", "n_sellers", "n_categories"]:
        orders[c] = orders[c].fillna(0).astype("int16")
    for c in ["total_price", "total_freight"]:
        orders[c] = orders[c].fillna(0.0)
    for c in ["weight_missing", "category_missing"]:
        orders[c] = orders[c].fillna(1).astype("int8")  # no items -> attribute unavailable
    orders["product_category"] = orders["product_category"].fillna(UNKNOWN_CATEGORY)

    # --- reviews (descriptive only) ----------------------------------------------------
    if "reviews" in clean:
        orders = orders.merge(_reduce_reviews(clean["reviews"]), on="order_id", how="left",
                              validate="one_to_one")
    else:
        orders["review_score"] = np.nan
    orders["review_score"] = orders["review_score"].astype("float64")

    # --- features, targets, geography ---------------------------------------------------
    anomalies_all = _anomaly_counts(orders)
    delivered_mask = orders["order_status"] == "delivered"
    anomalies_delivered = _anomaly_counts(orders, delivered_mask)
    orders = add_time_features(orders)
    orders = add_delivery_targets(orders)
    orders = add_geo_features(orders)

    # --- dtypes / column selection ------------------------------------------------------
    for c in ["order_status", "customer_state", "customer_city", "seller_state", "seller_city",
              "state_pair", "product_category"]:
        orders[c] = orders[c].astype("category")
    orders["exclusion_reason"] = orders["exclusion_reason"].astype(str)
    orders["purchase_year_month"] = orders["purchase_year_month"].astype("object")
    orders = orders[PROCESSED_COLUMNS].sort_values(["order_purchase_timestamp", "order_id"], kind="stable",
                                                   na_position="last").reset_index(drop=True)

    problems = validate_processed(orders)
    if problems:
        raise ValueError("Processed table failed validation: " + "; ".join(problems))
    if len(orders) != n_orders_raw - report["raw_files"]["orders"]["duplicates"]["exact_duplicate_rows"] \
            - report["raw_files"]["orders"]["duplicates"]["duplicate_key_removed"]:
        raise ValueError("Row count changed unexpectedly during processing (orders were dropped).")

    # --- report -------------------------------------------------------------------------
    elig = orders["analysis_eligible"]
    n_elig = int(elig.sum())
    late_rate = float(orders.loc[elig, TARGET_COL].astype(float).mean()) if n_elig else None
    pre = orders.loc[orders["n_items"] == 0, "order_status"]
    report.update({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_orders_raw": n_orders_raw,
        "n_orders_all": int(len(orders)),
        "logical_order_anomalies_all_orders": anomalies_all,
        "logical_order_anomalies_delivered_orders": anomalies_delivered,
        "product_quality": product_quality,
        "referential_integrity": {
            "item_rows_with_unknown_order": item_orders_unknown,
            "item_rows_with_unknown_product": items_product_missing,
            "item_rows_with_unknown_seller": items_seller_missing,
        },
        "order_status_counts": _observed_counts(orders["order_status"]),
        "exclusion_reason_counts": {str(k): int(v) for k, v in
                                    orders.loc[~elig, "exclusion_reason"].value_counts().items()},
        "exclusions": {
            "non_delivered_status": int((orders["order_status"] != "delivered").sum()),
            "delivered_missing_delivery_date": int((orders["exclusion_reason"] == "delivered_missing_delivery_date").sum()),
            "delivered_before_purchase": int((orders["exclusion_reason"] == "delivered_before_purchase").sum()),
            "delivered_no_items": int((orders["exclusion_reason"] == "delivered_no_items").sum()),
            "missing_purchase_or_estimate": int((orders["exclusion_reason"] == "missing_purchase_or_estimate").sum()),
        },
        "orders_without_items": {
            "count": int((orders["n_items"] == 0).sum()),
            "status_counts": _observed_counts(pre),
            "eligible_among_them": int((elig & (orders["n_items"] == 0)).sum()),
            "handling": "kept in orders_all with n_items=0, zero price/freight, NaN weight/dimensions; never eligible",
        },
        "multi_item_orders": int((orders["n_items"] > 1).sum()),
        "multi_seller_orders": int((orders["n_sellers"] > 1).sum()),
        "multi_category_orders": int((orders["n_categories"] > 1).sum()),
        "eligible_quality": {
            "carrier_date_invalid": int(orders.loc[elig, "carrier_date_invalid"].sum()),
            "approval_missing": int(orders.loc[elig, "approval_missing"].sum()),
            "category_missing": int(orders.loc[elig, "category_missing"].sum()),
            "weight_missing": int(orders.loc[elig, "weight_missing"].sum()),
            "carrier_to_customer_negative_set_nan": int(
                (elig & (orders["order_delivered_customer_date"] < orders["order_delivered_carrier_date"])).sum()),
            "review_score_missing": int(orders.loc[elig, "review_score"].isna().sum()),
        },
        "n_eligible": n_elig,
        "n_late_eligible": int(orders.loc[elig, TARGET_COL].astype(float).sum()) if n_elig else 0,
        "late_rate_eligible": late_rate,
        "late_definition": "calendar date of customer delivery > calendar date of estimated delivery",
        "late_rate_timestamp_based_for_reference": (
            float((orders.loc[elig, "order_delivered_customer_date"] >
                   orders.loc[elig, "order_estimated_delivery_date"]).mean()) if n_elig else None),
        "date_range_all": {
            "min": _py(orders["order_purchase_timestamp"].min()),
            "max": _py(orders["order_purchase_timestamp"].max()),
        },
        "date_range_eligible": {
            "min": _py(orders.loc[elig, "order_purchase_timestamp"].min()) if n_elig else None,
            "max": _py(orders.loc[elig, "order_purchase_timestamp"].max()) if n_elig else None,
        },
        "validation_problems": problems,
    })
    return orders, _py(report)


# --------------------------------------------------------------------------- #
# Pipeline / IO
# --------------------------------------------------------------------------- #
def run_pipeline(raw_dir: Path | str = RAW_DIR, out_dir: Path | str = PROCESSED_DIR) -> dict:
    """Load raw CSVs, build the processed tables and write all artefacts.

    Writes ``orders_all.parquet``, ``orders_delivered.parquet``, ``orders_delivered.csv`` and
    ``data_quality_report.json`` into *out_dir*.  Returns the quality report (with an
    ``outputs`` section listing the written files).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = load_raw(raw_dir)
    orders, report = build_orders_table(raw)
    delivered = _drop_unused_levels(orders[orders["analysis_eligible"]].reset_index(drop=True))

    paths = {
        "orders_all": out_dir / ORDERS_ALL_PATH.name,
        "orders_delivered": out_dir / ORDERS_DELIVERED_PATH.name,
        "orders_delivered_csv": out_dir / ORDERS_DELIVERED_CSV.name,
        "quality_report": out_dir / QUALITY_REPORT_PATH.name,
    }
    orders.to_parquet(paths["orders_all"], index=False)
    delivered.to_parquet(paths["orders_delivered"], index=False)
    delivered.to_csv(paths["orders_delivered_csv"], index=False)
    report["outputs"] = {k: str(v) for k, v in paths.items()}
    report["outputs"]["rows"] = {"orders_all": int(len(orders)), "orders_delivered": int(len(delivered))}
    paths["quality_report"].write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def load_processed(path: Path | str = ORDERS_DELIVERED_PATH) -> pd.DataFrame:
    """Load a processed parquet (default: the delivered analysis set).  No Streamlit dependency.

    Raises ``FileNotFoundError`` with the command that generates it.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Processed data not found at {path}. Generate it with:\n"
            "  .venv/bin/python -m src.preprocessing\n"
            f"(raw Olist CSVs must be in {RAW_DIR}; see {KAGGLE_URL})"
        )
    return pd.read_parquet(path)


def _print_log(r: dict) -> None:
    """Short human-readable cleaning log."""
    pct = lambda x: f"{100 * x:.2f}%"  # noqa: E731
    print("=" * 66)
    print("DeliveryIQ preprocessing log")
    print("=" * 66)
    for name, info in r["raw_files"].items():
        d = info["duplicates"]
        print(f"[{name:<20}] rows={info['rows']:>7,}  exact dups={d['exact_duplicate_rows']}  "
              f"dup keys removed={d['duplicate_key_removed']}  cols with NaN={len(info['missing_values'])}")
    print("-" * 66)
    bad_ts = {k: v["unparseable"] for k, v in r["timestamp_parsing"].items() if v["unparseable"]}
    print(f"Unparseable timestamps: {bad_ts or 'none'}")
    print("Logical anomalies (delivered orders):", r["logical_order_anomalies_delivered_orders"])
    print(f"Orders: {r['n_orders_all']:,} (raw {r['n_orders_raw']:,}) | without items: "
          f"{r['orders_without_items']['count']:,} {r['orders_without_items']['status_counts']}")
    print(f"Multi-item {r['multi_item_orders']:,} | multi-seller {r['multi_seller_orders']:,} | "
          f"multi-category {r['multi_category_orders']:,}")
    print("Exclusions:", r["exclusion_reason_counts"])
    print(f"Eligible (delivered, valid dates): {r['n_eligible']:,} | late: {r['n_late_eligible']:,} "
          f"({pct(r['late_rate_eligible'])}; timestamp-based reference "
          f"{pct(r['late_rate_timestamp_based_for_reference'])})")
    print(f"Eligible quality flags: {r['eligible_quality']}")
    print(f"Purchase range (eligible): {r['date_range_eligible']['min']} -> {r['date_range_eligible']['max']}")
    print(f"Validation problems: {r['validation_problems'] or 'none'}")
    print("Wrote:", ", ".join(Path(p).name for k, p in r["outputs"].items() if k != "rows"))


def main() -> int:
    try:
        report = run_pipeline()
    except MissingDatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    _print_log(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
