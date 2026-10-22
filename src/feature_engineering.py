"""Pure, vectorised feature-engineering functions for DeliveryIQ.

Everything here operates on DataFrames and has no I/O, so it can be unit-tested with
tiny hand-built fixtures.  The definitions implemented are the ones in
``docs/ARCHITECTURE.md`` section 1:

* lateness is **calendar-date based** (arriving any time on the promised date is on time);
* only ``delivered`` orders with a valid purchase / estimate / customer-delivery
  timestamp are *analysis eligible*; every other order gets ``is_late = <NA>`` and an
  ``exclusion_reason`` -- a non-delivered order is never labelled "on time";
* multi-item / multi-seller orders are aggregated to one row per order, with a
  deterministic representative seller and product category.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TARGET_COL

UNKNOWN_CATEGORY = "unknown"

#: Column order and the full set of columns allowed in the processed tables.
PROCESSED_COLUMNS: list[str] = [
    # key
    "order_id",
    # status
    "order_status", "analysis_eligible", "exclusion_reason",
    # timestamps
    "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
    "order_delivered_customer_date", "order_estimated_delivery_date",
    # targets / outcomes
    "is_late", "delivery_delay_days", "delivery_duration_days", "processing_days",
    "approval_days", "carrier_to_customer_days",
    # quality flags
    "approval_missing", "carrier_date_invalid", "category_missing", "weight_missing",
    # time
    "purchase_date", "purchase_year", "purchase_month", "purchase_quarter",
    "purchase_dow", "purchase_year_month", "estimated_delivery_days",
    # geography
    "customer_state", "customer_city", "seller_state", "seller_city", "state_pair",
    "same_state",
    # order aggregates
    "n_items", "n_sellers", "n_products", "n_categories", "total_price", "total_freight",
    "freight_ratio", "total_weight_g", "total_volume_cm3", "max_product_length_cm",
    "max_product_height_cm", "max_product_width_cm", "product_category",
    # descriptive extras
    "review_score",
]

FLAG_COLUMNS = ["approval_missing", "carrier_date_invalid", "category_missing", "weight_missing"]
AGGREGATE_NONNEG = [
    "n_items", "n_sellers", "n_products", "n_categories", "total_price", "total_freight",
    "total_weight_g", "total_volume_cm3", "max_product_length_cm", "max_product_height_cm",
    "max_product_width_cm", "processing_days", "approval_days", "carrier_to_customer_days",
]
_TS = {
    "purchase": "order_purchase_timestamp",
    "approved": "order_approved_at",
    "carrier": "order_delivered_carrier_date",
    "customer": "order_delivered_customer_date",
    "estimate": "order_estimated_delivery_date",
}
_ONE_DAY = pd.Timedelta(days=1)


def _days(later: pd.Series, earlier: pd.Series) -> pd.Series:
    """Fractional days between two datetime Series (NaN if either is missing)."""
    return (later - earlier) / _ONE_DAY


# --------------------------------------------------------------------------- #
# Time features
# --------------------------------------------------------------------------- #
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add purchase-time calendar features and the checkout delivery promise.

    ``estimated_delivery_days`` = estimated date - purchase date (both normalised to
    midnight, integer days).  It is the promise shown at checkout, hence known at
    purchase time.  Missing purchase timestamps give <NA> (rows are never dropped).
    """
    out = df.copy()
    purchase = out[_TS["purchase"]]
    out["purchase_date"] = purchase.dt.normalize()
    out["purchase_year"] = purchase.dt.year.astype("Int16")
    out["purchase_month"] = purchase.dt.month.astype("Int8")
    out["purchase_quarter"] = purchase.dt.quarter.astype("Int8")
    out["purchase_dow"] = purchase.dt.day_of_week.astype("Int8")  # 0 = Monday
    out["purchase_year_month"] = purchase.dt.strftime("%Y-%m")
    est_days = (out[_TS["estimate"]].dt.normalize() - out["purchase_date"]).dt.days
    out["estimated_delivery_days"] = est_days.astype("Int16")
    return out


# --------------------------------------------------------------------------- #
# Eligibility, targets and quality flags
# --------------------------------------------------------------------------- #
def add_delivery_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add eligibility, exclusion reason, target and outcome / diagnostic columns.

    Exclusion precedence (first match wins):
      1. ``status_<status>``            -- any status other than ``delivered``
      2. ``missing_purchase_or_estimate`` -- cannot compute a promise
      3. ``delivered_missing_delivery_date``
      4. ``delivered_before_purchase``  -- customer date earlier than purchase
      5. ``delivered_no_items``         -- defensive; only applied when an ``n_items``
         column is present (0 rows in the real data: item-less orders are all
         unavailable / cancelled / early-stage).

    Outcomes (``is_late``, ``delivery_delay_days``, ``delivery_duration_days``,
    ``carrier_to_customer_days``) are computed **only for eligible rows**, so a
    cancelled order can never be labelled on time.  ``processing_days`` and
    ``approval_days`` are retrospective diagnostics available for any row with valid
    timestamps.

    Quality flags: ``approval_missing`` (no approval timestamp), ``carrier_date_invalid``
    (carrier timestamp present but earlier than purchase; ``processing_days`` is then
    NaN, the row is kept).  A negative approval gap or carrier->customer gap is set to
    NaN (never silently "fixed").
    """
    out = df.copy()
    ts = {k: out[v] for k, v in _TS.items()}
    status = out["order_status"].astype("string")

    # null status -> not delivered (-> "status_missing"); never a nullable-boolean NA
    delivered = (status == "delivered").fillna(False).astype(bool)
    no_purchase_or_est = ts["purchase"].isna() | ts["estimate"].isna()
    no_customer_date = ts["customer"].isna()
    before_purchase = ts["customer"] < ts["purchase"]  # NaT compares False
    no_items = (out["n_items"] == 0) if "n_items" in out.columns else pd.Series(False, index=out.index)

    reason = pd.Series("", index=out.index, dtype="object")
    # assign in reverse precedence so that the highest-precedence reason wins
    reason[delivered & no_items] = "delivered_no_items"
    reason[delivered & before_purchase] = "delivered_before_purchase"
    reason[delivered & no_customer_date] = "delivered_missing_delivery_date"
    reason[delivered & no_purchase_or_est] = "missing_purchase_or_estimate"
    reason[~delivered] = "status_" + status[~delivered].fillna("missing").astype(str)

    eligible = reason == ""
    out["analysis_eligible"] = eligible.astype(bool)
    out["exclusion_reason"] = reason.astype(str)

    # --- outcomes, eligible rows only ------------------------------------------------
    delay = (ts["customer"].dt.normalize() - ts["estimate"].dt.normalize()).dt.days
    delay = delay.where(eligible)
    out["delivery_delay_days"] = delay.astype("Int32")
    out[TARGET_COL] = (delay > 0).astype("Int8").where(eligible, pd.NA).astype("Int8")
    out["delivery_duration_days"] = _days(ts["customer"], ts["purchase"]).where(eligible)

    carrier_c2c = _days(ts["customer"], ts["carrier"]).where(eligible)
    out["carrier_to_customer_days"] = carrier_c2c.where(carrier_c2c >= 0)

    # --- retrospective diagnostics + quality flags -----------------------------------
    carrier_invalid = ts["carrier"] < ts["purchase"]
    out["carrier_date_invalid"] = carrier_invalid.astype("int8")
    processing = _days(ts["carrier"], ts["purchase"])
    out["processing_days"] = processing.where(~carrier_invalid)

    approval = _days(ts["approved"], ts["purchase"])
    out["approval_days"] = approval.where(approval >= 0)
    out["approval_missing"] = ts["approved"].isna().astype("int8")
    return out


# --------------------------------------------------------------------------- #
# Item aggregation
# --------------------------------------------------------------------------- #
def aggregate_items(
    items: pd.DataFrame,
    products: pd.DataFrame,
    sellers: pd.DataFrame,
    translation: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate order-item rows to one row per order (only orders that have items).

    Per-item product weight / volume come from the products table; each item row is
    one unit, so ``total_weight_g`` and ``total_volume_cm3`` are plain sums.  If *any*
    item of the order lacks the attribute, the order-level total / max is NaN (an
    understated total would be misleading) and ``weight_missing = 1``.  A weight or
    dimension that is zero or negative is physically implausible and is treated as missing.

    Representative seller: largest summed ``price`` in the order (ties -> smallest
    ``seller_id``).  Representative category: category of the highest-priced item (ties
    -> smallest ``order_item_id``); a missing category becomes ``"unknown"`` and sets
    ``category_missing``.  ``n_categories`` counts distinct *known* categories.

    Returns columns: order_id, n_items, n_products, n_sellers, n_categories, total_price,
    total_freight, freight_ratio, total_weight_g, total_volume_cm3, max_product_*_cm,
    product_category, category_missing, weight_missing, representative_seller_id,
    seller_state, seller_city.  (``representative_seller_id`` is for traceability and is
    dropped before the processed tables are written.)
    """
    it = items[["order_id", "order_item_id", "product_id", "seller_id", "price", "freight_value"]].copy()
    it["price"] = pd.to_numeric(it["price"], errors="coerce")
    it["freight_value"] = pd.to_numeric(it["freight_value"], errors="coerce")
    it["order_item_id"] = pd.to_numeric(it["order_item_id"], errors="coerce")

    prod_cols = ["product_id", "product_category_name", "product_weight_g",
                 "product_length_cm", "product_height_cm", "product_width_cm"]
    prod = products[prod_cols].drop_duplicates("product_id")
    it = it.merge(prod, on="product_id", how="left", validate="many_to_one")

    cat = it["product_category_name"].astype("object")
    if translation is not None and len(translation):
        mapping = dict(zip(translation["product_category_name"], translation["product_category_name_english"]))
        translated = cat.map(mapping)
        cat = translated.where(translated.notna(), cat)  # untranslated -> keep source name
    it["category"] = cat.where(cat.notna() & (cat.astype(str).str.strip() != ""), UNKNOWN_CATEGORY)

    for c in ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]:
        v = pd.to_numeric(it[c], errors="coerce")
        it[c] = v.where(v > 0)  # 0 g / 0 cm is physically implausible -> treated as missing
    it["volume"] = it["product_length_cm"] * it["product_height_cm"] * it["product_width_cm"]
    it["price_cents"] = np.round(it["price"].fillna(0) * 100).astype("int64")

    g = it.groupby("order_id", sort=True)

    # any-missing flags per order, vectorised
    any_missing = it[["product_weight_g", "volume", "product_length_cm",
                      "product_height_cm", "product_width_cm"]].isna().groupby(it["order_id"], sort=True).any()

    agg = pd.DataFrame({
        "n_items": g.size(),
        "n_products": g["product_id"].nunique(),
        "n_sellers": g["seller_id"].nunique(),
        "total_price": g["price"].sum(),
        "total_freight": g["freight_value"].sum(),
        "total_weight_g": g["product_weight_g"].sum(min_count=1).where(~any_missing["product_weight_g"]),
        "total_volume_cm3": g["volume"].sum(min_count=1).where(~any_missing["volume"]),
        "max_product_length_cm": g["product_length_cm"].max().where(~any_missing["product_length_cm"]),
        "max_product_height_cm": g["product_height_cm"].max().where(~any_missing["product_height_cm"]),
        "max_product_width_cm": g["product_width_cm"].max().where(~any_missing["product_width_cm"]),
    })
    known = it[it["category"] != UNKNOWN_CATEGORY]
    agg["n_categories"] = known.groupby("order_id")["category"].nunique().reindex(agg.index).fillna(0)
    agg["weight_missing"] = (any_missing["product_weight_g"] | any_missing["volume"]).astype("int8")

    price = agg["total_price"]
    agg["freight_ratio"] = (agg["total_freight"] / price).where(price > 0)

    # representative seller: max summed price, tie -> smallest seller_id
    seller_sum = it.groupby(["order_id", "seller_id"], as_index=False)["price_cents"].sum()
    seller_sum = seller_sum.sort_values(["order_id", "price_cents", "seller_id"], ascending=[True, False, True])
    rep_seller = seller_sum.drop_duplicates("order_id").set_index("order_id")["seller_id"]
    agg["representative_seller_id"] = rep_seller.reindex(agg.index)

    # representative category: highest-priced item, tie -> smallest order_item_id
    ordered = it.sort_values(["order_id", "price_cents", "order_item_id"], ascending=[True, False, True])
    rep_item = ordered.drop_duplicates("order_id").set_index("order_id")
    agg["product_category"] = rep_item["category"].reindex(agg.index)
    agg["category_missing"] = (agg["product_category"] == UNKNOWN_CATEGORY).astype("int8")

    agg = agg.reset_index()
    sel = sellers[["seller_id", "seller_state", "seller_city"]].drop_duplicates("seller_id")
    agg = agg.merge(sel, left_on="representative_seller_id", right_on="seller_id", how="left",
                    validate="many_to_one").drop(columns="seller_id")

    for c in ["n_items", "n_products", "n_sellers", "n_categories"]:
        agg[c] = agg[c].astype("int16")
    return agg


# --------------------------------------------------------------------------- #
# Geography
# --------------------------------------------------------------------------- #
def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``state_pair`` (``"SP→RJ"`` seller -> customer) and ``same_state``.

    When the seller state is unknown (orders with no items) ``state_pair`` is NaN and
    ``same_state`` is 0; such orders are never analysis-eligible (checked by
    :func:`validate_processed`).
    """
    out = df.copy()
    seller = out["seller_state"].astype("object")
    cust = out["customer_state"].astype("object")
    both = seller.notna() & cust.notna()
    pair = (seller.astype(str) + "→" + cust.astype(str)).where(both)
    out["state_pair"] = pair
    out["same_state"] = (both & (seller == cust)).astype("int8")
    return out


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_processed(df: pd.DataFrame) -> list[str]:
    """Return a list of schema / invariant problems (empty list when the frame is clean)."""
    problems: list[str] = []
    missing = [c for c in PROCESSED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in PROCESSED_COLUMNS]
    if missing:
        problems.append(f"missing columns: {missing}")
    if extra:
        problems.append(f"unexpected columns (possible identifier/leakage): {extra}")
    if missing:
        return problems  # invariants below need the full schema

    if df["order_id"].isna().any():
        problems.append("order_id contains nulls")
    n_dup = int(df["order_id"].duplicated().sum())
    if n_dup:
        problems.append(f"order_id not unique ({n_dup} duplicated)")

    elig = df["analysis_eligible"].astype(bool)
    is_late = df[TARGET_COL]
    bad_vals = is_late.dropna().loc[~is_late.dropna().isin([0, 1])]
    if len(bad_vals):
        problems.append(f"is_late has values outside {{0,1}}: {int(len(bad_vals))} rows")
    if is_late[elig].isna().any():
        problems.append("eligible rows with missing is_late")
    if is_late[~elig].notna().any():
        problems.append(f"ineligible rows labelled with is_late ({int(is_late[~elig].notna().sum())})")
    if df["delivery_delay_days"][~elig].notna().any():
        problems.append("ineligible rows carry delivery_delay_days")

    status = df["order_status"].astype("string")
    if (elig & ~(status == "delivered").fillna(False).astype(bool)).any():
        problems.append("eligible rows whose status is not 'delivered'")
    for col in ["order_purchase_timestamp", "order_estimated_delivery_date", "order_delivered_customer_date"]:
        if df.loc[elig, col].isna().any():
            problems.append(f"eligible rows with missing {col}")
    if (df.loc[elig, "order_delivered_customer_date"] < df.loc[elig, "order_purchase_timestamp"]).any():
        problems.append("eligible rows delivered before purchase")

    delay = df["delivery_delay_days"]
    mismatch = elig & (is_late.astype("float") != (delay > 0).astype("float"))
    if mismatch.any():
        problems.append(f"is_late inconsistent with delivery_delay_days>0 ({int(mismatch.sum())} rows)")

    reason = df["exclusion_reason"].astype(str)
    if ((reason == "") != elig).any():
        problems.append("exclusion_reason must be empty exactly for eligible rows")

    for c in FLAG_COLUMNS:
        if not df[c].isin([0, 1]).all():
            problems.append(f"flag column {c} has values outside {{0,1}}")
    if not df["same_state"].isin([0, 1]).all():
        problems.append("same_state has values outside {0,1}")

    for c in AGGREGATE_NONNEG:
        if (df[c] < 0).any():
            problems.append(f"negative values in {c}")
    if (df.loc[elig, "n_items"] < 1).any():
        problems.append("eligible rows with no items")
    if (df["n_categories"] > df["n_items"]).any() or (df["n_sellers"] > df["n_items"]).any():
        problems.append("n_categories / n_sellers exceed n_items")
    if df.loc[elig, "delivery_duration_days"].isna().any():
        problems.append("eligible rows with missing delivery_duration_days")
    if (df.loc[elig, "delivery_duration_days"] < 0).any():
        problems.append("eligible rows with negative delivery_duration_days")
    if df.loc[elig, "seller_state"].isna().any() or df.loc[elig, "customer_state"].isna().any():
        problems.append("eligible rows with unknown seller/customer state")
    if (df.loc[elig, "product_category"].isna()).any():
        problems.append("eligible rows with missing product_category")
    return problems
