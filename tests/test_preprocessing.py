"""Tests for the data layer (feature_engineering + preprocessing).

Unit tests use SMALL HAND-BUILT SYNTHETIC FIXTURES (clearly synthetic ids such as
"o_ontime_2359"); they are *test fixtures only* and never used by the app.
Integration tests at the bottom run against the real processed parquet and are skipped
when it has not been generated.
"""
from __future__ import annotations

import functools
import json

import pandas as pd
import pytest

from src import config
from src.feature_engineering import (
    PROCESSED_COLUMNS,
    add_delivery_targets,
    add_geo_features,
    add_time_features,
    aggregate_items,
    validate_processed,
)
from src import preprocessing
from src.preprocessing import (
    MissingDatasetError,
    build_orders_table,
    load_processed,
    load_raw,
    run_pipeline,
)

# --------------------------------------------------------------------------- #
# Synthetic fixtures (TEST FIXTURES ONLY)
# --------------------------------------------------------------------------- #
FMT = "%Y-%m-%d %H:%M:%S"


def _orders_rows() -> list[dict]:
    """(order_id, customer, status, purchase, approved, carrier, customer_delivery, estimate)"""
    rows = [
        # on time: arrives 23:59 on the estimated calendar date
        ("o_ontime_2359", "c1", "delivered", "2018-01-02 10:00:00", "2018-01-02 11:00:00",
         "2018-01-03 09:00:00", "2018-01-12 23:59:00", "2018-01-12 00:00:00"),
        # late: arrives 00:01 the day after the estimated date
        ("o_late_0001", "c2", "delivered", "2018-01-02 10:00:00", "2018-01-02 11:00:00",
         "2018-01-03 09:00:00", "2018-01-13 00:01:00", "2018-01-12 00:00:00"),
        # early
        ("o_early", "c3", "delivered", "2018-01-02 10:00:00", "2018-01-02 11:00:00",
         "2018-01-03 09:00:00", "2018-01-08 12:00:00", "2018-01-12 00:00:00"),
        # cancelled, but with an (erroneous) delivery date -> must never be labelled
        ("o_cancelled", "c4", "canceled", "2018-01-02 10:00:00", None, None,
         "2018-01-05 12:00:00", "2018-01-12 00:00:00"),
        ("o_shipped", "c5", "shipped", "2018-01-02 10:00:00", "2018-01-02 11:00:00",
         "2018-01-03 09:00:00", None, "2018-01-12 00:00:00"),
        # delivered but no customer date
        ("o_nodate", "c6", "delivered", "2018-01-02 10:00:00", "2018-01-02 11:00:00",
         "2018-01-03 09:00:00", None, "2018-01-12 00:00:00"),
        # delivered before purchase
        ("o_before_purchase", "c7", "delivered", "2018-01-10 10:00:00", "2018-01-10 11:00:00",
         "2018-01-10 12:00:00", "2018-01-09 12:00:00", "2018-01-20 00:00:00"),
        # carrier date before purchase (kept; processing_days NaN + flag)
        ("o_carrier_before", "c8", "delivered", "2018-01-10 10:00:00", "2018-01-10 11:00:00",
         "2018-01-08 12:00:00", "2018-01-15 12:00:00", "2018-01-20 00:00:00"),
        # multi-item, multi-seller, tie on summed price -> smallest seller_id wins
        ("o_multi_tie", "c9", "delivered", "2018-01-10 10:00:00", "2018-01-10 11:00:00",
         "2018-01-11 12:00:00", "2018-01-15 12:00:00", "2018-01-20 00:00:00"),
        # multi-item, representative seller by summed price (not by the single priciest item)
        ("o_multi_sum", "c10", "delivered", "2018-01-10 10:00:00", "2018-01-10 11:00:00",
         "2018-01-11 12:00:00", "2018-01-15 12:00:00", "2018-01-20 00:00:00"),
        # missing category + missing weight
        ("o_nocat", "c11", "delivered", "2018-01-10 10:00:00", "2018-01-10 11:00:00",
         "2018-01-11 12:00:00", "2018-01-15 12:00:00", "2018-01-20 00:00:00"),
        # no item rows at all, unavailable
        ("o_noitems", "c12", "unavailable", "2018-01-10 10:00:00", None, None, None,
         "2018-01-20 00:00:00"),
    ]
    cols = ["order_id", "customer_id", "order_status", "order_purchase_timestamp", "order_approved_at",
            "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date"]
    return [dict(zip(cols, r)) for r in rows]


def _items_rows() -> list[dict]:
    def it(order, n, prod, seller, price, freight=10.0):
        return {"order_id": order, "order_item_id": str(n), "product_id": prod, "seller_id": seller,
                "shipping_limit_date": "2018-01-05 00:00:00", "price": str(price), "freight_value": str(freight)}
    rows = []
    for o in ["o_ontime_2359", "o_late_0001", "o_early", "o_cancelled", "o_shipped", "o_nodate",
              "o_before_purchase", "o_carrier_before"]:
        rows.append(it(o, 1, "p_a", "s_aaa", 100.0))
    rows += [it("o_multi_tie", 1, "p_a", "s_zzz", 50.0), it("o_multi_tie", 2, "p_b", "s_bbb", 50.0)]
    rows += [it("o_multi_sum", 1, "p_a", "s_aaa", 70.0), it("o_multi_sum", 2, "p_b", "s_bbb", 40.0),
             it("o_multi_sum", 3, "p_b", "s_bbb", 40.0)]
    rows += [it("o_nocat", 1, "p_nocat", "s_aaa", 30.0)]
    return rows


@pytest.fixture()
def raw_fixture() -> dict[str, pd.DataFrame]:
    products = pd.DataFrame([
        {"product_id": "p_a", "product_category_name": "cat_a_pt", "product_weight_g": "500",
         "product_length_cm": "10", "product_height_cm": "5", "product_width_cm": "4"},
        {"product_id": "p_b", "product_category_name": "cat_b_pt", "product_weight_g": "1000",
         "product_length_cm": "20", "product_height_cm": "10", "product_width_cm": "8"},
        {"product_id": "p_nocat", "product_category_name": None, "product_weight_g": None,
         "product_length_cm": None, "product_height_cm": None, "product_width_cm": None},
    ])
    sellers = pd.DataFrame([
        {"seller_id": "s_aaa", "seller_zip_code_prefix": "1", "seller_city": "city_a", "seller_state": "SP"},
        {"seller_id": "s_bbb", "seller_zip_code_prefix": "2", "seller_city": "city_b", "seller_state": "RJ"},
        {"seller_id": "s_zzz", "seller_zip_code_prefix": "3", "seller_city": "city_z", "seller_state": "MG"},
    ])
    customers = pd.DataFrame([
        {"customer_id": f"c{i}", "customer_unique_id": f"u{i}", "customer_zip_code_prefix": "1",
         "customer_city": "city_c", "customer_state": "SP" if i % 2 else "RJ"} for i in range(1, 13)
    ])
    translation = pd.DataFrame({"product_category_name": ["cat_a_pt", "cat_b_pt"],
                                "product_category_name_english": ["cat_a", "cat_b"]})
    reviews = pd.DataFrame([
        {"review_id": "r1", "order_id": "o_early", "review_score": "1", "review_creation_date": "2018-01-09 00:00:00",
         "review_answer_timestamp": "2018-01-09 10:00:00"},
        {"review_id": "r2", "order_id": "o_early", "review_score": "5", "review_creation_date": "2018-01-10 00:00:00",
         "review_answer_timestamp": "2018-01-10 10:00:00"},
    ])
    return {
        "orders": pd.DataFrame(_orders_rows()), "items": pd.DataFrame(_items_rows()),
        "customers": customers, "sellers": sellers, "products": products,
        "reviews": reviews, "category_translation": translation,
    }


@pytest.fixture()
def built(raw_fixture):
    orders, report = build_orders_table(raw_fixture)
    return orders.set_index("order_id"), report


# --------------------------------------------------------------------------- #
# Dtype contract (ARCHITECTURE section 2)
# --------------------------------------------------------------------------- #
_TS_COLS = ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
            "order_delivered_customer_date", "order_estimated_delivery_date", "purchase_date"]
_EXPECTED_DTYPES = {
    **{c: "datetime64[ns]" for c in _TS_COLS},
    "order_status": "category", "analysis_eligible": "bool",
    "is_late": "Int8", "delivery_delay_days": "Int32",
    "delivery_duration_days": "float64", "processing_days": "float64",
    "approval_days": "float64", "carrier_to_customer_days": "float64",
    "approval_missing": "int8", "carrier_date_invalid": "int8", "category_missing": "int8",
    "weight_missing": "int8", "same_state": "int8",
    "purchase_year": "Int16", "purchase_month": "Int8", "purchase_quarter": "Int8",
    "purchase_dow": "Int8", "estimated_delivery_days": "Int16",
    "customer_state": "category", "customer_city": "category", "seller_state": "category",
    "seller_city": "category", "state_pair": "category", "product_category": "category",
    "n_items": "int16", "n_sellers": "int16", "n_products": "int16", "n_categories": "int16",
    "total_price": "float64", "total_freight": "float64", "freight_ratio": "float64",
    "total_weight_g": "float64", "total_volume_cm3": "float64", "max_product_length_cm": "float64",
    "max_product_height_cm": "float64", "max_product_width_cm": "float64", "review_score": "float64",
}
_STRING_COLS = ["order_id", "exclusion_reason", "purchase_year_month"]
_CATEGORICAL_COLS = [c for c, t in _EXPECTED_DTYPES.items() if t == "category"]


def _assert_contract_dtypes(df: pd.DataFrame) -> None:
    """Assert every contract column has its documented dtype (so NA/float regressions fail)."""
    got = {c: str(df[c].dtype) for c in _EXPECTED_DTYPES}
    assert got == _EXPECTED_DTYPES
    for c in _STRING_COLS:
        # str (or object holding str / NaN when a purchase timestamp is missing); never categorical
        assert not isinstance(df[c].dtype, pd.CategoricalDtype), c
        assert df[c].dropna().map(type).eq(str).all(), c
    assert set(_EXPECTED_DTYPES) | set(_STRING_COLS) == set(PROCESSED_COLUMNS)


def _assert_no_unused_levels(df: pd.DataFrame) -> None:
    for c in _CATEGORICAL_COLS:
        levels = set(df[c].cat.categories)
        assert levels == set(df[c].dropna().unique()), f"{c} carries unused levels: {levels - set(df[c].dropna())}"


def test_processed_dtypes_match_contract(built):
    _assert_contract_dtypes(built[0].reset_index())


# --------------------------------------------------------------------------- #
# Pure-function tests
# --------------------------------------------------------------------------- #
def _ts_frame(purchase, estimate, customer, status="delivered", carrier=None, approved=None):
    f = lambda x: pd.to_datetime(x) if x is not None else pd.NaT  # noqa: E731
    return pd.DataFrame({
        "order_id": ["x"], "order_status": [status],
        "order_purchase_timestamp": [f(purchase)], "order_approved_at": [f(approved)],
        "order_delivered_carrier_date": [f(carrier)], "order_delivered_customer_date": [f(customer)],
        "order_estimated_delivery_date": [f(estimate)],
    })


def test_calendar_date_lateness_boundaries():
    on_time = add_delivery_targets(_ts_frame("2018-01-02 10:00", "2018-01-12", "2018-01-12 23:59"))
    late = add_delivery_targets(_ts_frame("2018-01-02 10:00", "2018-01-12", "2018-01-13 00:01"))
    assert on_time.loc[0, "is_late"] == 0 and on_time.loc[0, "delivery_delay_days"] == 0
    assert late.loc[0, "is_late"] == 1 and late.loc[0, "delivery_delay_days"] == 1


def test_non_delivered_never_labelled(raw_fixture, built):
    orders, _ = built
    for oid in ["o_cancelled", "o_shipped", "o_noitems"]:
        assert pd.isna(orders.loc[oid, "is_late"])
        assert not orders.loc[oid, "analysis_eligible"]
        assert orders.loc[oid, "exclusion_reason"].startswith("status_")
        assert pd.isna(orders.loc[oid, "delivery_delay_days"])
        assert pd.isna(orders.loc[oid, "delivery_duration_days"])


def test_outcomes_only_for_eligible_rows_even_with_both_dates():
    """Every outcome, incl. carrier_to_customer_days, is NaN for a non-delivered order that has dates."""
    kw = dict(carrier="2018-01-03 09:00", approved="2018-01-02 11:00")
    bad = add_delivery_targets(_ts_frame("2018-01-02 10:00", "2018-01-12", "2018-01-05 12:00",
                                         status="canceled", **kw))
    for col in ["is_late", "delivery_delay_days", "delivery_duration_days", "carrier_to_customer_days"]:
        assert pd.isna(bad.loc[0, col]), col
    assert bad.loc[0, "processing_days"] == pytest.approx(23 / 24)  # retrospective diagnostics stay
    ok = add_delivery_targets(_ts_frame("2018-01-02 10:00", "2018-01-12", "2018-01-05 12:00", **kw))
    assert ok.loc[0, "carrier_to_customer_days"] == pytest.approx(2.0 + 3 / 24)


def test_order_id_unique_and_row_count_preserved(raw_fixture, built):
    orders, _ = built
    assert orders.index.is_unique
    assert len(orders) == len(raw_fixture["orders"])


def test_lateness_end_to_end(built):
    orders, _ = built
    assert orders.loc["o_ontime_2359", "is_late"] == 0
    assert orders.loc["o_late_0001", "is_late"] == 1
    assert orders.loc["o_early", "is_late"] == 0
    assert orders.loc["o_early", "delivery_delay_days"] < 0


def test_delivered_missing_date_and_before_purchase_excluded(built):
    orders, report = built
    assert orders.loc["o_nodate", "exclusion_reason"] == "delivered_missing_delivery_date"
    assert orders.loc["o_before_purchase", "exclusion_reason"] == "delivered_before_purchase"
    assert not orders.loc[["o_nodate", "o_before_purchase"], "analysis_eligible"].any()
    assert pd.isna(orders.loc[["o_nodate", "o_before_purchase"], "is_late"]).all()
    assert report["exclusions"]["delivered_missing_delivery_date"] == 1
    assert report["exclusions"]["delivered_before_purchase"] == 1


def test_carrier_before_purchase_flag_and_nan(built):
    orders, report = built
    row = orders.loc["o_carrier_before"]
    assert row["carrier_date_invalid"] == 1
    assert pd.isna(row["processing_days"])
    assert row["analysis_eligible"] and row["is_late"] == 0  # kept in the analysis set
    assert orders.loc["o_ontime_2359", "carrier_date_invalid"] == 0
    assert orders.loc["o_ontime_2359", "processing_days"] == pytest.approx(23 / 24)
    assert report["eligible_quality"]["carrier_date_invalid"] == 1


def test_multi_item_aggregation_and_representative_seller_tie_break(built):
    orders, report = built
    tie = orders.loc["o_multi_tie"]
    assert tie["n_items"] == 2 and tie["n_sellers"] == 2 and tie["n_products"] == 2
    assert tie["total_price"] == pytest.approx(100.0) and tie["total_freight"] == pytest.approx(20.0)
    assert tie["freight_ratio"] == pytest.approx(0.2)
    # equal summed price (50 vs 50): lexicographically smallest seller_id = s_bbb (RJ) beats s_zzz (MG)
    assert tie["seller_state"] == "RJ"
    # tie on item price -> smallest order_item_id => p_a => cat_a
    assert tie["product_category"] == "cat_a" and tie["n_categories"] == 2
    assert tie["total_weight_g"] == 1500 and tie["total_volume_cm3"] == 10 * 5 * 4 + 20 * 10 * 8
    assert tie["max_product_length_cm"] == 20 and tie["max_product_height_cm"] == 10
    assert tie["max_product_width_cm"] == 8
    # s_bbb sums to 80 > s_aaa 70 although s_aaa owns the single highest-priced item
    s = orders.loc["o_multi_sum"]
    assert s["seller_state"] == "RJ" and s["n_sellers"] == 2 and s["n_items"] == 3
    assert s["n_products"] == 2  # p_b appears twice: distinct products, not item rows
    assert s["product_category"] == "cat_a"  # highest-priced single item (70) is p_a
    assert report["multi_seller_orders"] == 2 and report["multi_category_orders"] == 2


def test_representative_seller_independent_of_row_order(raw_fixture):
    items = raw_fixture["items"]
    kwargs = dict(products=raw_fixture["products"], sellers=raw_fixture["sellers"],
                  translation=raw_fixture["category_translation"])
    a = aggregate_items(items, **kwargs).set_index("order_id")
    b = aggregate_items(items.sample(frac=1, random_state=0), **kwargs).set_index("order_id")
    pd.testing.assert_frame_equal(a.sort_index(), b.sort_index(), check_like=True)


def test_missing_category_and_weight_flags(built):
    orders, _ = built
    row = orders.loc["o_nocat"]
    assert row["product_category"] == "unknown" and row["category_missing"] == 1
    assert row["weight_missing"] == 1 and pd.isna(row["total_weight_g"])
    assert orders.loc["o_early", "category_missing"] == 0 and orders.loc["o_early", "weight_missing"] == 0
    assert orders.loc["o_nocat", "analysis_eligible"]  # kept, not dropped


def test_approval_missing_flag(built):
    orders, report = built
    assert orders.loc["o_cancelled", "approval_missing"] == 1
    assert orders.loc["o_ontime_2359", "approval_missing"] == 0
    assert report["eligible_quality"]["approval_missing"] == 0


def test_orders_without_items_handled_explicitly(built):
    orders, report = built
    row = orders.loc["o_noitems"]
    assert row["n_items"] == 0 and row["n_sellers"] == 0 and row["total_price"] == 0
    assert pd.isna(row["freight_ratio"]) and pd.isna(row["total_weight_g"]) and pd.isna(row["seller_state"])
    assert row["product_category"] == "unknown" and row["same_state"] == 0
    assert row["weight_missing"] == 1 and row["category_missing"] == 1 and row["n_categories"] == 0
    assert report["orders_without_items"]["count"] == 1
    assert report["orders_without_items"]["eligible_among_them"] == 0


def test_delivered_order_without_items_cannot_leak_into_eligible(raw_fixture):
    raw_fixture["items"] = raw_fixture["items"][raw_fixture["items"]["order_id"] != "o_early"]
    orders, report = build_orders_table(raw_fixture)
    row = orders.set_index("order_id").loc["o_early"]
    assert not row["analysis_eligible"] and row["exclusion_reason"] == "delivered_no_items"
    assert report["orders_without_items"]["eligible_among_them"] == 0


def test_review_reduced_to_latest_per_order(built):
    orders, _ = built
    assert orders.loc["o_early", "review_score"] == 5
    assert pd.isna(orders.loc["o_late_0001", "review_score"])


def test_time_and_geo_features(built):
    orders, _ = built
    row = orders.loc["o_ontime_2359"]
    assert row["purchase_year_month"] == "2018-01" and row["purchase_dow"] == 1  # Tuesday
    assert row["purchase_quarter"] == 1 and row["purchase_month"] == 1
    assert row["estimated_delivery_days"] == 10
    assert row["state_pair"] == "SP→SP" and row["same_state"] == 1  # customer c1 is SP, seller s_aaa SP
    assert orders.loc["o_late_0001", "state_pair"] == "SP→RJ" and orders.loc["o_late_0001", "same_state"] == 0


def test_time_and_geo_functions_standalone():
    df = pd.DataFrame({
        "order_purchase_timestamp": pd.to_datetime(["2018-03-31 23:30:00"]),
        "order_estimated_delivery_date": pd.to_datetime(["2018-04-10"]),
        "seller_state": ["SP"], "customer_state": ["RJ"],
    })
    out = add_geo_features(add_time_features(df))
    assert out.loc[0, "estimated_delivery_days"] == 10 and out.loc[0, "purchase_quarter"] == 1
    assert out.loc[0, "state_pair"] == "SP→RJ" and out.loc[0, "same_state"] == 0


def test_no_unexpected_or_identifier_columns(built):
    orders, _ = built
    cols = list(orders.reset_index().columns)
    assert cols == PROCESSED_COLUMNS
    for banned in ["customer_id", "customer_unique_id", "seller_id", "representative_seller_id", "product_id"]:
        assert banned not in cols
    # every leakage column that exists must be one the contract lists in the schema (only outcomes, never features)
    assert set(config.LEAKAGE_COLUMNS) <= set(PROCESSED_COLUMNS)
    assert set(config.PURCHASE_TIME_FEATURES) <= set(PROCESSED_COLUMNS)
    assert not set(config.PURCHASE_TIME_FEATURES) & set(config.LEAKAGE_COLUMNS)


def test_validate_processed_clean(built):
    orders, _ = built
    assert validate_processed(orders.reset_index()) == []


def test_validate_processed_catches_injected_problems(built):
    clean = built[0].reset_index()

    dup = clean.copy()
    dup.loc[1, "order_id"] = dup.loc[0, "order_id"]
    assert any("not unique" in p for p in validate_processed(dup))

    cancelled = clean.copy()
    cancelled.loc[cancelled["order_id"] == "o_cancelled", "is_late"] = 0
    assert any("ineligible rows labelled" in p for p in validate_processed(cancelled))

    leak = clean.copy()
    leak["customer_id"] = "abc"
    assert any("unexpected columns" in p for p in validate_processed(leak))

    dropped = clean.drop(columns=["is_late"])
    assert any("missing columns" in p for p in validate_processed(dropped))

    wrong = clean.copy()
    idx = wrong.index[wrong["order_id"] == "o_late_0001"][0]
    wrong.loc[idx, "is_late"] = 0
    assert any("inconsistent" in p for p in validate_processed(wrong))

    neg = clean.copy()
    neg.loc[0, "total_price"] = -5.0
    assert any("negative values in total_price" in p for p in validate_processed(neg))

    flag = clean.copy()
    flag.loc[0, "carrier_date_invalid"] = 3
    assert any("carrier_date_invalid" in p for p in validate_processed(flag))

    bad_elig = clean.copy()
    bad_elig.loc[bad_elig["order_id"] == "o_shipped", "analysis_eligible"] = True
    problems = validate_processed(bad_elig)
    assert any("status is not 'delivered'" in p for p in problems)
    assert any("exclusion_reason must be empty" in p for p in problems)


def _set(df, oid, **cols):
    df = df.copy()
    idx = df.index[df["order_id"] == oid][0]
    for k, v in cols.items():
        df[k] = df[k].astype("object") if str(df[k].dtype) == "category" else df[k]
        df.loc[idx, k] = v
    return df


_ELIG = "o_late_0001"      # eligible, late
_INELIG = "o_shipped"      # ineligible
_CORRUPTIONS = [
    # (id, order_id, column changes, expected message substring)
    ("status_not_delivered", _ELIG, {"order_status": "shipped"}, "status is not 'delivered'"),
    ("status_null", _ELIG, {"order_status": None}, "status is not 'delivered'"),
    ("missing_purchase", _ELIG, {"order_purchase_timestamp": pd.NaT}, "missing order_purchase_timestamp"),
    ("missing_estimate", _ELIG, {"order_estimated_delivery_date": pd.NaT},
     "missing order_estimated_delivery_date"),
    ("missing_customer_date", _ELIG, {"order_delivered_customer_date": pd.NaT},
     "missing order_delivered_customer_date"),
    ("reason_on_eligible", _ELIG, {"exclusion_reason": "status_x"}, "exclusion_reason must be empty"),
    ("delay_on_ineligible", _INELIG, {"delivery_delay_days": 3}, "ineligible rows carry delivery_delay_days"),
    ("duration_missing", _ELIG, {"delivery_duration_days": float("nan")}, "missing delivery_duration_days"),
    ("duration_negative", _ELIG, {"delivery_duration_days": -1.0}, "negative delivery_duration_days"),
    ("seller_state_unknown", _ELIG, {"seller_state": None}, "unknown seller/customer state"),
    ("customer_state_unknown", _ELIG, {"customer_state": None}, "unknown seller/customer state"),
    ("null_order_id", _ELIG, {"order_id": None}, "order_id contains nulls"),
    ("same_state_bad", _ELIG, {"same_state": 2}, "same_state has values outside"),
    ("eligible_no_items", _ELIG, {"n_items": 0}, "eligible rows with no items"),
    ("category_missing", _ELIG, {"product_category": None}, "missing product_category"),
    ("delivered_before_purchase", _ELIG,
     {"order_delivered_customer_date": pd.Timestamp("2017-01-01")}, "delivered before purchase"),
    ("is_late_out_of_range", _ELIG, {"is_late": 2}, "is_late has values outside"),
    ("is_late_missing", _ELIG, {"is_late": pd.NA}, "eligible rows with missing is_late"),
    ("too_many_categories", _ELIG, {"n_categories": 99}, "n_categories / n_sellers exceed n_items"),
    ("too_many_sellers", _ELIG, {"n_sellers": 99}, "n_categories / n_sellers exceed n_items"),
]


@pytest.mark.parametrize("oid,changes,msg", [c[1:] for c in _CORRUPTIONS], ids=[c[0] for c in _CORRUPTIONS])
def test_validate_processed_each_guard_fires(built, oid, changes, msg):
    clean = built[0].reset_index()
    assert validate_processed(clean) == []
    problems = validate_processed(_set(clean, oid, **changes))
    assert any(msg in p for p in problems), problems


def test_null_status_is_never_eligible_or_labelled(raw_fixture):
    """A null order_status must fall into status_missing, not slip through as eligible."""
    raw_fixture["orders"].loc[raw_fixture["orders"]["order_id"] == "o_late_0001", "order_status"] = None
    orders, _ = build_orders_table(raw_fixture)
    row = orders.set_index("order_id").loc["o_late_0001"]
    assert not row["analysis_eligible"]
    assert row["exclusion_reason"] == "status_missing"
    assert pd.isna(row["is_late"]) and pd.isna(row["delivery_delay_days"])
    # and directly on the pure function
    df = _ts_frame("2018-01-02 10:00", "2018-01-12", "2018-01-13 00:01", status=None)
    out = add_delivery_targets(df)
    assert not out.loc[0, "analysis_eligible"] and out.loc[0, "exclusion_reason"] == "status_missing"
    assert pd.isna(out.loc[0, "is_late"])


def test_validate_flags_eligible_row_with_null_status(built):
    bad = _set(built[0].reset_index(), "o_late_0001", order_status=None)
    assert any("status is not 'delivered'" in p for p in validate_processed(bad))


def test_output_sorted_chronologically_and_independent_of_raw_row_order(raw_fixture):
    # force ties on the purchase timestamp (the fixture already has several)
    a, _ = build_orders_table(raw_fixture)
    shuffled = {k: v.sample(frac=1.0, random_state=7).reset_index(drop=True) for k, v in raw_fixture.items()}
    b, _ = build_orders_table(shuffled)
    ts = a["order_purchase_timestamp"]
    assert ts.is_monotonic_increasing
    assert a["order_id"].duplicated().sum() == 0 and ts.duplicated().any()  # ties exist
    ties = a[ts.duplicated(keep=False)]
    assert ties.groupby("order_purchase_timestamp")["order_id"].apply(lambda s: s.is_monotonic_increasing).all()
    assert a["order_id"].equals(b["order_id"])
    pd.testing.assert_frame_equal(a, b)


def test_duplicates_are_removed_and_reported(raw_fixture):
    raw_fixture["orders"] = pd.concat([raw_fixture["orders"], raw_fixture["orders"].iloc[[0]]], ignore_index=True)
    raw_fixture["items"] = pd.concat([raw_fixture["items"], raw_fixture["items"].iloc[[0]]], ignore_index=True)
    orders, report = build_orders_table(raw_fixture)
    assert orders["order_id"].is_unique and len(orders) == 12
    assert report["raw_files"]["orders"]["duplicates"]["exact_duplicate_rows"] == 1
    assert report["raw_files"]["items"]["duplicates"]["exact_duplicates_removed"] == 1
    assert orders.set_index("order_id").loc["o_ontime_2359", "n_items"] == 1  # not double counted


def test_unparseable_timestamps_are_counted_and_not_dropped(raw_fixture):
    raw_fixture["orders"].loc[0, "order_delivered_customer_date"] = "not-a-date"
    orders, report = build_orders_table(raw_fixture)
    assert report["timestamp_parsing"]["order_delivered_customer_date"]["unparseable"] == 1
    row = orders.set_index("order_id").loc["o_ontime_2359"]
    assert row["exclusion_reason"] == "delivered_missing_delivery_date" and pd.isna(row["is_late"])
    assert len(orders) == 12


def test_report_is_json_serialisable_and_consistent(built):
    import json
    _, report = built
    json.dumps(report)
    assert report["n_orders_all"] == 12
    assert report["n_eligible"] == 7  # on-time, late, early, carrier_before, multi_tie, multi_sum, nocat
    assert report["late_rate_eligible"] == pytest.approx(1 / 7)
    assert report["logical_order_anomalies_all_orders"]["carrier_before_purchase"] == 1
    assert report["logical_order_anomalies_all_orders"]["customer_delivery_before_purchase"] == 1
    assert report["n_late_eligible"] == 1
    # eligible rows: on-time, late, early, carrier_before, multi_tie, multi_sum, nocat
    assert report["eligible_quality"] == {
        "carrier_date_invalid": 1, "approval_missing": 0, "category_missing": 1, "weight_missing": 1,
        "carrier_to_customer_negative_set_nan": 0, "review_score_missing": 6,  # only o_early has a review
    }
    assert report["date_range_all"]["min"] == "2018-01-02T10:00:00"
    assert report["date_range_all"]["max"] == "2018-01-10T10:00:00"


def test_py_serialises_nat_as_none():
    assert preprocessing._py(pd.NaT) is None
    assert preprocessing._py(pd.Series([pd.NaT], dtype="datetime64[ns]").min()) is None
    assert preprocessing._py(pd.Series([], dtype="datetime64[ns]").max()) is None
    assert preprocessing._py(pd.Timestamp("2018-01-02 10:00:00")) == "2018-01-02T10:00:00"
    assert preprocessing._py({"a": [pd.NaT, pd.Timestamp("2018-01-02")]}) == {"a": [None, "2018-01-02T00:00:00"]}


def test_missing_raw_files_raise_helpful_error(tmp_path):
    with pytest.raises(MissingDatasetError) as exc:
        load_raw(tmp_path)
    msg = str(exc.value)
    assert config.KAGGLE_URL in msg and str(tmp_path) in msg and "olist_orders_dataset.csv" in msg


def test_load_processed_missing_file_gives_command(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        load_processed(tmp_path / "nope.parquet")
    assert "python -m src.preprocessing" in str(exc.value)


# --------------------------------------------------------------------------- #
# Integration tests on the REAL processed data (skipped when not generated)
# --------------------------------------------------------------------------- #
needs_real = pytest.mark.skipif(
    not (config.ORDERS_ALL_PATH.is_file() and config.ORDERS_DELIVERED_PATH.is_file()
         and (config.RAW_DIR / config.REQUIRED_RAW_FILES["orders"]).is_file()),
    reason="processed parquet / raw orders CSV not present; run `.venv/bin/python -m src.preprocessing`",
)


@pytest.fixture(scope="module")
def real_all():
    return pd.read_parquet(config.ORDERS_ALL_PATH)


@pytest.fixture(scope="module")
def real_delivered():
    return load_processed()


@needs_real
def test_real_unique_ids_and_row_counts(real_all, real_delivered):
    raw_orders = pd.read_csv(config.RAW_DIR / config.REQUIRED_RAW_FILES["orders"], dtype=str)
    assert real_all["order_id"].is_unique and real_delivered["order_id"].is_unique
    assert len(real_all) == len(raw_orders) == raw_orders["order_id"].nunique()
    assert set(real_delivered["order_id"]) <= set(real_all["order_id"])


@needs_real
def test_real_eligible_count_matches_raw_definition(real_all, real_delivered):
    raw = pd.read_csv(config.RAW_DIR / config.REQUIRED_RAW_FILES["orders"], dtype=str)
    ts = {c: pd.to_datetime(raw[c], errors="coerce") for c in config.TIMESTAMP_COLS}
    expected = ((raw["order_status"] == "delivered") & ts["order_purchase_timestamp"].notna()
                & ts["order_estimated_delivery_date"].notna() & ts["order_delivered_customer_date"].notna()
                & (ts["order_delivered_customer_date"] >= ts["order_purchase_timestamp"]))
    assert len(real_delivered) == int(expected.sum()) == int(real_all["analysis_eligible"].sum())
    assert (real_delivered["order_status"].astype(str) == "delivered").all()
    assert (real_delivered["exclusion_reason"] == "").all()


@needs_real
def test_real_is_late_consistent_with_delay_days(real_delivered):
    assert real_delivered["is_late"].notna().all()
    assert ((real_delivered["delivery_delay_days"] > 0).astype(int) == real_delivered["is_late"].astype(int)).all()
    assert 0.04 < real_delivered["is_late"].astype(float).mean() < 0.12


@needs_real
def test_real_ineligible_never_labelled_and_no_item_orders_excluded(real_all):
    inel = real_all[~real_all["analysis_eligible"]]
    assert inel["is_late"].isna().all() and (inel["exclusion_reason"] != "").all()
    no_items = real_all[real_all["n_items"] == 0]
    assert len(no_items) > 0 and not no_items["analysis_eligible"].any()


@needs_real
def test_real_validate_processed_clean_and_schema(real_all, real_delivered):
    assert validate_processed(real_all) == []
    assert validate_processed(real_delivered) == []
    assert list(real_all.columns) == PROCESSED_COLUMNS
    assert set(config.PURCHASE_TIME_FEATURES) <= set(real_delivered.columns)
    assert real_delivered[config.PURCHASE_TIME_NUMERIC + config.PURCHASE_TIME_CATEGORICAL].drop(
        columns=["total_weight_g", "total_volume_cm3", "max_product_length_cm", "max_product_height_cm",
                 "max_product_width_cm", "freight_ratio"]).notna().all().all()


@needs_real
def test_real_delivered_csv_matches_parquet(real_delivered):
    csv = pd.read_csv(config.ORDERS_DELIVERED_CSV, usecols=["order_id", "is_late"])
    assert len(csv) == len(real_delivered)
    assert set(csv["order_id"]) == set(real_delivered["order_id"])


# --------------------------------------------------------------------------- #
# Edge-case fixtures (separate from raw_fixture so existing counts stay stable)
# --------------------------------------------------------------------------- #
P0 = "2018-01-10 10:00:00"
_EDGE_COLS = ["order_id", "customer_id", "order_status", "order_purchase_timestamp", "order_approved_at",
              "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date"]


def _edge_raw() -> dict[str, pd.DataFrame]:
    """order_id, status, approved, carrier, customer, estimate (purchase is always P0)"""
    D = "delivered"
    spec = {
        "e_c2c_neg":      (D, "2018-01-10 11:00:00", "2018-01-14 12:00:00", "2018-01-13 12:00:00", "2018-01-20 00:00:00"),
        "e_appr_neg":     (D, "2018-01-09 11:00:00", "2018-01-11 12:00:00", "2018-01-15 12:00:00", "2018-01-20 00:00:00"),
        "e_appr_missing": (D, None, "2018-01-11 12:00:00", "2018-01-15 12:00:00", "2018-01-20 00:00:00"),
        "e_partial_w":    (D, "2018-01-10 11:00:00", "2018-01-11 12:00:00", "2018-01-15 12:00:00", "2018-01-20 00:00:00"),
        "e_zero_price":   (D, "2018-01-10 11:00:00", "2018-01-11 12:00:00", "2018-01-15 12:00:00", "2018-01-20 00:00:00"),
        "e_no_estimate":  (D, "2018-01-10 11:00:00", "2018-01-11 12:00:00", "2018-01-15 12:00:00", None),
        "e_no_est_no_cust": (D, "2018-01-10 11:00:00", "2018-01-11 12:00:00", None, None),
        "e_carrier_eq_p": (D, "2018-01-10 10:00:00", P0, "2018-01-15 12:00:00", "2018-01-20 00:00:00"),
        "e_cust_eq_p":    (D, "2018-01-10 10:00:00", P0, P0, "2018-01-20 00:00:00"),
        "e_zero_weight":  (D, "2018-01-10 11:00:00", "2018-01-11 12:00:00", "2018-01-15 12:00:00", "2018-01-20 00:00:00"),
        "e_supp":         (D, "2018-01-10 11:00:00", "2018-01-11 12:00:00", "2018-01-15 12:00:00", "2018-01-20 00:00:00"),
    }
    orders = [dict(zip(_EDGE_COLS, [o, f"ec_{o}", st, P0, ap, ca, cu, es])) for o, (st, ap, ca, cu, es) in spec.items()]

    def it(order, n, prod, price, freight=10.0):
        return {"order_id": order, "order_item_id": str(n), "product_id": prod, "seller_id": "s_aaa",
                "shipping_limit_date": "2018-01-05 00:00:00", "price": str(price), "freight_value": str(freight)}
    items = [it(o, 1, "p_a", 100.0) for o in spec if o not in
             ("e_partial_w", "e_zero_price", "e_zero_weight", "e_supp")]
    items += [it("e_partial_w", 1, "p_a", 100.0), it("e_partial_w", 2, "p_nocat", 5.0),
              it("e_zero_price", 1, "p_a", 0.0, 10.0),
              it("e_zero_weight", 1, "p_zero", 20.0),
              it("e_supp", 1, "p_supp", 20.0), it("e_supp", 2, "p_gamer", 10.0)]

    def prod(pid, cat, w="500", l="10", h="5", wd="4"):
        return {"product_id": pid, "product_category_name": cat, "product_weight_g": w,
                "product_length_cm": l, "product_height_cm": h, "product_width_cm": wd}
    products = pd.DataFrame([
        prod("p_a", "cat_a_pt"),
        prod("p_nocat", None, None, None, None, None),
        prod("p_zero", "cat_a_pt", w="0"),
        prod("p_supp", "portateis_cozinha_e_preparadores_de_alimentos"),
        prod("p_gamer", "pc_gamer"),
    ])
    sellers = pd.DataFrame([{"seller_id": "s_aaa", "seller_zip_code_prefix": "1",
                             "seller_city": "city_a", "seller_state": "SP"}])
    customers = pd.DataFrame([{"customer_id": f"ec_{o}", "customer_unique_id": f"eu_{o}",
                               "customer_zip_code_prefix": "1", "customer_city": "city_c",
                               "customer_state": "RJ"} for o in spec])
    translation = pd.DataFrame({"product_category_name": ["cat_a_pt"], "product_category_name_english": ["cat_a"]})
    return {"orders": pd.DataFrame(orders), "items": pd.DataFrame(items), "customers": customers,
            "sellers": sellers, "products": products, "category_translation": translation}


@pytest.fixture()
def edge():
    orders, report = build_orders_table(_edge_raw())
    return orders.set_index("order_id"), report


def test_negative_carrier_to_customer_gap_is_nan(edge):
    orders, report = edge
    row = orders.loc["e_c2c_neg"]
    assert row["analysis_eligible"] and pd.isna(row["carrier_to_customer_days"])
    assert row["processing_days"] == pytest.approx(4.0 + 2 / 24)  # 01-10 10:00 -> 01-14 12:00
    assert report["eligible_quality"]["carrier_to_customer_negative_set_nan"] == 1
    # a normal order keeps its (positive) gap
    assert orders.loc["e_appr_neg", "carrier_to_customer_days"] == pytest.approx(4.0)


def test_negative_approval_gap_is_nan_but_not_missing(edge):
    orders, _ = edge
    row = orders.loc["e_appr_neg"]
    assert pd.isna(row["approval_days"]) and row["approval_missing"] == 0
    assert orders.loc["e_appr_missing", "approval_missing"] == 1
    assert pd.isna(orders.loc["e_appr_missing", "approval_days"])
    assert orders.loc["e_c2c_neg", "approval_days"] == pytest.approx(1 / 24)


def test_partly_missing_weight_makes_order_total_nan(edge):
    orders, _ = edge
    row = orders.loc["e_partial_w"]
    assert row["n_items"] == 2
    assert pd.isna(row["total_weight_g"]) and pd.isna(row["total_volume_cm3"])  # not an understated 500
    assert pd.isna(row["max_product_length_cm"]) and pd.isna(row["max_product_height_cm"])
    assert pd.isna(row["max_product_width_cm"])
    assert row["weight_missing"] == 1
    # representative category is the priciest item's (known), so category_missing stays 0
    assert row["product_category"] == "cat_a" and row["category_missing"] == 0


def test_n_categories_ignores_unknown(edge):
    orders, _ = edge
    # cat_a + unknown -> one *known* category
    assert orders.loc["e_partial_w", "n_categories"] == 1
    assert orders.loc["e_partial_w", "n_items"] == 2


def test_zero_price_order_has_nan_freight_ratio(edge):
    orders, _ = edge
    row = orders.loc["e_zero_price"]
    assert row["total_price"] == 0 and row["total_freight"] == 10
    assert pd.isna(row["freight_ratio"])  # never inf
    assert orders.loc["e_appr_neg", "freight_ratio"] == pytest.approx(0.1)


def test_zero_weight_product_is_treated_as_missing(edge):
    orders, report = edge
    row = orders.loc["e_zero_weight"]
    assert pd.isna(row["total_weight_g"]) and row["weight_missing"] == 1
    assert row["analysis_eligible"]
    pq = report["product_quality"]
    assert pq["zero_weight_products"] == 1 and pq["products_missing_weight"] == 1
    assert pq["orders_with_nonpositive_attribute"] == 1 and pq["item_rows_with_nonpositive_attribute"] == 1
    assert orders["total_weight_g"].dropna().gt(0).all()  # no physically impossible 0 g survives


def test_missing_estimate_excluded_with_precedence(edge):
    orders, report = edge
    for oid in ["e_no_estimate", "e_no_est_no_cust"]:
        assert not orders.loc[oid, "analysis_eligible"]
        assert orders.loc[oid, "exclusion_reason"] == "missing_purchase_or_estimate"
        assert pd.isna(orders.loc[oid, "is_late"])
    assert report["exclusions"]["missing_purchase_or_estimate"] == 2


def test_timestamp_equality_boundaries(edge):
    orders, _ = edge
    carrier_eq = orders.loc["e_carrier_eq_p"]
    assert carrier_eq["carrier_date_invalid"] == 0 and carrier_eq["processing_days"] == 0
    cust_eq = orders.loc["e_cust_eq_p"]  # delivered the instant of purchase: not "before purchase"
    assert cust_eq["analysis_eligible"] and cust_eq["exclusion_reason"] == ""
    assert cust_eq["delivery_duration_days"] == 0 and cust_eq["is_late"] == 0
    # zero gaps are real values, not NaN (approved == purchase; carrier == customer)
    assert cust_eq["approval_days"] == 0 and cust_eq["carrier_to_customer_days"] == 0
    assert carrier_eq["approval_days"] == 0


def test_supplemental_translations_applied(edge):
    orders, report = edge
    # priciest item is p_supp -> supplemented English name; pc_gamer is the second category
    row = orders.loc["e_supp"]
    assert row["product_category"] == "portable_kitchen_food_preparers" and row["n_categories"] == 2
    tr = report["category_translation"]
    assert set(tr["supplemental_translations_applied"]) == set(preprocessing.SUPPLEMENTAL_TRANSLATIONS)
    assert tr["still_untranslated"] == []


# --------------------------------------------------------------------------- #
# "Full" fixture: edge fixture + aggregation / category / report / precedence cases
# --------------------------------------------------------------------------- #
def _full_raw() -> dict[str, pd.DataFrame]:
    raw = _edge_raw()
    D, S = "delivered", "shipped"
    Q = "2018-01-10 11:00:00"   # approved an hour after P0
    C, U, E = "2018-01-11 12:00:00", "2018-01-15 12:00:00", "2018-01-20 00:00:00"
    #  order_id: (status, purchase, approved, carrier, customer, estimate, customer_id or None)
    spec = {
        "e_dup_prod": (D, P0, Q, C, U, E, None),
        "e_big": (D, P0, Q, C, U, E, None),
        "e_nodim": (D, P0, Q, C, U, E, None),
        "e_w_only": (D, P0, Q, C, U, E, None),
        "e_blank": (D, P0, Q, C, U, E, None),
        "e_empty": (D, P0, Q, C, U, E, None),
        "e_untr": (D, P0, Q, C, U, E, None),
        "e_carrier_before_appr": (D, P0, "2018-01-12 10:00:00", C, U, E, None),   # carrier < approval
        "e_late": (D, P0, Q, C, "2018-01-21 12:00:00", E, None),                  # 1 calendar day late
        "e_early_purchase": (D, "2017-12-30 10:00:00", "2017-12-30 11:00:00", "2017-12-31 12:00:00",
                             "2018-01-05 12:00:00", E, None),
        "e_orphan_cust": (S, P0, Q, C, None, E, "ec_ghost"),                      # no such customer
        "e_orphan_seller": (S, "2018-02-01 10:00:00", "2018-02-01 11:00:00", "2018-01-31 09:00:00",
                            None, "2018-02-20 00:00:00", None),                   # seller not in table
        # delivered, item-less AND delivered before purchase -> delivered_before_purchase wins
        "e_noitems_before": (D, "2017-12-25 10:00:00", "2017-12-25 11:00:00", "2017-12-26 12:00:00",
                             "2017-12-24 12:00:00", E, None),
        "e_no_purchase": (D, None, Q, C, U, E, None),
    }
    orders = [dict(zip(_EDGE_COLS, [o, cid or f"ec_{o}", st, pu, ap, ca, cu, es]))
              for o, (st, pu, ap, ca, cu, es, cid) in spec.items()]
    raw["orders"] = pd.concat([raw["orders"], pd.DataFrame(orders)], ignore_index=True)

    def it(order, n, prod, price, seller="s_aaa", freight=10.0):
        return {"order_id": order, "order_item_id": str(n), "product_id": prod, "seller_id": seller,
                "shipping_limit_date": "2018-01-05 00:00:00", "price": str(price), "freight_value": str(freight)}
    items = [it("e_dup_prod", 1, "p_a", 10.0), it("e_dup_prod", 2, "p_a", 10.0),   # same product twice
             it("e_big", 1, "p_a", 5.0), it("e_big", 2, "p_big", 50.0),
             it("e_nodim", 1, "p_a", 5.0), it("e_nodim", 2, "p_wnd", 5.0),
             it("e_w_only", 1, "p_wnd", 5.0),
             it("e_blank", 1, "p_blank", 5.0), it("e_empty", 1, "p_empty", 5.0),
             it("e_untr", 1, "p_untr", 5.0),
             it("e_orphan_seller", 1, "p_a", 5.0, seller="s_ghost")]
    items += [it(o, 1, "p_a", 5.0) for o in spec
              if o not in {"e_dup_prod", "e_big", "e_nodim", "e_w_only", "e_blank", "e_empty", "e_untr",
                           "e_orphan_seller", "e_noitems_before"}]
    raw["items"] = pd.concat([raw["items"], pd.DataFrame(items)], ignore_index=True)

    def prod(pid, cat, w="500", l="10", h="5", wd="4"):
        return {"product_id": pid, "product_category_name": cat, "product_weight_g": w,
                "product_length_cm": l, "product_height_cm": h, "product_width_cm": wd}
    raw["products"] = pd.concat([raw["products"], pd.DataFrame([
        prod("p_big", "cat_a_pt", w="700", l="30", h="2", wd="20"),
        prod("p_wnd", "cat_a_pt", w="300", l=None, h=None, wd=None),   # weight but no dimensions
        prod("p_blank", "   "), prod("p_empty", ""), prod("p_untr", "cat_unlisted_pt"),
    ])], ignore_index=True)

    cust = [{"customer_id": f"ec_{o}", "customer_unique_id": f"eu_{o}", "customer_zip_code_prefix": "1",
             "customer_city": "city_c", "customer_state": "RJ"} for o, v in spec.items() if v[6] is None]
    raw["customers"] = pd.concat([raw["customers"], pd.DataFrame(cust)], ignore_index=True)
    raw["reviews"] = pd.DataFrame([{"review_id": "r_late", "order_id": "e_late", "review_score": "4",
                                    "review_creation_date": "2018-01-22 00:00:00",
                                    "review_answer_timestamp": "2018-01-22 10:00:00"}])
    return raw


@pytest.fixture()
def full():
    orders, report = build_orders_table(_full_raw())
    return orders.set_index("order_id"), report


def test_full_fixture_dtypes_and_validation(full):
    orders, _ = full
    _assert_contract_dtypes(orders.reset_index())
    assert validate_processed(orders.reset_index()) == []
    assert len(orders) == 25 and int(orders["analysis_eligible"].sum()) == 19


def test_aggregate_max_dimensions_and_repeated_product(full):
    orders, _ = full
    big = orders.loc["e_big"]  # p_a (10x5x4, 500 g) + p_big (30x2x20, 700 g): max, not min / first
    assert big["max_product_length_cm"] == 30 and big["max_product_height_cm"] == 5
    assert big["max_product_width_cm"] == 20 and big["n_products"] == 2
    assert big["total_weight_g"] == 1200 and big["total_volume_cm3"] == 10 * 5 * 4 + 30 * 2 * 20
    dup = orders.loc["e_dup_prod"]  # same product on two item rows
    assert dup["n_items"] == 2 and dup["n_products"] == 1 and dup["n_sellers"] == 1
    assert dup["total_weight_g"] == 1000 and dup["max_product_length_cm"] == 10
    # one item lacks dimensions: every max_* and the volume total is NaN, weight is still summed
    nd = orders.loc["e_nodim"]
    assert nd["total_weight_g"] == 800 and pd.isna(nd["total_volume_cm3"])
    assert all(pd.isna(nd[c]) for c in ["max_product_length_cm", "max_product_height_cm", "max_product_width_cm"])
    assert nd["weight_missing"] == 1
    # weight present but dimensions (hence volume) missing still sets weight_missing
    wo = orders.loc["e_w_only"]
    assert wo["total_weight_g"] == 300 and pd.isna(wo["total_volume_cm3"]) and wo["weight_missing"] == 1


def test_category_edge_cases(full):
    orders, report = full
    for oid in ["e_blank", "e_empty"]:  # whitespace-only / empty names are unknown, not their own category
        assert orders.loc[oid, "product_category"] == "unknown" and orders.loc[oid, "category_missing"] == 1
        assert orders.loc[oid, "n_categories"] == 0
    # a name absent from a *present* translation table keeps its Portuguese name
    assert orders.loc["e_untr", "product_category"] == "cat_unlisted_pt"
    assert orders.loc["e_untr", "category_missing"] == 0 and orders.loc["e_untr", "n_categories"] == 1
    assert "cat_unlisted_pt" in report["category_translation"]["still_untranslated"]


def test_missing_purchase_excluded_and_precedence(full):
    orders, report = full
    row = orders.loc["e_no_purchase"]  # delivered, has estimate + customer date, but no purchase timestamp
    assert not row["analysis_eligible"] and row["exclusion_reason"] == "missing_purchase_or_estimate"
    assert pd.isna(row["is_late"]) and pd.isna(row["delivery_delay_days"]) and pd.isna(row["estimated_delivery_days"])
    # item-less AND delivered before purchase: the documented precedence puts before_purchase first
    row = orders.loc["e_noitems_before"]
    assert row["n_items"] == 0 and row["exclusion_reason"] == "delivered_before_purchase"
    assert report["exclusions"]["delivered_no_items"] == 0


def test_report_counters_hand_computed(full):
    _, r = full
    assert r["n_orders_raw"] == r["n_orders_all"] == 25
    assert r["customer_join"]["orders_without_matching_customer"] == 1          # e_orphan_cust
    assert r["referential_integrity"]["item_rows_with_unknown_seller"] == 1     # e_orphan_seller
    assert r["referential_integrity"]["item_rows_with_unknown_order"] == 0
    # all orders: e_appr_neg approved before purchase; e_orphan_seller carrier before purchase;
    # e_c2c_neg + e_noitems_before customer before carrier; e_noitems_before customer before purchase;
    # e_carrier_before_appr + e_orphan_seller carrier before approval
    assert r["logical_order_anomalies_all_orders"] == {
        "approved_before_purchase": 1, "carrier_before_purchase": 1, "customer_delivery_before_carrier": 2,
        "customer_delivery_before_purchase": 1, "carrier_before_approval": 2}
    # delivered orders only: e_orphan_seller / e_orphan_cust are 'shipped'
    assert r["logical_order_anomalies_delivered_orders"] == {
        "approved_before_purchase": 1, "carrier_before_purchase": 0, "customer_delivery_before_carrier": 2,
        "customer_delivery_before_purchase": 1, "carrier_before_approval": 1}
    assert r["order_status_counts"] == {"delivered": 23, "shipped": 2}
    # exclusion reasons count ineligible rows only (6 of 25); eligible rows ("") never appear
    assert r["exclusion_reason_counts"] == {"missing_purchase_or_estimate": 3, "status_shipped": 2,
                                            "delivered_before_purchase": 1}
    assert r["exclusions"] == {"non_delivered_status": 2, "delivered_missing_delivery_date": 0,
                               "delivered_before_purchase": 1, "delivered_no_items": 0,
                               "missing_purchase_or_estimate": 3}
    assert r["orders_without_items"] == {"count": 1, "status_counts": {"delivered": 1},
                                         "eligible_among_them": 0, "handling": r["orders_without_items"]["handling"]}
    assert r["multi_item_orders"] == 5 and r["multi_seller_orders"] == 0
    assert r["n_eligible"] == 19 and r["n_late_eligible"] == 1 and r["late_rate_eligible"] == pytest.approx(1 / 19)
    assert r["eligible_quality"] == {
        "carrier_date_invalid": 0, "approval_missing": 1,          # e_appr_missing
        "category_missing": 2,                                      # e_blank, e_empty
        "weight_missing": 4,                                        # e_partial_w, e_zero_weight, e_nodim, e_w_only
        "carrier_to_customer_negative_set_nan": 1,                  # e_c2c_neg (e_noitems_before is ineligible)
        "review_score_missing": 18,                                 # only e_late has a review
    }
    assert r["date_range_all"] == {"min": "2017-12-25T10:00:00", "max": "2018-02-01T10:00:00"}
    assert r["date_range_eligible"] == {"min": "2017-12-30T10:00:00", "max": "2018-01-10T10:00:00"}


def test_duplicate_key_with_different_content_keeps_first(raw_fixture):
    dup = raw_fixture["orders"].iloc[[0]].copy()
    dup["order_status"] = "canceled"
    raw_fixture["orders"] = pd.concat([raw_fixture["orders"], dup], ignore_index=True)
    orders, report = build_orders_table(raw_fixture)
    dups = report["raw_files"]["orders"]["duplicates"]
    assert dups["exact_duplicate_rows"] == 0 and dups["duplicate_key_removed"] == 1
    assert len(orders) == 12 and orders["order_id"].is_unique
    assert orders.set_index("order_id").loc["o_ontime_2359", "order_status"] == "delivered"


def test_optional_files_absent(raw_fixture):
    raw_fixture.pop("reviews")
    raw_fixture.pop("category_translation")
    orders, report = build_orders_table(raw_fixture)
    assert orders["review_score"].isna().all()
    indexed = orders.set_index("order_id")
    assert indexed.loc["o_early", "product_category"] == "cat_a_pt"  # Portuguese name kept
    assert "absent" in report["category_translation"]["note"]
    assert set(report["category_translation"]["still_untranslated"]) == {"cat_a_pt", "cat_b_pt"}


def test_build_raises_when_validation_fails(raw_fixture, monkeypatch):
    monkeypatch.setattr(preprocessing, "validate_processed", lambda df: ["injected problem"])
    with pytest.raises(ValueError, match="injected problem"):
        build_orders_table(raw_fixture)


# --------------------------------------------------------------------------- #
# Leakage-guard configuration
# --------------------------------------------------------------------------- #
def test_every_processed_column_is_classified_and_leaky_flags_are_guarded():
    groups = {
        "features": set(config.PURCHASE_TIME_FEATURES), "leakage": set(config.LEAKAGE_COLUMNS),
        "non_feature": set(config.NON_FEATURE_COLUMNS), "unused": set(config.PURCHASE_TIME_UNUSED),
    }
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not groups[a] & groups[b], f"{a} and {b} overlap"
    assert set(PROCESSED_COLUMNS) == set().union(*groups.values())
    # derived from post-purchase timestamps -> must be blocked as predictors
    assert {"carrier_date_invalid", "approval_missing"} <= groups["leakage"]


# --------------------------------------------------------------------------- #
# run_pipeline / CLI on a tmp directory
# --------------------------------------------------------------------------- #
def _write_raw(raw_fixture, raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = {**config.REQUIRED_RAW_FILES, **config.OPTIONAL_RAW_FILES}
    for name, df in raw_fixture.items():
        df.to_csv(raw_dir / files[name], index=False)


def test_run_pipeline_writes_consistent_artifacts_and_is_idempotent(raw_fixture, tmp_path):
    raw_dir, out_dir = tmp_path / "raw", tmp_path / "out"
    _write_raw(raw_fixture, raw_dir)
    report = run_pipeline(raw_dir, out_dir)

    allp, delp = out_dir / "orders_all.parquet", out_dir / "orders_delivered.parquet"
    csvp, repp = out_dir / "orders_delivered.csv", out_dir / "data_quality_report.json"
    assert all(p.is_file() for p in (allp, delp, csvp, repp))

    all_df, del_df = pd.read_parquet(allp), pd.read_parquet(delp)
    assert len(all_df) == 12 and len(del_df) == 7
    assert del_df["analysis_eligible"].all() and (del_df["exclusion_reason"] == "").all()
    assert set(del_df["order_id"]) == set(all_df.loc[all_df["analysis_eligible"], "order_id"])
    assert all_df["is_late"].isna().sum() == 5 and not del_df["is_late"].isna().any()  # NA survives parquet
    assert list(all_df.columns) == PROCESSED_COLUMNS
    # dtypes survive the parquet round trip (Int8 NA, categories, int8 flags)
    _assert_contract_dtypes(all_df)
    _assert_contract_dtypes(del_df)
    # the delivered subset carries no zero-count categorical levels
    _assert_no_unused_levels(del_df)
    assert del_df["order_status"].cat.categories.tolist() == ["delivered"]

    csv = pd.read_csv(csvp)
    assert len(csv) == 7 and set(csv["order_id"]) == set(del_df["order_id"])
    assert csv.set_index("order_id")["is_late"].astype(int).to_dict() == \
        del_df.set_index("order_id")["is_late"].astype(int).to_dict()

    saved = json.loads(repp.read_text())
    assert saved["outputs"]["rows"] == {"orders_all": 12, "orders_delivered": 7}
    assert report["outputs"]["rows"] == saved["outputs"]["rows"]
    assert saved["n_eligible"] == 7
    assert saved["order_status_counts"] == {"delivered": 9, "canceled": 1, "shipped": 1, "unavailable": 1}
    assert saved["orders_without_items"]["status_counts"] == {"unavailable": 1}  # no zero-count levels

    run_pipeline(raw_dir, out_dir)  # second run: identical data
    pd.testing.assert_frame_equal(pd.read_parquet(allp), all_df)
    pd.testing.assert_frame_equal(pd.read_parquet(delp), del_df)


def test_cli_main_reports_missing_data_and_success(raw_fixture, tmp_path, monkeypatch, capsys):
    def boom(*a, **k):
        raise MissingDatasetError("no data here")
    monkeypatch.setattr(preprocessing, "run_pipeline", boom)
    assert preprocessing.main() == 1
    assert "no data here" in capsys.readouterr().err

    raw_dir, out_dir = tmp_path / "raw", tmp_path / "out"
    _write_raw(raw_fixture, raw_dir)
    monkeypatch.setattr(preprocessing, "run_pipeline", functools.partial(run_pipeline, raw_dir, out_dir))
    assert preprocessing.main() == 0
    assert "Eligible" in capsys.readouterr().out and (out_dir / "orders_delivered.parquet").is_file()


@needs_real
def test_real_dtypes_and_no_unused_levels(real_all, real_delivered):
    _assert_contract_dtypes(real_all)
    _assert_contract_dtypes(real_delivered)
    _assert_no_unused_levels(real_delivered)
    assert real_delivered["order_status"].value_counts().to_dict() == {"delivered": len(real_delivered)}
    saved = json.loads(config.QUALITY_REPORT_PATH.read_text())
    assert all(v > 0 for v in saved["order_status_counts"].values())
    assert all(v > 0 for v in saved["orders_without_items"]["status_counts"].values())


@needs_real
def test_real_no_zero_weight_and_report_counts(real_all):
    assert not (real_all["total_weight_g"] <= 0).any()
    pq = json.loads(config.QUALITY_REPORT_PATH.read_text())["product_quality"]
    assert pq["zero_weight_products"] == 4
