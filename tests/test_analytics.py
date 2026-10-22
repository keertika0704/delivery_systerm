"""Tests for analytics, statistical_analysis and insights.

Unit tests use SMALL HAND-BUILT SYNTHETIC FIXTURES (clearly synthetic ids such as "o_t01" /
"o_s00001"); they are test fixtures only and never used by the app.  Known answers were
worked out by hand (see the comments next to each fixture).  Integration tests at the bottom
run against the real processed parquet and are skipped when it has not been generated.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src import analytics as an
from src import config
from src import insights as ins
from src import statistical_analysis as sa

# --------------------------------------------------------------------------- #
# Fixture 1: 12 synthetic orders with hand-computed answers (TEST FIXTURE ONLY)
# --------------------------------------------------------------------------- #
_NAN = np.nan
#      id      date          cust seller  cust_city        seller_city       cat     late delay dur   freight price  weight proc
_ROWS = [
    ("o_t01", "2017-01-05", "SP", "SP", "campinas", "campinas", "toys", 0, -3, 5.0, 10.0, 100.0, 500, 1.0),
    ("o_t02", "2017-01-20", "SP", "SP", "santos", "campinas", "toys", 0, -1, 7.0, 12.0, 60.0, 800, 2.0),
    ("o_t03", "2017-01-31", "SP", "MG", "santos", "belo horizonte", "toys", 1, 2, 12.0, 20.0, 40.0, 1500, 3.0),
    ("o_t04", "2017-02-01", "SP", "SP", "campinas", "santos", "books", 0, 0, 6.0, 8.0, 80.0, 300, 1.0),
    ("o_t05", "2017-02-10", "SP", "SP", "campinas", "campinas", "books", 0, -5, 4.0, 9.0, 90.0, 250, 1.0),
    ("o_t06", "2017-02-15", "SP", "MG", "santos", "belo horizonte", "books", 0, -2, 9.0, 15.0, 50.0, 1200, 2.0),
    ("o_t07", "2017-01-15", "RJ", "SP", "niteroi", "campinas", "toys", 1, 4, 20.0, 30.0, 60.0, 2500, 4.0),
    ("o_t08", "2017-02-20", "RJ", "SP", "rio de janeiro", "santos", "toys", 1, 6, 25.0, 35.0, 70.0, 3000, 5.0),
    ("o_t09", "2017-02-25", "RJ", "RJ", "rio de janeiro", "rio de janeiro", "books", 0, -1, 8.0, 10.0, 100.0, 400, 1.0),
    ("o_t10", "2017-01-10", "RJ", "RJ", "niteroi", "rio de janeiro", "books", 0, -2, 6.0, 11.0, 90.0, 450, 2.0),
    ("o_t11", "2017-02-28", "BA", "SP", "salvador", "campinas", "toys", 1, 9, 30.0, 45.0, 50.0, 4000, _NAN),
    ("o_t12", "2017-01-25", "BA", "SP", "salvador", "santos", "books", 1, 3, 22.0, 40.0, 55.0, 3500, 6.0),
]
_COLS = ["order_id", "purchase_date", "customer_state", "seller_state", "customer_city", "seller_city",
         "product_category", "is_late", "delivery_delay_days", "delivery_duration_days",
         "total_freight", "total_price", "total_weight_g", "processing_days"]
# Hand-computed: n=12, late=5 (o03,o07,o08,o11,o12) -> late_rate 5/12; SP n=6 k=1, RJ n=4 k=2, BA n=2 k=2;
# Jan n=6 k=3 (o03,o07,o12), Feb n=6 k=2 (o08,o11); sum(duration)=154, sum(freight)=245;
# late-order delays 2+4+6+9+3 = 24; toys n=6 k=4, books n=6 k=1.


def make_orders(rows=_ROWS) -> pd.DataFrame:
    """Build a frame with the processed dtypes from the fixture rows."""
    df = pd.DataFrame(rows, columns=_COLS)
    df["purchase_date"] = pd.to_datetime(df["purchase_date"])
    df["purchase_year_month"] = df["purchase_date"].dt.strftime("%Y-%m")
    df["purchase_month"] = df["purchase_date"].dt.month.astype("Int8")
    for c in ["customer_state", "seller_state", "customer_city", "seller_city", "product_category"]:
        df[c] = df[c].astype("category")
    df["state_pair"] = (df["seller_state"].astype(str) + "→" + df["customer_state"].astype(str)).astype("category")
    df["same_state"] = (df["customer_state"].astype(str) == df["seller_state"].astype(str)).astype("int8")
    df["is_late"] = df["is_late"].astype("Int8")
    df["delivery_delay_days"] = df["delivery_delay_days"].astype("Int32")
    df["freight_ratio"] = df["total_freight"] / df["total_price"]
    df["carrier_to_customer_days"] = df["delivery_duration_days"] - df["processing_days"]
    return df


@pytest.fixture()
def fx() -> pd.DataFrame:
    return make_orders()


# --------------------------------------------------------------------------- #
# Fixture 2: block-built synthetic frame for insights (TEST FIXTURE ONLY)
# --------------------------------------------------------------------------- #
def make_blocks(blocks) -> pd.DataFrame:
    """blocks = (customer_state, seller_state, 'YYYY-MM', category, n_orders, n_late)."""
    parts, start = [], 0
    for cs, ss, month, cat, n, k in blocks:
        j = np.arange(n)
        late = (j < k).astype(int)
        parts.append(pd.DataFrame({
            "order_id": [f"o_s{start + i:05d}" for i in j],
            "purchase_date": pd.Timestamp(f"{month}-15"),
            "customer_state": cs, "seller_state": ss, "customer_city": "city_x", "seller_city": "city_y",
            "product_category": cat, "is_late": late,
            "delivery_delay_days": np.where(late == 1, 3 + j % 4, -(1 + j % 3)),
            "delivery_duration_days": 10.0 + (j % 5) + 8.0 * late,
            "total_freight": 10.0 + (j % 7), "total_price": 100.0, "total_weight_g": 500.0 + 100.0 * (j % 9),
            "processing_days": 1.0 + (j % 3) + 3.0 * late,
        }))
        start += n
    df = pd.concat(parts, ignore_index=True)
    df["purchase_year_month"] = df["purchase_date"].dt.strftime("%Y-%m")
    df["purchase_month"] = df["purchase_date"].dt.month.astype("Int8")
    df["same_state"] = (df["customer_state"] == df["seller_state"]).astype("int8")
    df["state_pair"] = df["seller_state"] + "→" + df["customer_state"]
    df["freight_ratio"] = df["total_freight"] / df["total_price"]
    df["carrier_to_customer_days"] = df["delivery_duration_days"] - df["processing_days"]
    df["is_late"] = df["is_late"].astype("Int8")
    df["delivery_delay_days"] = df["delivery_delay_days"].astype("Int32")
    for c in ["customer_state", "seller_state", "product_category"]:
        df[c] = df[c].astype("category")
    return df


# SP: n=900 k=90 (10%); RJ: n=600 k=108 (18%); AC: n=20 k=10 (50%, too small to rank).
# 2017-03 has 500 orders with 120 late (24%); the other months are far lower.  Overall 208/1520.
BLOCKS = [
    ("SP", "SP", "2017-01", "toys", 300, 15), ("SP", "SP", "2017-02", "toys", 300, 15),
    ("SP", "SP", "2017-03", "books", 300, 60),
    ("RJ", "SP", "2017-01", "toys", 200, 24), ("RJ", "SP", "2017-02", "books", 200, 24),
    ("RJ", "SP", "2017-03", "books", 200, 60),
    ("AC", "SP", "2017-02", "toys", 20, 10),
]


@pytest.fixture(scope="module")
def big() -> pd.DataFrame:
    return make_blocks(BLOCKS)


# --------------------------------------------------------------------------- #
# Wilson CI
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k,n", [(0, 10), (10, 10), (1, 1), (5, 12), (65, 1000), (6534, 96470), (3, 20)])
def test_wilson_matches_scipy(k, n):
    ref = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    lo, hi = an.wilson_ci(k, n)
    assert lo == pytest.approx(ref.low, abs=1e-9)
    assert hi == pytest.approx(ref.high, abs=1e-9)


def test_wilson_alpha_and_undefined():
    ref = stats.binomtest(7, 40).proportion_ci(confidence_level=0.99, method="wilson")
    lo, hi = an.wilson_ci(7, 40, alpha=0.01)
    assert (lo, hi) == pytest.approx((ref.low, ref.high), abs=1e-9)
    lo, hi = an.wilson_ci(0, 0)
    assert math.isnan(lo) and math.isnan(hi)


# --------------------------------------------------------------------------- #
# apply_filters
# --------------------------------------------------------------------------- #
def test_filters_no_filter_returns_equal_copy(fx):
    out = an.apply_filters(fx)
    pd.testing.assert_frame_equal(out, fx)
    assert out is not fx


def test_filters_never_mutate_input(fx):
    before = fx.copy(deep=True)
    out = an.apply_filters(fx, ("2017-01-01", "2017-01-31"), ["SP"], ["toys"], ["late"], "BELO")
    out["is_late"] = 0
    pd.testing.assert_frame_equal(fx, before)


def test_filters_date_range_inclusive(fx):
    out = an.apply_filters(fx, (pd.Timestamp("2017-01-20"), pd.Timestamp("2017-02-01")))
    assert set(out["order_id"]) == {"o_t02", "o_t12", "o_t03", "o_t04"}  # both bounds included
    only_end = an.apply_filters(fx, (None, "2017-01-05"))
    assert set(only_end["order_id"]) == {"o_t01"}
    only_start = an.apply_filters(fx, ("2017-02-28", None))
    assert set(only_start["order_id"]) == {"o_t11"}


def test_filters_state_category_and_empty_lists(fx):
    assert len(an.apply_filters(fx, states=["SP"])) == 6
    assert len(an.apply_filters(fx, states=["SP", "BA"])) == 8
    assert len(an.apply_filters(fx, categories=["books"])) == 6
    assert len(an.apply_filters(fx, states=[], categories=[], statuses=[], search="")) == 12
    assert len(an.apply_filters(fx, states=["ZZ"])) == 0


def test_filters_status_labels(fx):
    assert len(an.apply_filters(fx, statuses=["late"])) == 5
    assert len(an.apply_filters(fx, statuses=["on_time"])) == 7
    assert len(an.apply_filters(fx, statuses=["On time"])) == 7
    assert len(an.apply_filters(fx, statuses=["late", "on_time"])) == 12
    labels = an.delivery_status_label(fx)
    assert (labels == "late").sum() == 5 and (labels == "on_time").sum() == 7


def test_filters_search_case_insensitive_multi_column(fx):
    assert set(an.apply_filters(fx, search="NITEROI")["order_id"]) == {"o_t07", "o_t10"}  # customer_city
    assert set(an.apply_filters(fx, search="Belo Horizonte")["order_id"]) == {"o_t03", "o_t06"}  # seller_city
    assert len(an.apply_filters(fx, search="toys")) == 6  # category
    assert set(an.apply_filters(fx, search="O_T11")["order_id"]) == {"o_t11"}  # order id
    assert an.apply_filters(fx, search="[").empty  # literal match, no regex error
    assert len(an.apply_filters(fx, search="  ")) == 12


def test_filters_combined_consistency(fx):
    out = an.apply_filters(fx, ("2017-01-01", "2017-01-31"), ["SP", "RJ"], ["toys"], ["late"])
    assert set(out["order_id"]) == {"o_t03", "o_t07"}
    # per-state n sums to the filtered total and KPI n equals len(filtered)
    sp = an.state_performance(out)
    assert sp["n_orders"].sum() == len(out) == an.kpi_summary(out)["n_orders"]


def test_filters_on_empty_frame(fx):
    e = fx.iloc[0:0]
    assert an.apply_filters(e, ("2017-01-01", "2017-02-01"), ["SP"], ["toys"], ["late"], "x").empty


# --------------------------------------------------------------------------- #
# kpi_summary
# --------------------------------------------------------------------------- #
def test_kpi_known_values(fx):
    k = an.kpi_summary(fx)
    assert k["n_orders"] == 12 and k["n_late"] == 5 and k["n_states"] == 3
    assert k["late_rate"] == pytest.approx(5 / 12)
    assert k["on_time_rate"] == pytest.approx(7 / 12)
    assert k["mean_duration"] == pytest.approx(154 / 12)
    assert k["median_duration"] == pytest.approx(8.5)
    assert k["mean_late_days"] == pytest.approx(24 / 5)
    assert k["total_freight"] == pytest.approx(245.0)
    assert set(k) == set(an.KPI_KEYS)


def test_kpi_empty_single_all_late_all_on_time(fx):
    k = an.kpi_summary(fx.iloc[0:0])
    assert k["n_orders"] == 0 and k["n_late"] == 0 and k["total_freight"] == 0 and k["n_states"] == 0
    assert all(math.isnan(k[x]) for x in ("late_rate", "on_time_rate", "mean_duration",
                                          "median_duration", "mean_late_days"))
    one = an.kpi_summary(fx.iloc[[2]])  # o_t03: late, 2 days
    assert one["n_orders"] == 1 and one["late_rate"] == 1 and one["mean_late_days"] == 2
    all_late = an.kpi_summary(fx[fx["is_late"] == 1])
    assert all_late["late_rate"] == 1 and all_late["on_time_rate"] == 0
    on_time = an.kpi_summary(fx[fx["is_late"] == 0])
    assert on_time["late_rate"] == 0 and on_time["n_late"] == 0
    assert math.isnan(on_time["mean_late_days"])  # undefined: nobody is late


def test_kpi_ignores_unlabelled_rows(fx):
    d = fx.copy()
    d.loc[0, "is_late"] = pd.NA
    assert an.kpi_summary(d)["n_orders"] == 11


# --------------------------------------------------------------------------- #
# Time series, duration
# --------------------------------------------------------------------------- #
def test_monthly_performance(fx):
    m = an.monthly_performance(fx).set_index("purchase_year_month")
    assert list(m.index) == ["2017-01", "2017-02"]
    assert m.loc["2017-01", ["n_orders", "n_late"]].tolist() == [6, 3]
    assert m.loc["2017-02", ["n_orders", "n_late"]].tolist() == [6, 2]
    assert m.loc["2017-01", "late_rate"] == pytest.approx(0.5)
    assert m.loc["2017-02", "on_time_rate"] == pytest.approx(4 / 6)
    assert m["n_orders"].sum() == 12
    lo, hi = an.wilson_ci(3, 6)
    assert m.loc["2017-01", "late_ci_low"] == pytest.approx(lo)
    assert an.monthly_performance(fx, min_orders=7)["reliable"].sum() == 0


def test_delay_trend_matches_monthly_and_weekly_sums(fx):
    monthly = an.monthly_performance(fx)
    trend = an.delay_trend(fx, "M")
    assert trend["n_orders"].tolist() == monthly["n_orders"].tolist()
    assert trend["n_late"].tolist() == monthly["n_late"].tolist()
    assert trend["period"].iloc[0] == pd.Timestamp("2017-01-01")
    weekly = an.delay_trend(fx, "W")
    assert weekly["n_orders"].sum() == 12 and weekly["n_late"].sum() == 5
    assert an.delay_trend(fx, "ME")["n_orders"].sum() == 12  # pandas offset alias accepted


def test_duration_summary_known_and_degenerate(fx):
    s = an.duration_summary(fx)
    v = fx["delivery_duration_days"]
    assert s["n"] == 12 and s["mean"] == pytest.approx(v.mean()) and s["median"] == pytest.approx(8.5)
    assert s["std"] == pytest.approx(v.std()) and s["p90"] == pytest.approx(np.percentile(v, 90))
    assert s["p95"] == pytest.approx(np.percentile(v, 95))
    assert s["min"] == 4 and s["max"] == 30 and s["skew"] == pytest.approx(v.skew())
    assert set(s) == set(an.DURATION_KEYS)
    e = an.duration_summary(fx.iloc[0:0])
    assert e["n"] == 0 and math.isnan(e["mean"])
    one = an.duration_summary(fx.iloc[[0]])
    assert one["n"] == 1 and one["mean"] == 5 and math.isnan(one["std"]) and math.isnan(one["skew"])


# --------------------------------------------------------------------------- #
# Segment tables
# --------------------------------------------------------------------------- #
def test_state_performance_known(fx):
    t = an.state_performance(fx, min_orders=4).set_index("customer_state")
    assert t.loc["SP", ["n_orders", "n_late"]].tolist() == [6, 1]
    assert t.loc["RJ", ["n_orders", "n_late"]].tolist() == [4, 2]
    assert t.loc["BA", ["n_orders", "n_late"]].tolist() == [2, 2]
    assert t.loc["RJ", "late_rate"] == pytest.approx(0.5)
    assert t.loc["SP", "mean_duration"] == pytest.approx(43 / 6)
    assert t.loc["BA", "mean_late_days"] == pytest.approx(6.0)  # (9 + 3) / 2
    assert t.loc["SP", "mean_late_days"] == pytest.approx(2.0)  # only o_t03 is late
    assert t.loc["SP", "mean_freight"] == pytest.approx(74 / 6)
    assert t["reliable"].to_dict() == {"SP": True, "RJ": True, "BA": False}
    for st, k, n in (("SP", 1, 6), ("RJ", 2, 4), ("BA", 2, 2)):
        ref = stats.binomtest(k, n).proportion_ci(method="wilson")
        assert t.loc[st, "late_ci_low"] == pytest.approx(ref.low, abs=1e-9)
        assert t.loc[st, "late_ci_high"] == pytest.approx(ref.high, abs=1e-9)
    assert t["n_orders"].sum() == 12


def test_performance_table_columns_order_and_empty(fx):
    expected = ["customer_state", "n_orders", "n_late", "on_time_rate", "late_rate", "late_ci_low",
                "late_ci_high", "mean_duration", "median_duration", "mean_late_days", "mean_freight",
                "mean_weight_g", "reliable"]
    t = an.state_performance(fx)
    assert list(t.columns) == expected
    # reliable first, then late rate descending
    t2 = an.state_performance(fx, min_orders=4)
    assert list(t2["customer_state"]) == ["RJ", "SP", "BA"]
    e = an.state_performance(fx.iloc[0:0])
    assert e.empty and list(e.columns) == expected
    assert an.category_performance(fx.iloc[0:0]).empty
    assert an.state_pair_performance(fx.iloc[0:0]).empty
    assert "state_pair" in an.state_pair_performance(fx.iloc[0:0]).columns
    single = an.state_performance(fx.iloc[[0]])
    assert single["n_orders"].tolist() == [1] and single["late_rate"].tolist() == [0]


def test_state_performance_seller_axis(fx):
    t = an.state_performance(fx, by="seller_state").set_index("seller_state")
    assert t.loc["SP", "n_orders"] == 8 and t.loc["MG", "n_orders"] == 2 and t["n_orders"].sum() == 12


def test_category_performance_known(fx):
    t = an.category_performance(fx).set_index("product_category")
    assert t.loc["toys", ["n_orders", "n_late"]].tolist() == [6, 4]
    assert t.loc["books", ["n_orders", "n_late"]].tolist() == [6, 1]
    assert t.loc["toys", "late_rate"] == pytest.approx(4 / 6)
    assert t.loc["books", "mean_freight"] == pytest.approx(15.5)


def test_state_pair_performance_known(fx):
    t = an.state_pair_performance(fx).set_index("state_pair")
    assert t.loc["SP→SP", ["n_orders", "n_late"]].tolist() == [4, 0]
    assert t.loc["MG→SP", ["n_orders", "n_late"]].tolist() == [2, 1]
    assert t.loc["SP→RJ", ["n_orders", "n_late"]].tolist() == [2, 2]
    assert t.loc["SP→BA", ["n_orders", "n_late"]].tolist() == [2, 2]
    assert t.loc["RJ→RJ", ["n_orders", "n_late"]].tolist() == [2, 0]
    assert t.loc["SP→RJ", "seller_state"] == "SP" and t.loc["SP→RJ", "customer_state"] == "RJ"
    assert bool(t.loc["SP→SP", "same_state"]) and not bool(t.loc["SP→RJ", "same_state"])
    assert t["n_orders"].sum() == 12


def test_high_delay_segments_rules():
    perf = pd.DataFrame({
        "seg": ["big_high", "small_high", "big_near", "big_low", "mid_edge"],
        "n_orders": [1000, 20, 1000, 1000, 150],
        "n_late": [200, 10, 105, 20, 18],
    })
    perf["late_rate"] = perf["n_late"] / perf["n_orders"]
    out = an.high_delay_segments(perf, min_orders=100, overall_late_rate=0.10)
    assert list(out["seg"]) == ["big_high"]  # small_high fails n; big_near / mid_edge fail the Wilson bound
    assert out["excess_vs_overall"].iloc[0] == pytest.approx(0.10)
    assert out["rate_ratio"].iloc[0] == pytest.approx(2.0)
    lo, _ = an.wilson_ci(18, 150)
    assert lo < 0.10  # mid_edge is above 0.10 on the point estimate but not reliably
    low = an.high_delay_segments(perf, min_orders=100, overall_late_rate=0.10, direction="low")
    assert list(low["seg"]) == ["big_low"]
    # default overall = POOLED rate of the table (353 / 3170 = 11.1%), not the mean of the rates
    # (18.9%): big_high (20%, n = 1000) is above the pooled rate with its Wilson bound, but its
    # lower bound is below the unweighted mean, so only the pooled default flags it.
    pooled = perf["n_late"].sum() / perf["n_orders"].sum()
    assert pooled == pytest.approx(353 / 3170) and perf["late_rate"].mean() > 0.18
    assert list(an.high_delay_segments(perf, min_orders=100)["seg"]) == ["big_high"]
    assert an.high_delay_segments(perf, min_orders=100)["excess_vs_overall"].iloc[0] == pytest.approx(0.2 - pooled)
    assert list(an.high_delay_segments(perf, min_orders=100, direction="low")["seg"]) == ["big_low"]
    assert an.high_delay_segments(perf.iloc[0:0], 100, 0.1).empty


def test_binned_stat_fixed_edges_known(fx):
    t = an.weight_bins(fx)
    assert t["n_orders"].tolist() == [5, 1, 2, 4]
    assert t["n_late"].tolist() == [0, 0, 1, 4]
    assert t["bin"].tolist() == ["[0, 500]", "(500, 1000]", "(1000, 2000]", "(2000, 5000]"]
    assert t["n_orders"].sum() == 12
    assert t.loc[3, "late_rate"] == 1.0 and t.loc[0, "late_rate"] == 0.0
    assert t.loc[2, "mean_value"] == pytest.approx(1350.0)
    assert t["bin_low"].tolist() == [0, 500, 1000, 2000]


def test_binned_stat_boundaries_and_overflow(fx):
    d = fx.copy()
    d["x"] = np.arange(12, dtype=float)  # 0..11
    t = an.binned_stat(d, "x", [0, 3, 6], quantile=False)
    assert t["n_orders"].tolist() == [4, 3]  # [0,3] -> 0,1,2,3 ; (3,6] -> 4,5,6 ; 7..11 dropped
    t = an.binned_stat(d, "x", [0, 5, np.inf])
    assert t["n_orders"].tolist() == [6, 6] and t["bin"].tolist()[-1] == "> 5"


def test_binned_stat_quantile_duplicate_edges_and_constant(fx):
    d = fx.copy()
    d["x"] = [0.0] * 8 + [1.0, 2.0, 3.0, 4.0]  # heavy ties -> duplicate quantile edges
    t = an.binned_stat(d, "x", bins=4, quantile=True)
    assert t["n_orders"].sum() == 12
    assert t["bin_low"].is_monotonic_increasing and not t["bin_low"].duplicated().any()
    d["c"] = 7.0
    c = an.binned_stat(d, "c", bins=5)
    assert len(c) == 1 and c["n_orders"].iloc[0] == 12 and c["bin"].iloc[0] == "7"
    eq = an.binned_stat(fx, "total_weight_g", bins=3, quantile=False)
    assert eq["n_orders"].sum() == 12 and len(eq) <= 3


def test_binned_stat_nan_empty_missing_column(fx):
    d = fx.copy()
    d.loc[[0, 1], "total_weight_g"] = np.nan
    assert an.weight_bins(d)["n_orders"].sum() == 10
    assert an.weight_bins(fx.iloc[0:0]).empty
    assert an.binned_stat(fx, "nope").empty
    one = an.freight_bins(fx.iloc[[0]])
    assert one["n_orders"].tolist() == [1]
    q = an.freight_bins(fx, quantile=True, bins=3)
    assert q["n_orders"].sum() == 12


def test_processing_summary_and_buckets(fx):
    p = an.processing_summary(fx)
    assert p["retrospective"] is True and p["n"] == 11 and p["n_missing"] == 1
    assert p["mean"] == pytest.approx(28 / 11) and p["median"] == pytest.approx(2.0)
    assert p["max"] == 6.0
    both = fx["processing_days"].notna()
    assert p["share_of_duration"] == pytest.approx(28 / fx.loc[both, "delivery_duration_days"].sum())
    b = an.late_by_processing_bucket(fx)
    assert b["n_orders"].sum() == 11  # the NaN row is ignored
    assert b.loc[b["bin"] == "[0, 1]", "n_orders"].iloc[0] == 4  # processing 1,1,1,1
    e = an.processing_summary(fx.iloc[0:0])
    assert e["n"] == 0 and math.isnan(e["mean"]) and e["retrospective"] is True
    assert an.late_by_processing_bucket(fx.iloc[0:0]).empty
    assert "retrospective" in an.late_by_processing_bucket.__doc__.lower()
    assert "retrospective" in an.processing_summary.__doc__.lower()


def test_scatter_sample_deterministic(big):
    a = an.scatter_sample(big, "total_weight_g", "delivery_duration_days", n=200)
    b = an.scatter_sample(big, "total_weight_g", "delivery_duration_days", n=200)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 200 and list(a.columns) == ["total_weight_g", "delivery_duration_days", "is_late"]
    c = an.scatter_sample(big, "total_weight_g", "delivery_duration_days", n=200, seed=1)
    assert not a.equals(c)  # a different seed gives a different sample
    small = an.scatter_sample(make_orders(), "total_weight_g", "processing_days", n=100)
    assert len(small) == 11  # NaN pair dropped, fewer rows than n -> all kept
    assert an.scatter_sample(big.iloc[0:0], "total_weight_g", "delivery_duration_days").empty
    assert an.scatter_sample(big, "nope", "delivery_duration_days").empty


def test_category_cost_comparison(fx):
    t = an.category_cost_comparison(fx, min_orders=1).set_index("product_category")
    assert t.loc["toys", "mean_freight"] == pytest.approx(152 / 6)
    assert t.loc["books", "mean_freight"] == pytest.approx(15.5)
    assert t.loc["toys", "total_freight"] == pytest.approx(152.0)
    assert t.loc["books", "median_freight"] == pytest.approx(10.5)  # 8,9,10,11,15,40
    exp_ratio = (fx.loc[fx["product_category"] == "books", "total_freight"]
                 / fx.loc[fx["product_category"] == "books", "total_price"]).mean()
    assert t.loc["books", "mean_freight_ratio"] == pytest.approx(exp_ratio)
    assert list(an.category_cost_comparison(fx)["product_category"]) == ["toys", "books"]  # freight desc
    assert an.category_cost_comparison(fx.iloc[0:0]).empty


def test_state_matrix(fx):
    m = an.state_matrix(fx)
    n = m["n_orders"]
    assert n.loc["SP", "SP"] == 4 and n.loc["MG", "SP"] == 2 and n.loc["SP", "RJ"] == 2
    assert n.loc["RJ", "SP"] == 0  # unobserved pair
    assert int(n.to_numpy().sum()) == 12
    assert m["late_rate"].loc["SP", "RJ"] == 1.0 and m["late_rate"].loc["SP", "SP"] == 0.0
    assert math.isnan(m["late_rate"].loc["RJ", "SP"])
    masked = an.state_matrix(fx, min_orders=3)
    assert masked["late_rate"].loc["SP", "SP"] == 0.0 and math.isnan(masked["late_rate"].loc["SP", "RJ"])
    assert m["mean_duration"].loc["SP", "RJ"] == pytest.approx(22.5)
    top = an.state_matrix(fx, top_n=1)
    assert top["n_orders"].shape == (1, 1)
    assert all(v.empty for v in an.state_matrix(fx.iloc[0:0]).values())


def test_same_state_comparison(fx):
    t = an.same_state_comparison(fx).set_index("group")
    assert t.loc["same_state", ["n_orders", "n_late"]].tolist() == [6, 0]
    assert t.loc["cross_state", ["n_orders", "n_late"]].tolist() == [6, 5]
    assert t.loc["cross_state", "late_rate"] == pytest.approx(5 / 6)
    assert an.same_state_comparison(fx.iloc[0:0]).empty


# --------------------------------------------------------------------------- #
# Statistical analysis
# --------------------------------------------------------------------------- #
def _df_from_counts(table: dict, col: str = "g") -> pd.DataFrame:
    """Rows reproducing {level: (n_on_time, n_late)}."""
    parts = []
    for lvl, (ok, late) in table.items():
        parts.append(pd.DataFrame({col: lvl, "is_late": [0] * ok + [1] * late}))
    return pd.concat(parts, ignore_index=True)


def test_chi_square_matches_scipy():
    counts = {"A": (70, 30), "B": (50, 50), "C": (80, 20)}
    res = sa.chi_square_test(_df_from_counts(counts), "g")
    obs = np.array(list(counts.values()))
    chi2, p, dof, _ = stats.chi2_contingency(obs, correction=False)
    n = obs.sum()
    assert res["chi2"] == pytest.approx(chi2) and res["p_value"] == pytest.approx(p)
    assert res["dof"] == dof == 2 and res["n"] == n == 300
    assert res["cramers_v"] == pytest.approx(math.sqrt(chi2 / (n * 1)))
    assert res["effect_label"] == sa.cramers_v_label(res["cramers_v"], 1)
    assert res["merged_rare_levels"] == [] and res["n_levels_tested"] == 3
    assert res["table"].to_numpy().tolist() == obs.tolist()
    assert res["significant"] is True
    ls = res["level_summary"]
    assert set(ls["level"]) == {"A", "B", "C"} and ls["n"].sum() == 300
    assert abs(ls.iloc[0]["adj_residual"]) == ls["adj_residual"].abs().max()  # top contributor first
    assert 0 <= res["cramers_v_corrected"] <= res["cramers_v"]


def test_chi_square_2x2_no_yates_and_effect_labels():
    res = sa.chi_square_test(_df_from_counts({"x": (90, 10), "y": (70, 30)}), "g")
    chi2, p, _, _ = stats.chi2_contingency(np.array([[90, 10], [70, 30]]), correction=False)
    assert res["chi2"] == pytest.approx(chi2)
    # thresholds: 0.1 / 0.3 / 0.5 for one degree of freedom, scaled by 1/sqrt(df*)
    assert [sa.cramers_v_label(v, 1) for v in (0.05, 0.2, 0.4, 0.6)] == ["negligible", "small", "medium", "large"]
    assert sa.cramers_v_label(0.06, 4) == "small"  # 0.06 >= 0.1/sqrt(4)=0.05
    assert sa.cramers_v_label(float("nan"), 1) == "undefined"


def test_chi_square_independent_data_is_not_significant():
    res = sa.chi_square_test(_df_from_counts({"A": (90, 10), "B": (90, 10), "C": (45, 5)}), "g")
    assert res["chi2"] == pytest.approx(0.0, abs=1e-9) and res["p_value"] == pytest.approx(1.0)
    assert res["significant"] is False and res["effect_label"] == "negligible"


def test_chi_square_rare_level_merging():
    counts = {"A": (800, 200), "B": (900, 100), "C": (24, 6), "D": (17, 3)}
    res = sa.chi_square_test(_df_from_counts(counts), "g", min_expected=5)
    assert res["merged_rare_levels"] == ["C", "D"]
    assert sa.MERGED_LABEL in res["table"].index and res["n_levels_tested"] == 3
    assert res["table"].loc[sa.MERGED_LABEL].sum() == 50
    assert res["min_expected_after"] >= 5
    assert res["n_levels_original"] == 4 and res["n"] == 2050
    # a single rare level is pooled with the next smallest so the pool is >= 2 levels
    res2 = sa.chi_square_test(_df_from_counts({"A": (800, 200), "B": (900, 100), "C": (17, 3)}), "g")
    assert "C" in res2["merged_rare_levels"] and len(res2["merged_rare_levels"]) >= 2
    assert res2["min_expected_after"] >= 5
    # top_n pools everything beyond the n most frequent levels
    res3 = sa.chi_square_test(_df_from_counts({"A": (800, 200), "B": (900, 100), "C": (600, 300)}), "g", top_n=2)
    assert res3["merged_rare_levels"] == ["C"] and res3["n"] == 2900  # exactly the least frequent level
    assert list(res3["table"].index) == ["A", "B", sa.MERGED_LABEL] and res3["n_levels_tested"] == 3
    assert res3["table"].loc[sa.MERGED_LABEL].tolist() == [600, 300]
    # top_n keeps the n most frequent levels (by total), pooling the rest -- 4 levels, top 2
    res4 = sa.chi_square_test(_df_from_counts({"A": (800, 200), "B": (900, 100), "C": (600, 300),
                                               "D": (500, 300)}), "g", top_n=2)
    assert res4["merged_rare_levels"] == ["C", "D"] and res4["table"].loc[sa.MERGED_LABEL].sum() == 1700


def test_chi_square_degenerate_inputs(fx):
    for frame in (fx.iloc[0:0], fx.iloc[[0]], fx[fx["is_late"] == 1]):
        res = sa.chi_square_test(frame, "customer_state")
        assert math.isnan(res["chi2"]) and math.isnan(res["p_value"]) and res["note"]
        assert res["significant"] is False
    assert math.isnan(sa.chi_square_test(fx, "no_such_column")["chi2"])
    # works on category + nullable-integer columns (pandas dtypes of the real data)
    assert np.isfinite(sa.chi_square_test(make_blocks(BLOCKS), "customer_state")["chi2"])
    assert np.isfinite(sa.chi_square_test(make_blocks(BLOCKS), "purchase_month")["chi2"])


def test_spearman_matches_scipy_and_drops_nan():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan, 3.0])
    y = np.array([2, 1, 4, 3, 7, 8, 6, 10, 9, 12, 5.0, np.nan])
    df = pd.DataFrame({"x": x, "y": y})
    res = sa.spearman_test(df, "x", "y")
    ok = ~(np.isnan(x) | np.isnan(y))
    rho, p = stats.spearmanr(x[ok], y[ok])
    assert res["rho"] == pytest.approx(rho) and res["p_value"] == pytest.approx(p)
    assert res["n"] == 10  # NaN pairs dropped and counted
    z, se = math.atanh(rho), math.sqrt((1 + rho ** 2 / 2) / (10 - 3))
    assert res["ci_low"] == pytest.approx(math.tanh(z - 1.959964 * se), abs=1e-5)
    assert res["ci_high"] == pytest.approx(math.tanh(z + 1.959964 * se), abs=1e-5)
    assert res["ci_low"] < res["rho"] < res["ci_high"]
    assert res["strength_label"] == "very strong" and res["direction"] == "positive"
    assert res["significant"] is True


def test_spearman_labels_edge_cases_and_nullable_ints(fx):
    assert [sa.spearman_strength(r) for r in (0.05, -0.2, 0.4, -0.6, 0.9)] == \
        ["negligible", "weak", "moderate", "strong", "very strong"]
    perfect = sa.spearman_test(pd.DataFrame({"a": range(10), "b": [v * 2 for v in range(10)]}), "a", "b")
    assert perfect["rho"] == pytest.approx(1.0) and perfect["ci_low"] == perfect["ci_high"] == perfect["rho"]  # degenerate CI
    const = sa.spearman_test(pd.DataFrame({"a": [1.0] * 6, "b": range(6)}), "a", "b")
    assert math.isnan(const["rho"]) and "constant" in const["note"]
    assert math.isnan(sa.spearman_test(fx.iloc[0:0], "total_weight_g", "total_freight")["rho"])
    assert math.isnan(sa.spearman_test(fx.iloc[:3], "total_weight_g", "total_freight")["rho"])  # n < 4
    assert sa.spearman_test(fx, "nope", "total_freight")["note"] == "column not found"
    # point-biserial style with the nullable Int8 target
    r = sa.spearman_test(fx, "total_freight", "is_late")
    ref = stats.spearmanr(fx["total_freight"], fx["is_late"].astype(float))
    assert r["rho"] == pytest.approx(ref.statistic) and r["n"] == 12
    assert sa.spearman_test(fx, "total_weight_g", "processing_days")["n"] == 11


def test_proportion_diff_ci_newcombe():
    # Newcombe (1998), Table II example: 56/70 vs 48/80 -> method 10: 0.0524 to 0.3339
    r = sa.proportion_diff_ci(56, 70, 48, 80)
    assert r["diff"] == pytest.approx(0.2)
    assert r["ci_low"] == pytest.approx(0.0524, abs=5e-4) and r["ci_high"] == pytest.approx(0.3339, abs=5e-4)
    # independent rebuild from scipy's Wilson intervals
    w1 = stats.binomtest(56, 70).proportion_ci(method="wilson")
    w2 = stats.binomtest(48, 80).proportion_ci(method="wilson")
    d = 56 / 70 - 48 / 80
    assert r["ci_low"] == pytest.approx(d - math.sqrt((56 / 70 - w1.low) ** 2 + (w2.high - 48 / 80) ** 2), abs=1e-9)
    assert r["ci_high"] == pytest.approx(d + math.sqrt((w1.high - 56 / 70) ** 2 + (48 / 80 - w2.low) ** 2), abs=1e-9)
    swapped = sa.proportion_diff_ci(48, 80, 56, 70)
    assert swapped["ci_low"] == pytest.approx(-r["ci_high"]) and swapped["diff"] == pytest.approx(-0.2)
    z = sa.two_proportion_ztest(56, 70, 48, 80)
    pooled = 104 / 150
    zref = (0.8 - 0.6) / math.sqrt(pooled * (1 - pooled) * (1 / 70 + 1 / 80))
    assert z["z"] == pytest.approx(zref) and z["p_value"] == pytest.approx(2 * stats.norm.sf(abs(zref)))
    bad = sa.proportion_diff_ci(1, 0, 1, 5)
    assert math.isnan(bad["diff"]) and math.isnan(sa.two_proportion_ztest(0, 0, 1, 5)["p_value"])
    assert math.isnan(sa.two_proportion_ztest(0, 10, 0, 10)["p_value"])  # zero variance


def test_mann_whitney_and_kruskal_match_scipy():
    df = pd.DataFrame({"v": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10.0], "g": ["a"] * 5 + ["b"] * 5})
    res = sa.kruskal_or_mannwhitney(df, "v", "g")
    u, p = stats.mannwhitneyu(df["v"][:5], df["v"][5:], alternative="two-sided")
    assert res["test"] == "mann_whitney" and res["statistic"] == pytest.approx(u)
    assert res["p_value"] == pytest.approx(p) and res["effect_size"] == pytest.approx(2 * u / 25 - 1)
    df3 = pd.DataFrame({"v": [1, 2, 3, 10, 11, 12, 20, 21, 22.0], "g": list("aaabbbccc")})
    r3 = sa.kruskal_or_mannwhitney(df3, "v", "g")
    h, p3 = stats.kruskal([1, 2, 3], [10, 11, 12], [20, 21, 22])
    assert r3["test"] == "kruskal" and r3["statistic"] == pytest.approx(h) and r3["p_value"] == pytest.approx(p3)
    assert r3["n_groups"] == 3 and r3["group_summary"]["n"].tolist() == [3, 3, 3]
    assert math.isnan(sa.kruskal_or_mannwhitney(df.iloc[0:0], "v", "g")["p_value"])
    assert math.isnan(sa.kruskal_or_mannwhitney(df[df["g"] == "a"], "v", "g")["p_value"])
    same = sa.kruskal_or_mannwhitney(pd.DataFrame({"v": [1.0] * 9, "g": list("aaabbbccc")}), "v", "g")
    assert math.isnan(same["p_value"])


def test_holm_adjust():
    adj = sa.holm_adjust([0.01, 0.04, 0.03, np.nan])
    assert adj[0] == pytest.approx(0.03) and adj[2] == pytest.approx(0.06) and adj[1] == pytest.approx(0.06)
    assert math.isnan(adj[3])
    assert sa.holm_adjust([]).size == 0


def test_run_all_tests_structure_on_synthetic(big):
    res = sa.run_all_tests(big)
    assert set(res["chi_square"]) == {"customer_state", "product_category", "same_state", "purchase_month"}
    assert "total_weight_g~delivery_duration_days" in res["spearman"]
    assert "freight_ratio~is_late" in res["spearman"]
    s = res["summary"]
    assert {"p_holm", "significant_holm", "effect_label", "n"} <= set(s.columns)
    assert (s["p_holm"].dropna() >= s["p_value"].dropna() - 1e-15).all()
    assert res["n_orders"] == len(big)
    assert res["chi_square"]["customer_state"]["significant"] is True  # SP 10% vs RJ 18% vs AC 50%


def test_run_all_tests_on_empty_and_tiny(fx):
    for frame in (fx.iloc[0:0], fx.iloc[[0]]):
        res = sa.run_all_tests(frame)
        assert len(res["summary"]) == 4 + len(sa.SPEARMAN_SPECS) + 2
        assert res["summary"]["p_value"].isna().all()
        for r in list(res["chi_square"].values()) + list(res["spearman"].values()):
            assert isinstance(sa.plain_language(r), str)


def test_plain_language_caveats_and_large_n():
    small = sa.chi_square_test(_df_from_counts({"A": (70, 30), "B": (50, 50)}), "g")
    txt = sa.plain_language(small)
    assert "not the same as practical importance" in txt and "causality" in txt
    assert "even tiny effects" not in txt  # n = 200 is not huge
    assert f"n = {small['n']:,}" in txt and f"{small['chi2']:.1f}" in txt
    big_res = sa.chi_square_test(_df_from_counts({"A": (9000, 1000), "B": (8900, 1100)}), "g")
    big_txt = sa.plain_language(big_res)
    assert "even tiny effects can be statistically 'significant'" in big_txt
    assert "causality" in big_txt and "20,000" in big_txt
    for r in (sa.spearman_test(pd.DataFrame({"a": range(30000), "b": range(30000)[::-1]}), "a", "b"),
              sa.proportion_diff_ci(50, 100, 40, 100),
              sa.kruskal_or_mannwhitney(pd.DataFrame({"v": range(10), "g": list("aaaaabbbbb")}), "v", "g"),
              {}, {"test": "chi_square", "col": "x", "chi2": float("nan"), "n": 0, "note": "n"}):
        assert "causality" in sa.plain_language(r) or "cause" in sa.plain_language(r)


# --------------------------------------------------------------------------- #
# Insights
# --------------------------------------------------------------------------- #
def _by_kind(items, kind):
    return [i for i in items if i.kind == kind]


def test_insights_empty_frame_single_no_data(fx):
    out = ins.generate_insights(fx.iloc[0:0])
    assert [i.kind for i in out] == ["no_data"] and out[0].n == 0
    assert ins.recommendations(out) and "no delivered orders" in ins.recommendations(out)[0].lower()


def test_insights_tiny_sample_says_insufficient_instead_of_ranking(fx):
    out = ins.generate_insights(fx, min_orders=100)
    kinds = [i.kind for i in out]
    assert kinds == ["overview", "insufficient_data"]
    assert "12" in out[1].text and "100" in out[1].text
    # a moderate sample whose individual segments are all below min_orders: no ranking, a notice
    out2 = ins.generate_insights(fx, min_orders=7)
    ranked = {i.kind for i in out2}
    assert "region_high" not in ranked and "pair_high" not in ranked  # no state reaches n = 7
    assert any(i.kind == "insufficient_data" and "regional" in i.text for i in out2)


def test_insights_contain_computed_numbers(big):
    out = ins.generate_insights(big, min_orders=100)
    n, k = len(big), int(big["is_late"].sum())
    overview = _by_kind(out, "overview")[0]
    assert f"{n:,}" in overview.text and f"{100 * k / n:.1f}%" in overview.text
    assert overview.metrics["n_orders"] == n and overview.n == n
    rj = _by_kind(out, "region_high")[0]
    assert rj.metrics["n_flagged"] == 1 and rj.metrics["segments"][0]["customer_state"] == "RJ"
    assert "RJ" in rj.text and "18.0%" in rj.text and "n = 600" in rj.text
    assert "AC" not in rj.text and "50.0%" not in rj.text  # n = 20 segment is never ranked
    assert rj.severity in ("watch", "alert")
    low = _by_kind(out, "region_low")[0]
    assert low.metrics["segments"][0]["customer_state"] == "SP" and "10.0%" in low.text
    test = _by_kind(out, "region_test")[0]
    chi = sa.chi_square_test(big, "customer_state")
    assert f"{chi['chi2']:.1f}" in test.text and f"{chi['cramers_v']:.3f}" in test.text
    assert test.metrics["p_value"] == chi["p_value"]
    assert (test.metrics["min_late_rate"], test.metrics["max_late_rate"]) == (0.1, 0.18)  # SP .. RJ
    assert "ranges from 10.0% (SP (" in test.text and "n = 900) to 18.0% (RJ (" in test.text


def test_insights_month_screen_flags_the_peak_month(big):
    out = ins.generate_insights(big, min_orders=100)
    mh = _by_kind(out, "month_high")[0]
    assert [m["month"] for m in mh.metrics["months"]] == ["2017-03"]
    assert "24.0%" in mh.text and "n = 500" in mh.text and "Bonferroni" in mh.text
    rest_rate = (208 - 120) / (1520 - 500)
    assert f"{100 * rest_rate:.1f}%" in mh.text
    ext = _by_kind(out, "month_extremes")[0]
    assert ext.metrics["worst_month"] == "2017-03" and ext.metrics["best_month"] in ("2017-01", "2017-02")


def test_insights_other_kinds_present_and_retrospective_label(big):
    out = ins.generate_insights(big, min_orders=100)
    kinds = {i.kind for i in out}
    assert {"category_duration", "category_delay", "category_freight", "pair_high", "same_state",
            "weight_freight", "processing_time"} <= kinds
    proc = _by_kind(out, "processing_time")[0]
    assert "retrospective" in proc.text.lower() and proc.metrics["retrospective"] is True
    wf = _by_kind(out, "weight_freight")[0]
    assert "rho" in wf.text and "causes" in wf.text
    pair = _by_kind(out, "pair_high")[0]
    assert "SP→RJ" in pair.text and "SP→AC" not in pair.text
    cat = _by_kind(out, "category_delay")[0]
    # books (n = 700, 144 late = 20.6%) outranks toys (n = 820, 64 late = 7.8%) on late rate
    assert [s["product_category"] for s in cat.metrics["segments"]] == ["books", "toys"]
    dur = _by_kind(out, "category_duration")[0]
    assert [s["product_category"] for s in dur.metrics["segments"]] == ["books", "toys"]
    assert dur.metrics["segments"][0]["mean_duration"] > dur.metrics["segments"][1]["mean_duration"]


def test_insights_change_when_data_change(big):
    full = ins.generate_insights(big, min_orders=100)
    sp_only = ins.generate_insights(an.apply_filters(big, states=["SP"]), min_orders=100)
    assert _by_kind(sp_only, "overview")[0].text != _by_kind(full, "overview")[0].text
    assert _by_kind(sp_only, "overview")[0].n == 900
    assert _by_kind(sp_only, "region_high")[0].metrics["n_flagged"] == 0  # only one region left
    assert [i.text for i in full] != [i.text for i in sp_only]
    march = ins.generate_insights(an.apply_filters(big, date_range=("2017-03-01", "2017-03-31")), min_orders=100)
    assert "month_high" not in {i.kind for i in march}  # a single month cannot be compared
    assert any(i.kind == "insufficient_data" and "month" in i.text for i in march)


def test_insights_deterministic(big):
    a = ins.generate_insights(big, min_orders=100)
    b = ins.generate_insights(big, min_orders=100)
    assert [(i.kind, i.text, i.n, i.severity) for i in a] == [(i.kind, i.text, i.n, i.severity) for i in b]


def test_recommendations_are_cautious(big):
    out = ins.generate_insights(big, min_orders=100)
    recs = ins.recommendations(out)
    assert len(recs) >= 3
    text = " ".join(recs).lower()
    assert "investigate" in text or "review" in text
    for banned in ("caused by", "root cause", "because of", "is responsible"):
        assert banned not in text
    assert "associations" in recs[-1]
    assert all(isinstance(r, str) and r for r in recs)
    assert ins.recommendations([]) == []


def test_summary_report_content(big):
    out = ins.generate_insights(big, min_orders=100)
    rep = ins.summary_report(big, "All dates; all states", out)
    assert rep.startswith("# DeliveryIQ")
    assert "All dates; all states" in rep
    assert f"{len(big):,}" in rep and "13.7%" in rep  # 208 / 1520
    assert "## Recommendations" in rep and "## Methodology and caveats" in rep
    assert "Prediction model" not in rep  # no model metrics supplied
    for i in out:
        if i.kind != "overview":
            assert i.title in rep
    assert "retrospective" in rep.lower()
    empty = ins.summary_report(big.iloc[0:0], "", ins.generate_insights(big.iloc[0:0]))
    assert "n = 0" in empty and "No data" in empty


def test_summary_report_model_metrics_tolerant(big):
    out = ins.generate_insights(big.iloc[:1], min_orders=100)
    full = {"selected_model": "hist_gb", "selection_rationale": "best validation PR-AUC",
            "models": {"dummy": {"test": {"roc_auc": 0.5, "pr_auc": 0.07}},
                       "hist_gb": {"test": {"accuracy": 0.93, "ROC-AUC": 0.71, "PR-AUC": 0.2, "f1": 0.25,
                                            "brier": 0.06}}},
            "leakage_audit": {"passed": True}}
    rep = ins.summary_report(big, "x", out, full)
    assert "Prediction model" in rep and "hist_gb" in rep and "0.710" in rep and "best validation PR-AUC" in rep
    assert "passed: True" in rep
    nested = {"selected_model": {"name": "m1", "rationale": "chosen on validation PR-AUC"},
              "models": {"m1": {"validation": {"roc_auc": 0.6}, "test": {"roc_auc": 0.65, "pr_auc": None}}},
              "split": {"windows": {"train": ["2017-01-01", "2018-01-01"], "test": ["2018-03-01", None]},
                        "monthly_late_rate": [{"m": "2017-01", "rate": 0.1}] * 50},
              "risk_bands": {"definition": "High = top 10% of validation scores", "validation": [{"a": 1}]}}
    rep2 = ins.summary_report(big, "x", out, nested)
    assert "m1" in rep2 and "0.650" in rep2 and "chosen on validation PR-AUC" in rep2
    assert "2018-03-01" in rep2 and "High = top 10%" in rep2 and "monthly_late_rate" not in rep2
    for weird in (None, {}, {"unrelated": 1}, {"models": "nope"}, {"selected_model": {"name": "m"}},
                  {"models": {"a": 5, "b": {"test": "x"}}}):
        assert isinstance(ins.summary_report(big, "x", out, weird), str)
    assert "no recognised fields" in ins.summary_report(big, "x", out, {"unrelated": 1})


def test_regional_report(big):
    rr = ins.regional_report(big, min_orders=100)
    assert list(rr.columns)[:3] == ["customer_state", "state_name", "n_orders"]
    assert rr["n_orders"].sum() == len(big)
    assert list(rr["customer_state"]) == ["AC", "RJ", "SP"]  # strictly by late rate: 50%, 18%, 10%
    flags = rr.set_index("customer_state")["flag"].to_dict()
    assert flags["RJ"] == "Above overall (interval excludes it)" and flags["SP"] == "Below overall (interval excludes it)"
    assert flags["AC"].startswith("Insufficient data")
    assert rr.set_index("customer_state").loc["RJ", "diff_vs_overall_pp"] == pytest.approx(
        (108 / 600 - 208 / 1520) * 100)
    assert rr.set_index("customer_state").loc["RJ", "state_name"] == "Rio de Janeiro"
    e = ins.regional_report(big.iloc[0:0], 100)
    assert e.empty and "flag" in e.columns
    csv = rr.to_csv(index=False)
    assert csv.splitlines()[0].startswith("customer_state,state_name")


# --------------------------------------------------------------------------- #
# Audit round: known-answer checks for numeric branches, wording, severity and ordering
# --------------------------------------------------------------------------- #
def _holm_reference(p):
    """Independent Holm step-down implementation (loop over sorted p-values)."""
    idx = sorted((i for i, v in enumerate(p) if not math.isnan(v)), key=lambda i: p[i])
    out, run, m = [math.nan] * len(p), 0.0, len(idx)
    for rank, i in enumerate(idx):
        run = max(run, (m - rank) * p[i])
        out[i] = min(1.0, run)
    return out


def test_holm_adjust_known_answer():
    # sorted: 0.005 -> 4*0.005 = 0.02; 0.01 -> 3*0.01 = 0.03; 0.03 -> 2*0.03 = 0.06; 0.04 -> max(0.06, 0.04)
    adj = sa.holm_adjust([0.01, 0.04, 0.03, 0.005])
    assert adj.tolist() == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert sa.holm_adjust([0.5, 0.9]).tolist() == pytest.approx([1.0, 1.0])  # capped at 1
    assert sa.holm_adjust([0.2, 0.9]).tolist() == pytest.approx([0.4, 0.9])


def test_run_all_tests_uses_holm_not_raw_p(big):
    s = sa.run_all_tests(big)["summary"]
    assert s["p_holm"].fillna(-1).tolist() == pytest.approx(
        pd.Series(_holm_reference(s["p_value"].tolist())).fillna(-1).tolist())
    ok = s["p_value"].notna() & (s["p_value"] > 0)
    assert (s.loc[ok, "p_holm"] > s.loc[ok, "p_value"]).any()  # adjustment really changed something


def test_run_all_tests_significance_follows_holm_adjusted_p():
    # a 300-row subsample where several raw p-values are < 0.05 but no Holm-adjusted one is
    d = make_blocks(BLOCKS).sample(n=300, random_state=1)
    res = sa.run_all_tests(d)
    s = res["summary"]
    raw_sig = s["p_value"] < 0.05
    assert (raw_sig & ~s["significant_holm"]).any()  # raw-significant yet not significant after Holm
    cs = s.set_index("variables").loc["customer_state x is_late"]
    assert cs["p_value"] < 0.05 <= cs["p_holm"] and not cs["significant_holm"]
    assert (s["significant_holm"] == (s["p_holm"] < res["alpha"])).all()
    assert res["chi_square"]["customer_state"]["significant"] is True  # per-test flag is unadjusted
    for alpha in (0.5,):
        s2 = sa.run_all_tests(d, alpha=alpha)["summary"]
        assert (s2["significant_holm"] == (s2["p_holm"] < alpha)).all()


def test_label_thresholds_at_boundaries():
    # Cramer's V (min_dim = 1): 0.1 / 0.3 / 0.5 are the lower edges of small / medium / large
    cut = [(0.0999, "negligible"), (0.1, "small"), (0.2999, "small"), (0.3, "medium"),
           (0.4999, "medium"), (0.5, "large")]
    assert [sa.cramers_v_label(v, 1) for v, _ in cut] == [lab for _, lab in cut]
    assert sa.cramers_v_label(0.0499, 4) == "negligible" and sa.cramers_v_label(0.05, 4) == "small"
    assert sa.cramers_v_label(0.1, 0) == "undefined"
    cut = [(0.0999, "negligible"), (0.1, "weak"), (0.2999, "weak"), (0.3, "moderate"), (0.4999, "moderate"),
           (0.5, "strong"), (0.6999, "strong"), (0.7, "very strong"), (1.0, "very strong")]
    assert [sa.spearman_strength(r) for r, _ in cut] == [lab for _, lab in cut]
    assert [sa.spearman_strength(-r) for r, _ in cut] == [lab for _, lab in cut]  # sign is ignored
    assert sa.spearman_strength(float("nan")) == "undefined"
    for fn, edges, labels in (
            (sa._rank_biserial_label, (0.1, 0.3, 0.5), ("negligible", "small", "medium", "large")),
            (sa._epsilon_squared_label, (0.01, 0.08, 0.26), ("negligible", "small", "medium", "large"))):
        eps = 1e-9
        for edge, below, above in zip(edges, labels[:-1], labels[1:]):
            assert fn(edge - eps) == below and fn(edge) == above
    assert sa._rank_biserial_label(-0.6) == "large" and sa._epsilon_squared_label(float("nan")) == "undefined"


def test_chi_square_adjusted_residuals_and_alpha():
    counts = {"A": (70, 30), "B": (50, 50), "C": (80, 20)}
    res = sa.chi_square_test(_df_from_counts(counts), "g")
    obs = np.array(list(counts.values()), dtype=float)
    n = obs.sum()
    exp = obs.sum(1, keepdims=True) * obs.sum(0, keepdims=True) / n
    adj = (obs - exp) / np.sqrt(exp * (1 - obs.sum(1, keepdims=True) / n) * (1 - obs.sum(0, keepdims=True) / n))
    pearson = (obs - exp) / np.sqrt(exp)
    assert res["residuals"].to_numpy() == pytest.approx(adj) and not np.allclose(adj, pearson)
    ls = res["level_summary"].set_index("level")
    assert ls.loc["B", "adj_residual"] == pytest.approx(adj[1, 1])  # late column
    assert ls.loc["A", "late_rate"] == pytest.approx(0.3) and list(res["level_summary"]["level"])[0] == "B"
    # alpha decides `significant`: this table has p ~ 0.03
    mild = _df_from_counts({"x": (90, 10), "y": (80, 20)})
    p = sa.chi_square_test(mild, "g")["p_value"]
    assert 0.01 < p < 0.05
    assert sa.chi_square_test(mild, "g", alpha=0.05)["significant"] is True
    assert sa.chi_square_test(mild, "g", alpha=0.01)["significant"] is False
    assert sa.chi_square_test(mild, "g", alpha=0.01)["alpha"] == 0.01


def test_chi_square_level_exactly_at_threshold_is_not_rare():
    # n = 1000, 100 late: a level needs 5 * 1000 / 100 = 50 orders for an expected late count of 5
    counts = {"A": (600, 55), "B": (255, 40), "C": (45, 5)}
    res = sa.chi_square_test(_df_from_counts(counts), "g", min_expected=5)
    assert res["n"] == 1000 and res["merged_rare_levels"] == [] and res["n_levels_tested"] == 3
    assert res["min_expected_after"] == pytest.approx(5.0)
    below = sa.chi_square_test(_df_from_counts({"A": (600, 55), "B": (256, 40), "C": (44, 5)}), "g")
    assert "C" in below["merged_rare_levels"]  # 49 < 50


def test_kruskal_effect_size_and_group_filter():
    df3 = pd.DataFrame({"v": [1, 2, 3, 10, 11, 12, 20, 21, 22.0], "g": list("aaabbbccc")})
    r3 = sa.kruskal_or_mannwhitney(df3, "v", "g")
    h, k, n = r3["statistic"], 3, 9
    assert h == pytest.approx(7.2) and r3["effect_size"] == pytest.approx(h / (n - 1))  # epsilon^2 = 0.9
    assert r3["effect_label"] == "large" and r3["significant"] is True
    # moderately separated groups: epsilon^2 in the "small" band
    rng = np.random.default_rng(0)
    v = np.concatenate([rng.normal(0, 1, 200), rng.normal(0.3, 1, 200), rng.normal(0.6, 1, 200)])
    r = sa.kruskal_or_mannwhitney(pd.DataFrame({"v": v, "g": np.repeat(list("abc"), 200)}), "v", "g")
    assert r["effect_size"] == pytest.approx(r["statistic"] / (600 - 1))
    assert r["effect_label"] == sa._epsilon_squared_label(r["effect_size"]) == "small"
    # min_group_size drops small groups (and the rows in them) before testing
    df = pd.DataFrame({"v": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10.0, 50.0, 60.0], "g": ["a"] * 5 + ["b"] * 5 + ["c"] * 2})
    kept = sa.kruskal_or_mannwhitney(df, "v", "g", min_group_size=3)
    assert kept["test"] == "mann_whitney" and kept["n_groups"] == 2 and kept["n"] == 10
    assert kept["group_summary"]["group"].tolist() == ["a", "b"]
    allg = sa.kruskal_or_mannwhitney(df, "v", "g", min_group_size=2)
    assert allg["test"] == "kruskal" and allg["n_groups"] == 3 and allg["n"] == 12
    assert math.isnan(sa.kruskal_or_mannwhitney(df, "v", "g", min_group_size=6)["p_value"])
    # Mann-Whitney: a fully separated pair has rank-biserial -1 (first level below second)
    sep = sa.kruskal_or_mannwhitney(df[df["g"] != "c"], "v", "g")
    assert sep["effect_size"] == pytest.approx(-1.0) and sep["effect_label"] == "large"
    assert sa.kruskal_or_mannwhitney(df[df["g"] != "c"], "v", "g", alpha=1e-9)["significant"] is False
    assert sep["significant"] is True


def test_plain_language_wording_follows_significance():
    sig = sa.chi_square_test(_df_from_counts({"A": (70, 30), "B": (50, 50)}), "g")
    ns = sa.chi_square_test(_df_from_counts({"A": (90, 10), "B": (90, 10)}), "g")
    assert sig["significant"] and not ns["significant"]
    assert "a statistically detectable association" in sa.plain_language(sig)
    assert "no statistically detectable association" in sa.plain_language(ns)
    assert "no statistically detectable" not in sa.plain_language(sig)
    x = np.arange(40.0)
    rho_sig = sa.spearman_test(pd.DataFrame({"a": x, "b": x[::-1] + (x % 3)}), "a", "b")
    rho_ns = sa.spearman_test(pd.DataFrame({"a": x, "b": (x * 7) % 11}), "a", "b")
    assert rho_sig["significant"] and not rho_ns["significant"]
    assert "i.e. statistically detectable" in sa.plain_language(rho_sig)
    assert "i.e. not statistically detectable" in sa.plain_language(rho_ns)
    assert "negative" in sa.plain_language(rho_sig)
    grp = pd.DataFrame({"v": list(range(10)), "g": list("aaaaabbbbb")})
    g_sig = sa.kruskal_or_mannwhitney(grp, "v", "g")
    g_ns = sa.kruskal_or_mannwhitney(pd.DataFrame({"v": [1, 2, 3, 4, 5, 2, 1, 4, 3, 5.0], "g": list("aaaaabbbbb")}),
                                     "v", "g")
    assert g_sig["significant"] and not g_ns["significant"]
    assert "the distributions differ detectably" in sa.plain_language(g_sig)
    assert "the distributions do not differ detectably" in sa.plain_language(g_ns)
    assert "Mann-Whitney U" in sa.plain_language(g_sig)
    # proportion difference is reported in percentage points (x 100) and at the requested level
    pd_txt = sa.plain_language(sa.proportion_diff_ci(56, 70, 48, 80))
    assert "+20.0 percentage points" in pd_txt and "95% CI +5.2 to +33.4" in pd_txt
    assert "80.0%" in pd_txt and "60.0%" in pd_txt
    assert "99% CI" in sa.plain_language(sa.proportion_diff_ci(56, 70, 48, 80, alpha=0.01))
    sp99 = sa.spearman_test(pd.DataFrame({"a": x, "b": x[::-1] + (x % 3)}), "a", "b", alpha=0.01)
    assert "approximate 99% CI" in sa.plain_language(sp99) and "95%" not in sa.plain_language(sp99)


def test_top_n_pool_is_not_forced_to_two_levels():
    # one level beyond top_n -> a single-level pool is fine when nothing is rare
    res = sa.chi_square_test(_df_from_counts({"A": (800, 200), "B": (900, 100), "C": (600, 300)}), "g", top_n=2)
    assert len(res["merged_rare_levels"]) == 1
    # but a rare single level is still pooled with the next smallest one
    res2 = sa.chi_square_test(_df_from_counts({"A": (800, 200), "B": (900, 100), "C": (17, 3)}), "g", top_n=3)
    assert "C" in res2["merged_rare_levels"] and len(res2["merged_rare_levels"]) == 2


# ---- analytics numeric branches ------------------------------------------------------------
def test_category_cost_comparison_late_rate_and_duration(fx):
    t = an.category_cost_comparison(fx, min_orders=1).set_index("product_category")
    assert t.loc["toys", "late_rate"] == pytest.approx(4 / 6) and t.loc["books", "late_rate"] == pytest.approx(1 / 6)
    assert t.loc["toys", "mean_duration"] == pytest.approx(99 / 6)  # 5 + 7 + 12 + 20 + 25 + 30
    assert t.loc["books", "mean_duration"] == pytest.approx(55 / 6)  # 6 + 4 + 9 + 8 + 6 + 22
    assert t.loc["toys", "mean_price"] == pytest.approx(380 / 6) and t.loc["toys", "mean_weight_g"] == pytest.approx(12300 / 6)
    assert t["n_orders"].tolist() == [6, 6]
    assert an.category_cost_comparison(fx, min_orders=7)["reliable"].sum() == 0


def test_processing_summary_known_answers(fx):
    p = an.processing_summary(fx)
    proc = np.array([1, 2, 3, 1, 1, 2, 4, 5, 1, 2, 6], dtype=float)  # o_t11 has no value
    assert p["p90"] == pytest.approx(np.percentile(proc, 90)) and p["p90"] != p["median"]
    assert p["share_of_duration"] == pytest.approx(28 / 124)  # processing sum / duration sum (o_t11 excluded)
    assert p["mean_carrier_to_customer_days"] == pytest.approx((124 - 28) / 11)
    assert math.isnan(an.processing_summary(fx.iloc[0:0])["mean_carrier_to_customer_days"])
    # share_of_duration only counts orders with BOTH values
    d = fx.copy()
    d.loc[0, "delivery_duration_days"] = np.nan
    assert an.processing_summary(d)["share_of_duration"] == pytest.approx(27 / 119)


def test_high_delay_segments_boundaries_alpha_and_order():
    perf = pd.DataFrame({"seg": list("abcd"), "n_orders": [150, 1000, 400, 1000], "n_late": [18, 150, 80, 60]})
    perf["late_rate"] = perf["n_late"] / perf["n_orders"]
    # min_orders is inclusive: b has exactly 1000 orders
    assert "b" in set(an.high_delay_segments(perf, 1000, 0.10)["seg"])
    assert "b" not in set(an.high_delay_segments(perf, 1001, 0.10)["seg"])
    # most extreme first (high: largest excess first; low: most negative first)
    hi = an.high_delay_segments(perf, 100, 0.10)
    assert list(hi["seg"]) == ["c", "b"] and hi["excess_vs_overall"].is_monotonic_decreasing
    lo_ = an.high_delay_segments(perf, 100, 0.10, direction="low")
    assert list(lo_["seg"]) == ["d"] and lo_["excess_vs_overall"].iloc[0] == pytest.approx(-0.04)
    # alpha is used: a 50% interval flags a (12%, n = 150), the default 95% interval does not
    assert "a" not in set(an.high_delay_segments(perf, 100, 0.10, alpha=0.05)["seg"])
    assert "a" in set(an.high_delay_segments(perf, 100, 0.10, alpha=0.5)["seg"])
    # the comparison is strict: overall exactly equal to the Wilson bound is not flagged
    bound = an.wilson_ci(150, 1000)[0]
    assert "b" not in set(an.high_delay_segments(perf, 100, bound)["seg"])
    assert "b" in set(an.high_delay_segments(perf, 100, bound - 1e-9)["seg"])
    upper = an.wilson_ci(60, 1000)[1]
    assert "d" not in set(an.high_delay_segments(perf, 100, upper, direction="low")["seg"])
    assert "d" in set(an.high_delay_segments(perf, 100, upper + 1e-9, direction="low")["seg"])


def test_binned_stat_ignores_infinite_values(fx):
    d = fx.copy()
    d["x"] = np.arange(12, dtype=float)
    d.loc[0, "x"] = np.inf
    d.loc[1, "x"] = -np.inf
    t = an.binned_stat(d, "x", [0, 5, np.inf])
    assert t["n_orders"].sum() == 10 and np.isfinite(t["mean_value"]).all()
    assert t["n_orders"].tolist() == [4, 6]  # values 2..5 and 6..11
    q = an.binned_stat(d, "x", bins=2, quantile=True)
    assert q["n_orders"].sum() == 10 and np.isfinite(q["bin_high"]).all()


def test_state_matrix_axis_order_by_volume(fx):
    m = an.state_matrix(fx)
    assert list(m["n_orders"].index) == ["SP", "MG", "RJ"]  # seller volume 8, 2, 2 (ties alphabetical)
    assert list(m["n_orders"].columns) == ["SP", "RJ", "BA"]  # customer volume 6, 4, 2
    for k in ("late_rate", "mean_duration"):
        assert list(m[k].index) == ["SP", "MG", "RJ"] and list(m[k].columns) == ["SP", "RJ", "BA"]
    assert list(an.state_matrix(fx, top_n=2)["n_orders"].columns) == ["SP", "RJ"]


def test_scatter_sample_keeps_original_row_order(big):
    s = an.scatter_sample(big, "total_weight_g", "delivery_duration_days", n=300, extra=("order_id",))
    assert len(s) == 300 and s["order_id"].is_monotonic_increasing  # ids are zero padded: row order
    assert s["order_id"].is_unique


def test_same_state_comparison_reliable_uses_min_orders(fx):
    t = an.same_state_comparison(fx, min_orders=6).set_index("group")
    assert t["reliable"].to_dict() == {"cross_state": True, "same_state": True}
    t = an.same_state_comparison(fx, min_orders=7).set_index("group")
    assert not t["reliable"].any()
    assert an.same_state_comparison(fx)["reliable"].all()  # default min_orders = 0
    lo, hi = an.wilson_ci(5, 6, 0.01)
    t99 = an.same_state_comparison(fx, alpha=0.01).set_index("group")
    assert t99.loc["cross_state", "late_ci_low"] == pytest.approx(lo)
    assert t99.loc["cross_state", "late_ci_high"] == pytest.approx(hi)


def test_table_functions_pass_alpha_through(fx):
    for t, key, val in ((an.category_performance(fx, alpha=0.01), "product_category", "toys"),
                        (an.state_performance(fx, alpha=0.01), "customer_state", "RJ")):
        r = t.set_index(key)
        k, n = (4, 6) if val == "toys" else (2, 4)
        assert r.loc[val, "late_ci_low"] == pytest.approx(an.wilson_ci(k, n, 0.01)[0])
    assert an.state_pair_performance(fx, alpha=0.01).set_index("state_pair").loc["SP→RJ", "late_ci_low"] == \
        pytest.approx(an.wilson_ci(2, 2, 0.01)[0])
    assert an.monthly_performance(fx, alpha=0.01).set_index("purchase_year_month").loc["2017-01", "late_ci_low"] == \
        pytest.approx(an.wilson_ci(3, 6, 0.01)[0])


def test_unknown_status_label_warns_and_valid_ones_do_not(fx):
    import warnings
    with pytest.warns(UserWarning, match="order_status"):
        assert an.apply_filters(fx, statuses=["delivered"]).empty
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert len(an.apply_filters(fx, statuses=["Late", "on-time"])) == 12


# ---- insights: severity, wording, ordering, screening ---------------------------------------
def _blocks(rows):
    return make_blocks(rows)


def test_insight_severity_alert_vs_watch_follows_alert_ratio(big):
    # regional / route screens: ratio = worst rate / overall rate
    watch = ins.generate_insights(big, 100)
    rj, pair = _by_kind(watch, "region_high")[0], _by_kind(watch, "pair_high")[0]
    assert rj.metrics["segments"][0]["rate_ratio"] < ins.ALERT_RATIO and rj.severity == "watch"
    assert pair.metrics["segments"][0]["rate_ratio"] < ins.ALERT_RATIO and pair.severity == "watch"
    hot = _blocks([("SP", "SP", "2017-01", "toys", 500, 25), ("SP", "SP", "2017-02", "toys", 400, 20),
                   ("RJ", "SP", "2017-01", "toys", 300, 100), ("RJ", "SP", "2017-02", "toys", 300, 100)])
    out = ins.generate_insights(hot, 100)
    rj, pair = _by_kind(out, "region_high")[0], _by_kind(out, "pair_high")[0]
    assert rj.metrics["segments"][0]["rate_ratio"] >= ins.ALERT_RATIO and rj.severity == "alert"
    assert pair.metrics["segments"][0]["rate_ratio"] >= ins.ALERT_RATIO and pair.severity == "alert"
    assert _by_kind(out, "pair_low")[0].severity == "info" and _by_kind(out, "region_low")[0].severity == "info"
    assert _by_kind(watch, "pair_low")[0].severity == "info"


def test_insight_severity_month_alert_and_watch(big):
    # the 2017-03 peak is 24.0% vs 8.6% in the other months (ratio 2.8) -> alert
    mh = _by_kind(ins.generate_insights(big, 100), "month_high")[0]
    top = mh.metrics["months"][0]
    assert top["late_rate"] / top["rest_late_rate"] >= ins.ALERT_RATIO and mh.severity == "alert"
    # a peak month at 14% vs 8% elsewhere (ratio 1.75) is reliable but below the alert ratio
    rows = [("SP", "SP", f"2017-{m:02d}", "toys", 1000, 80) for m in range(1, 6)]
    rows.append(("SP", "SP", "2017-06", "toys", 1000, 140))
    mh2 = _by_kind(ins.generate_insights(_blocks(rows), 100), "month_high")[0]
    assert [m["month"] for m in mh2.metrics["months"]] == ["2017-06"] and mh2.metrics["n_flagged"] == 1
    assert mh2.severity == "watch" and mh2.metrics["months"][0]["late_rate"] / \
        mh2.metrics["months"][0]["rest_late_rate"] < ins.ALERT_RATIO


def test_month_screen_uses_bonferroni_adjusted_p():
    # one month at 14.7% among 12: raw p ~ 0.02 (< 0.05) but x 12 months is no longer significant
    rows = [("SP", "SP", f"2017-{m:02d}", "toys", 300, 30) for m in range(1, 12)]
    rows.append(("SP", "SP", "2017-12", "toys", 300, 44))
    d = _blocks(rows)
    monthly = an.monthly_performance(d, 100)
    scr = ins._month_screen(monthly, 0.05)
    assert len(scr) == 12
    peak = scr.set_index("month").loc["2017-12"]
    assert peak["p_value"] < 0.05 and peak["p_bonferroni"] > 0.05
    assert peak["p_bonferroni"] == pytest.approx(min(1.0, peak["p_value"] * 12))
    z = sa.two_proportion_ztest(44, 300, 330, 3300)
    assert peak["p_value"] == pytest.approx(z["p_value"]) and peak["rest_late_rate"] == pytest.approx(330 / 3300)
    mh = _by_kind(ins.generate_insights(d, 100), "month_high")[0]
    assert mh.metrics["n_flagged"] == 0 and mh.metrics["n_tested"] == 12 and mh.severity == "info"
    assert "No month stands out" in mh.title


def test_region_test_severity_requires_nonnegligible_effect(big):
    assert _by_kind(ins.generate_insights(big, 100), "region_test")[0].severity == "watch"
    same = _blocks([("SP", "SP", "2017-01", "toys", 900, 90), ("RJ", "SP", "2017-01", "toys", 600, 60)])
    rt = _by_kind(ins.generate_insights(same, 100), "region_test")[0]
    assert rt.metrics["p_value"] == pytest.approx(1.0) and rt.severity == "info"
    # significant but negligible (V ~ 0.02 with n = 20,000): still only "info"
    huge = _blocks([("SP", "SP", "2017-01", "toys", 10000, 1000), ("RJ", "SP", "2017-01", "toys", 10000, 1130)])
    rt = _by_kind(ins.generate_insights(huge, 100), "region_test")[0]
    assert rt.metrics["p_value"] < 0.05 and rt.metrics["effect_label"] == "negligible" and rt.severity == "info"


def test_same_state_insight_numbers_and_severity(big):
    ss = _by_kind(ins.generate_insights(big, 100), "same_state")[0]
    m = ss.metrics
    # cross-state: RJ (600, 108 late) + AC (20, 10) = 620 orders, 118 late; same-state: SP->SP 900, 90 late
    assert (m["cross_n"], m["same_n"]) == (620, 900)
    assert m["cross_late_rate"] == pytest.approx(118 / 620) and m["same_late_rate"] == pytest.approx(0.10)
    assert m["diff"] == pytest.approx(118 / 620 - 0.10) and m["diff_ci_low"] > 0 and ss.severity == "watch"
    assert ss.text.index("Cross-state") < ss.text.index("Same-state")
    assert "19.0%" in ss.text.split("Same-state")[0] and "10.0%" in ss.text.split("Same-state")[1]
    flat = _blocks([("SP", "SP", "2017-01", "toys", 600, 60), ("RJ", "SP", "2017-01", "toys", 600, 60)])
    f = _by_kind(ins.generate_insights(flat, 100), "same_state")[0]
    assert f.severity == "info" and f.metrics["p_value"] == pytest.approx(1.0)
    # a group below min_orders is not compared (reliable follows min_orders, not a fixed 1)
    out = ins.generate_insights(big, 700)
    assert "same_state" not in {i.kind for i in out}
    assert any("same-state vs cross-state" in i.text for i in out if i.kind == "insufficient_data")


def test_weight_freight_severity_needs_moderate_and_significant(big):
    assert _by_kind(ins.generate_insights(big, 100), "weight_freight")[0].severity == "info"
    rng = np.random.default_rng(3)
    for noise_sd, expected in ((500.0, "watch"), (2000.0, "info")):  # rho ~ 0.43 vs ~ 0.15
        d = big.copy()
        d["total_weight_g"] = d["delivery_duration_days"] * 100 + rng.normal(0, noise_sd, len(d))
        wf = _by_kind(ins.generate_insights(d, 100), "weight_freight")[0]
        w = wf.metrics["weight_vs_duration"]
        if expected == "watch":
            assert abs(w["rho"]) >= 0.3
        else:  # significant (p < 0.05) but weak (< 0.3): "info"
            assert w["p_value"] < 0.05 and 0.05 < abs(w["rho"]) < 0.3
        assert wf.severity == expected
    assert "approx. 95% CI" in wf.text


def test_processing_time_severity_and_na_labels(big):
    pt = _by_kind(ins.generate_insights(big, 100), "processing_time")[0]
    assert pt.metrics["diff_ci_low"] > 0 and pt.severity == "watch"
    d = big.copy()
    d["processing_days"] = 1.0 + (np.arange(len(d)) % 4)  # unrelated to lateness
    pt2 = _by_kind(ins.generate_insights(d, 100), "processing_time")[0]
    assert pt2.metrics["diff_ci_low"] <= 0 and pt2.severity == "info"
    # rows with an unknown is_late are never counted, and never raise
    d = big.copy()
    d.loc[d.index[:5], "is_late"] = pd.NA
    out = ins.generate_insights(d, 100)
    assert _by_kind(out, "overview")[0].n == len(big) - 5
    pt3 = _by_kind(out, "processing_time")[0]
    assert pt3.n == pt3.metrics["slow_n"] + pt3.metrics["rest_n"] <= len(big) - 5
    ref = ins.generate_insights(d.dropna(subset=["is_late"]), 100)
    assert _by_kind(ref, "processing_time")[0].metrics == pt3.metrics
    allna = big.copy()
    allna["is_late"] = pd.NA
    assert [i.kind for i in ins.generate_insights(allna, 100)] == ["no_data"]


# categories: audio (reliably high), fashion (high rate, wide interval), books, toys, plus 'unknown'
CAT_BLOCKS = [("SP", "SP", "2017-01", "audio", 1000, 200), ("SP", "SP", "2017-01", "fashion", 110, 11),
              ("SP", "SP", "2017-02", "books", 2000, 140), ("SP", "SP", "2017-02", "toys", 2000, 100),
              ("SP", "SP", "2017-03", "unknown", 300, 150)]


def test_category_insights_ordering_unknown_and_flagged_only_recommendation():
    d = _blocks(CAT_BLOCKS)
    out = ins.generate_insights(d, 100)
    overall = int(d["is_late"].sum()) / len(d)
    cd = _by_kind(out, "category_delay")[0]
    names = [s["product_category"] for s in cd.metrics["segments"]]
    assert names == ["audio", "fashion", "books"]  # late rate descending; 'unknown' (50%) excluded
    rates = [s["late_rate"] for s in cd.metrics["segments"]]
    assert rates == sorted(rates, reverse=True)
    # fashion (10.0%, n = 110) has a Wilson interval that contains the overall rate -> not flagged
    lo, _ = an.wilson_ci(11, 110)
    assert 11 / 110 > (overall * len(d) - 150) / (len(d) - 300) and lo < overall
    assert cd.metrics["flagged"] == ["audio"] and cd.metrics["n_reliably_above"] == 1
    assert cd.text.count("[reliably above overall]") == 1
    assert cd.text.index("audio") < cd.text.index("[reliably above overall]") < cd.text.index("fashion")
    assert "unknown" not in cd.text and cd.metrics["n_screened"] == 4
    cdur = _by_kind(out, "category_duration")[0]
    durs = [s["mean_duration"] for s in cdur.metrics["segments"]]
    assert durs == sorted(durs, reverse=True) and "unknown" not in [s["product_category"] for s in cdur.metrics["segments"]]
    assert cdur.metrics["segments"][0]["product_category"] == "audio"
    cf = _by_kind(out, "category_freight")[0]
    assert "unknown" not in [s["product_category"] for s in cf.metrics["segments"]]
    rec = [r for r in ins.recommendations(out) if "reliably higher late rates" in r]
    assert len(rec) == 1 and "audio" in rec[0] and "fashion" not in rec[0] and "books" not in rec[0]
    # only unflagged categories lead the ranking -> no recommendation, severity info
    flat = _blocks([("SP", "SP", "2017-01", "a", 500, 50), ("SP", "SP", "2017-02", "b", 500, 50),
                    ("SP", "SP", "2017-03", "c", 500, 50)])
    out2 = ins.generate_insights(flat, 100)
    cd2 = _by_kind(out2, "category_delay")[0]
    assert cd2.severity == "info" and cd2.metrics["flagged"] == []
    assert not [r for r in ins.recommendations(out2) if "reliably higher late rates" in r]


def test_recommendation_lists_only_flagged_categories_real_style_case():
    d = _blocks(CAT_BLOCKS)
    out = ins.generate_insights(an.apply_filters(d, date_range=("2017-01-01", "2017-02-28")), 100)
    for r in ins.recommendations(out):
        if "reliably higher late rates" in r:
            flagged = _by_kind(out, "category_delay")[0].metrics["flagged"]
            assert flagged and all(c in r for c in flagged)
            for c in ("fashion", "books", "toys"):
                assert c not in r


def test_segment_screens_state_the_multiple_comparison_caveat(big):
    out = ins.generate_insights(big, 100)
    states = an.state_performance(big, 100)
    for kind, n_exp in (("region_high", int(states["reliable"].sum())),
                        ("region_low", int(states["reliable"].sum())),
                        ("pair_high", int(an.state_pair_performance(big, 100)["reliable"].sum())),
                        ("pair_low", int(an.state_pair_performance(big, 100)["reliable"].sum())),
                        ("category_delay", 2)):
        i = _by_kind(out, kind)[0]
        assert f"Unadjusted 95% intervals across {n_exp:,} segments screened" in i.text, kind
        assert "leads" in i.text
        assert i.metrics["n_screened"] == n_exp, kind
    assert _by_kind(out, "pair_high")[0].metrics["n_screened"] == 2  # SP->SP and SP->RJ (SP->AC has n = 20)
    assert "of 2 routes" in _by_kind(out, "pair_high")[0].text


def test_insight_texts_use_the_requested_alpha(big):
    out = ins.generate_insights(big, 100, alpha=0.01)
    for i in out:
        assert "95%" not in i.text, (i.kind, i.text)
    joined = " ".join(i.text for i in out)
    assert "99% CI" in joined and "99% interval" in joined and "Unadjusted 99% intervals" in joined
    # the displayed segment intervals ARE the 99% intervals
    rj = _by_kind(out, "region_high")[0].metrics["segments"][0]
    lo, hi = an.wilson_ci(108, 600, 0.01)
    assert rj["late_ci_low"] == pytest.approx(lo) and rj["late_ci_high"] == pytest.approx(hi)
    ov = _by_kind(out, "overview")[0].metrics
    assert (ov["late_ci_low"], ov["late_ci_high"]) == pytest.approx(an.wilson_ci(208, 1520, 0.01))
    default = " ".join(i.text for i in ins.generate_insights(big, 100))
    assert "95% CI" in default and "99%" not in default


def test_regional_report_sorted_and_summary_report_keeps_rate_order(big):
    rr = ins.regional_report(big, min_orders=100)
    assert rr["late_rate"].is_monotonic_decreasing and list(rr["customer_state"]) == ["AC", "RJ", "SP"]
    assert rr["late_rate"].tolist() == pytest.approx([0.5, 0.18, 0.1])


def test_run_all_tests_chi_square_orientation(big):
    res = sa.run_all_tests(big)
    s = res["summary"].set_index("variables")
    chi = res["chi_square"]["customer_state"]
    assert s.loc["customer_state x is_late", "p_value"] == chi["p_value"]
    assert s.loc["customer_state x is_late", "effect_size"] == chi["cramers_v"]


# --------------------------------------------------------------------------- #
# Audit round 3: known-answer tests for binning, pass-through of min_orders / alpha,
# statistical alpha handling, insight gates and report content
# --------------------------------------------------------------------------- #
def _mini(values, col="x", late=None, dur=None) -> pd.DataFrame:
    """Tiny frame with just the columns binned_stat needs (TEST FIXTURE ONLY)."""
    n = len(values)
    return pd.DataFrame({
        col: np.asarray(values, dtype=float),
        "is_late": pd.array([0] * n if late is None else late, dtype="Int8"),
        "delivery_duration_days": np.asarray([10.0] * n if dur is None else dur, dtype=float),
    })


def test_binned_stat_quantile_edges_known_answer():
    t = an.binned_stat(_mini(np.arange(100)), "x", bins=4, quantile=True)
    assert t["bin_low"].tolist() == pytest.approx([0, 24.75, 49.5, 74.25])
    assert t["bin_high"].tolist() == pytest.approx([24.75, 49.5, 74.25, 99])
    assert t["n_orders"].tolist() == [25, 25, 25, 25]
    assert t["bin"].tolist() == ["[0, 24.75]", "(24.75, 49.5]", "(49.5, 74.25]", "(74.25, 99]"]


def test_binned_stat_quantile_and_equal_width_differ_on_skewed_data():
    d = _mini([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
    q = an.binned_stat(d, "x", bins=2, quantile=True)  # equal count: median 5.5 splits 5 / 5
    assert q["bin_high"].tolist() == pytest.approx([5.5, 100]) and q["n_orders"].tolist() == [5, 5]
    w = an.binned_stat(d, "x", bins=2, quantile=False)  # equal width: midpoint (1 + 100) / 2
    assert w["bin_low"].tolist() == pytest.approx([1, 50.5]) and w["bin_high"].tolist() == pytest.approx([50.5, 100])
    assert w["n_orders"].tolist() == [9, 1]
    # the weight / freight wrappers route `quantile` to the same construction
    d2 = d.assign(total_weight_g=d["x"], total_freight=d["x"])
    assert an.weight_bins(d2, quantile=True, bins=2)["n_orders"].tolist() == [5, 5]
    assert an.weight_bins(d2, quantile=False, bins=2)["n_orders"].tolist() == [10]  # fixed edges: all <= 500
    assert an.freight_bins(d2, quantile=True, bins=2)["n_orders"].tolist() == [5, 5]


def test_binned_stat_mean_value_is_a_mean_not_a_median():
    t = an.binned_stat(_mini([1, 1, 1, 10], late=[0, 0, 0, 1], dur=[2, 2, 2, 20]), "x", [0, 20])
    assert t["n_orders"].tolist() == [4]
    assert t["mean_value"].iloc[0] == pytest.approx(3.25)  # median would be 1
    assert t["mean_duration"].iloc[0] == pytest.approx(6.5) and t["median_duration"].iloc[0] == pytest.approx(2.0)


def test_binned_stat_min_orders_and_alpha_reach_reliable_and_ci_columns():
    d = _mini([1] * 6 + [9] * 4, late=[1, 1, 0, 0, 0, 0, 1, 0, 0, 0])
    t = an.binned_stat(d, "x", [0, 5, 10], min_orders=5)
    assert t["n_orders"].tolist() == [6, 4] and t["reliable"].tolist() == [True, False]
    assert an.binned_stat(d, "x", [0, 5, 10], min_orders=7)["reliable"].tolist() == [False, False]
    assert an.binned_stat(d, "x", [0, 5, 10])["reliable"].tolist() == [True, True]  # default 0
    t99 = an.binned_stat(d, "x", [0, 5, 10], alpha=0.01)
    assert (t99.loc[0, "late_ci_low"], t99.loc[0, "late_ci_high"]) == pytest.approx(an.wilson_ci(2, 6, 0.01))
    assert (t99.loc[1, "late_ci_low"], t99.loc[1, "late_ci_high"]) == pytest.approx(an.wilson_ci(1, 4, 0.01))
    assert t99.loc[0, "late_ci_low"] != pytest.approx(t.loc[0, "late_ci_low"])


def test_binned_stat_drops_values_below_the_first_fixed_edge():
    t = an.binned_stat(_mini([-3, 0, 2, 5, 7]), "x", [0, 5, 10])
    assert t["n_orders"].tolist() == [3, 1]  # -3 is outside; 0 (the first edge) is inside; 5 closes bin 1
    assert t["bin"].tolist() == ["[0, 5]", "(5, 10]"]
    assert an.binned_stat(_mini([-3, -1]), "x", [0, 5, 10]).empty


def test_weight_bins_fixed_edges_one_value_per_interval():
    vals = [250, 750, 1500, 3500, 7500, 20000, 35000]  # 20000 and 35000 straddle the 30000 edge
    t = an.weight_bins(_mini(vals, col="total_weight_g"))
    assert t["n_orders"].tolist() == [1] * 7
    assert t["bin_low"].tolist() == [0, 500, 1000, 2000, 5000, 10000, 30000]
    assert t["bin_high"].tolist()[:-1] == [500, 1000, 2000, 5000, 10000, 30000] and np.isinf(t["bin_high"].iloc[-1])
    assert t["bin"].tolist()[-1] == "> 30000" and t["mean_value"].tolist() == pytest.approx(vals)


def test_freight_bins_read_freight_not_price_and_use_the_fixed_edges():
    vals = [5, 12, 18, 25, 40, 60]
    d = _mini(vals, col="total_freight").assign(total_price=1e6, total_weight_g=1.0)
    t = an.freight_bins(d)
    assert t["n_orders"].tolist() == [1] * 6 and t["mean_value"].tolist() == pytest.approx(vals)
    assert t["bin_low"].tolist() == [0, 10, 15, 20, 30, 50]
    assert t["bin_high"].tolist()[:-1] == [10, 15, 20, 30, 50] and np.isinf(t["bin_high"].iloc[-1])
    assert t["bin"].tolist()[-1] == "> 50"


def test_bin_wrappers_pass_min_orders_and_alpha_through():
    w = _mini([250, 750, 1500], col="total_weight_g", late=[1, 0, 0])
    f = _mini([5, 12, 18], col="total_freight", late=[1, 0, 0])
    p = _mini([0.5, 1.5, 2.5], col="processing_days", late=[1, 0, 0])
    for fn, d in ((an.weight_bins, w), (an.freight_bins, f), (an.late_by_processing_bucket, p)):
        assert fn(d)["reliable"].all()  # min_orders = 0
        assert fn(d, min_orders=1)["reliable"].all()
        assert not fn(d, min_orders=2)["reliable"].any()  # every bin holds one order
        ci = fn(d, alpha=0.01).loc[0, ["late_ci_low", "late_ci_high"]].tolist()
        assert ci == pytest.approx(an.wilson_ci(1, 1, 0.01))
        assert ci != pytest.approx(an.wilson_ci(1, 1, 0.05))


def test_delay_trend_honours_min_orders_alpha_and_quarter_aliases(fx):
    assert an.delay_trend(fx, "M", min_orders=6)["reliable"].tolist() == [True, True]
    assert an.delay_trend(fx, "M", min_orders=7)["reliable"].tolist() == [False, False]
    assert an.delay_trend(fx, "M")["reliable"].tolist() == [True, True]
    t99 = an.delay_trend(fx, "M", alpha=0.01)
    assert (t99["late_ci_low"].iloc[0], t99["late_ci_high"].iloc[0]) == pytest.approx(an.wilson_ci(3, 6, 0.01))
    assert t99["late_ci_low"].iloc[0] != pytest.approx(an.delay_trend(fx, "M")["late_ci_low"].iloc[0])
    for alias in ("Q", "QE", "QS"):  # both months fall in 2017 Q1 -> a single quarterly row
        q = an.delay_trend(fx, alias)
        assert q["n_orders"].tolist() == [12] and q["n_late"].tolist() == [5], alias
        assert q["period"].tolist() == [pd.Timestamp("2017-01-01")], alias


def test_category_cost_comparison_reliable_boundary_and_median_ratio(fx):
    t6 = an.category_cost_comparison(fx, min_orders=6)
    assert t6["reliable"].all()  # n == min_orders is reliable (inclusive)
    assert not an.category_cost_comparison(fx, min_orders=7)["reliable"].any()
    books = fx[fx["product_category"] == "books"]["freight_ratio"].astype(float)
    t = an.category_cost_comparison(fx).set_index("product_category")
    assert t.loc["books", "median_freight_ratio"] == pytest.approx(books.median())
    assert books.median() != pytest.approx(books.mean())  # skewed: the two statistics really differ
    assert t.loc["books", "mean_freight_ratio"] == pytest.approx(books.mean())


def test_scatter_sample_drops_infinite_values(fx):
    d = fx.copy()
    d["total_weight_g"] = d["total_weight_g"].astype(float)
    d.loc[0, "total_weight_g"] = np.inf
    d.loc[1, "delivery_duration_days"] = -np.inf
    s = an.scatter_sample(d, "total_weight_g", "delivery_duration_days", n=100)
    assert len(s) == 10 and np.isfinite(s[["total_weight_g", "delivery_duration_days"]].to_numpy()).all()


# ---- statistical_analysis: alpha, pooling, wording --------------------------------------------
_XS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan, 3.0])
_YS = np.array([2, 1, 4, 3, 7, 8, 6, 10, 9, 12, 5.0, np.nan])


def test_spearman_ci_endpoints_follow_alpha():
    df = pd.DataFrame({"x": _XS, "y": _YS})
    ok = ~(np.isnan(_XS) | np.isnan(_YS))
    rho = stats.spearmanr(_XS[ok], _YS[ok]).statistic
    z, se = math.atanh(rho), math.sqrt((1 + rho ** 2 / 2) / (10 - 3))
    zc99 = 2.5758293035489004  # two-sided 99% normal quantile
    r99 = sa.spearman_test(df, "x", "y", alpha=0.01)
    assert r99["ci_low"] == pytest.approx(math.tanh(z - zc99 * se), abs=1e-9)
    assert r99["ci_high"] == pytest.approx(math.tanh(z + zc99 * se), abs=1e-9)
    assert r99["ci_low"] < sa.spearman_test(df, "x", "y")["ci_low"]  # wider than the 95% interval


def test_proportion_diff_ci_endpoints_follow_alpha_for_both_groups():
    r = sa.proportion_diff_ci(56, 70, 48, 80, alpha=0.01)
    w1 = stats.binomtest(56, 70).proportion_ci(confidence_level=0.99, method="wilson")
    w2 = stats.binomtest(48, 80).proportion_ci(confidence_level=0.99, method="wilson")
    d = 56 / 70 - 48 / 80
    assert r["ci_low"] == pytest.approx(d - math.sqrt((56 / 70 - w1.low) ** 2 + (w2.high - 48 / 80) ** 2), abs=1e-9)
    assert r["ci_high"] == pytest.approx(d + math.sqrt((w1.high - 56 / 70) ** 2 + (48 / 80 - w2.low) ** 2), abs=1e-9)
    assert r["alpha"] == 0.01


def test_run_all_tests_passes_alpha_to_every_test(big):
    d = big.sample(n=300, random_state=1)
    res = sa.run_all_tests(d, alpha=0.5)
    every = [*res["chi_square"].values(), *res["spearman"].values(), *res["group_comparison"].values()]
    assert res["alpha"] == 0.5 and all(r["alpha"] == 0.5 for r in every)
    assert all(r["significant"] == (r["p_value"] < 0.5) for r in every if np.isfinite(r["p_value"]))
    # the level really changes the verdicts: p ~ 0.25 (Spearman) passes at 0.5 only; p ~ 0.02 (chi-square)
    # passes at the default 0.05 but not at 0.01
    default, strict = sa.run_all_tests(d), sa.run_all_tests(d, alpha=0.01)
    wd = "total_weight_g~delivery_duration_days"
    assert not default["spearman"][wd]["significant"] and res["spearman"][wd]["significant"]
    assert default["chi_square"]["customer_state"]["significant"]
    assert not strict["chi_square"]["customer_state"]["significant"] and strict["chi_square"]["customer_state"]["alpha"] == 0.01


def test_run_all_tests_drops_states_under_30_orders_from_the_kruskal_group(big):
    grp = sa.run_all_tests(big)["group_comparison"]
    g = grp["delivery_duration_days~customer_state"]
    assert g["n_groups"] == 2 and g["n"] == 1500  # AC (n = 20) is below the 30-order cut-off
    assert set(g["group_summary"]["group"]) == {"RJ", "SP"} and g["test"] == "mann_whitney"
    s = grp["delivery_duration_days~same_state"]
    assert s["n"] == 1520 and s["n_groups"] == 2  # min_group_size stays 2 for the same_state split


def test_chi_square_pool_grows_until_it_is_large_enough():
    # C and D are each rare (10 orders < 33.4 needed) and together (20) still too small, so the
    # pool must keep absorbing the next-smallest level until expected counts are valid.
    res = sa.chi_square_test(_df_from_counts({"A": (800, 200), "B": (900, 100), "C": (9, 1), "D": (9, 1)}), "g")
    assert {"C", "D"} < set(res["merged_rare_levels"]) and len(res["merged_rare_levels"]) >= 3
    assert res["min_expected_after"] >= 5
    # when the first two merged levels are already big enough, nothing more is pulled in
    ok = sa.chi_square_test(_df_from_counts({"A": (800, 200), "B": (900, 100), "C": (20, 4), "D": (20, 4)}), "g")
    assert ok["merged_rare_levels"] == ["C", "D"] and ok["min_expected_after"] >= 5


def test_spearman_ignores_infinite_values():
    df = pd.DataFrame({"x": [1.0, 2, 3, 4, 5, 6, np.inf, 8], "y": [2.0, 1, 4, 3, 7, 8, 6, -np.inf]})
    r = sa.spearman_test(df, "x", "y")
    ref = stats.spearmanr(df["x"][:6], df["y"][:6])
    assert r["n"] == 6 and r["rho"] == pytest.approx(ref.statistic) and r["p_value"] == pytest.approx(ref.pvalue)


def test_spearman_ci_is_not_reported_for_a_binary_variable(fx):
    r = sa.spearman_test(fx, "total_freight", "is_late")
    assert np.isfinite(r["rho"]) and math.isnan(r["ci_low"]) and math.isnan(r["ci_high"])
    assert "binary" in r["note"] and r["n"] == 12
    txt = sa.plain_language(r)
    assert "no confidence interval shown" in txt and "nan" not in txt.lower()
    assert np.isfinite(sa.spearman_test(fx, "total_weight_g", "total_freight")["ci_low"])  # continuous pair: kept
    assert math.isnan(sa.spearman_test(fx, "is_late", "total_freight")["ci_low"])  # either order
    res = sa.run_all_tests(make_blocks(BLOCKS))
    assert math.isnan(res["spearman"]["freight_ratio~is_late"]["ci_low"])
    assert np.isfinite(res["spearman"]["total_weight_g~delivery_duration_days"]["ci_low"])


def test_fmt_p_small_values_and_plain_language_pooled_and_top_level_sentences():
    assert sa._fmt_p(2.5e-7) == "p < 0.001 (p = 2.5e-07)" and sa._fmt_p(0.0123) == "p = 0.012"
    assert sa._fmt_p(0.0) == "p < 0.001 (below floating-point precision)" and sa._fmt_p(float("nan")) == "p = n/a"
    assert ins._p(0.0) == "p < 0.001" and ins._p(2.5e-7) == "p = 2.5e-07" and ins._p(0.0123) == "p = 0.012"
    res = sa.chi_square_test(_df_from_counts({"A": (800, 200), "B": (900, 100), "C": (24, 6), "D": (17, 3)}), "g")
    txt = sa.plain_language(res)
    assert "p < 0.001 (p = " in txt
    assert "2 rare level(s) were pooled so that expected counts stay valid" in txt
    top = res["level_summary"].iloc[0]
    assert "The level that departs most from independence is" in txt and f"'{top['level']}'" in txt
    assert f"(late rate {100 * top['late_rate']:.1f}%, n = {int(top['n']):,})" in txt
    plain = sa.plain_language(sa.chi_square_test(_df_from_counts({"A": (70, 30), "B": (50, 50)}), "g"))
    assert "were pooled" not in plain


def test_plain_language_closing_uses_the_results_alpha_and_notes_adjustment():
    big_counts = {"A": (9000, 1000), "B": (8900, 1100)}
    r01 = sa.chi_square_test(_df_from_counts(big_counts), "g", alpha=0.01)
    t01 = sa.plain_language(r01)
    assert "(p < 0.01)" in t01 and "(p < 0.05)" not in t01
    assert "(p < 0.05)" in sa.plain_language(sa.chi_square_test(_df_from_counts(big_counts), "g"))
    assert "for this test alone (unadjusted for multiple testing)" in t01  # a stand-alone result
    d = make_blocks(BLOCKS).sample(n=300, random_state=1)
    res = sa.run_all_tests(d)
    # estimated_delivery_days is absent from this fixture, so only 11 of the 12 tests produce a
    # p-value; Holm (and the quoted count) runs over those 11
    assert res["summary"]["p_value"].isna().sum() == 1
    cs = res["chi_square"]["customer_state"]  # raw p ~ 0.02, Holm p ~ 0.14 over the 11-test battery
    assert cs["significant"] and cs["n_tests"] == 11 and cs["p_holm"] > 0.05
    txt = sa.plain_language(cs)
    assert "a statistically detectable association" in txt  # the per-test wording is unchanged ...
    assert "after Holm adjustment across the 11 tests run together" in txt  # ... but now qualified
    assert "not statistically detectable" in txt.split("Holm adjustment")[1]
    strong = sa.plain_language(res["chi_square"]["product_category"])
    assert "still statistically detectable" in strong
    for kind, r in (("spearman", res["spearman"]["total_weight_g~delivery_duration_days"]),
                    ("kruskal", res["group_comparison"]["delivery_duration_days~same_state"])):
        assert "Holm adjustment across the 11 tests" in sa.plain_language(r), kind
    assert [r["p_holm"] for r in res["chi_square"].values()] == pytest.approx(
        res["summary"]["p_holm"].iloc[:4].tolist())


def test_plain_language_proportion_closing_follows_alpha():
    r = sa.proportion_diff_ci(5000, 10000, 4800, 10000, alpha=0.01)
    assert "(p < 0.01)" in sa.plain_language(r)


# ---- insights: weight / freight wording ------------------------------------------------------
def _wf_text(w, f, seed=0):
    d = make_blocks(BLOCKS).copy()
    d["total_weight_g"], d["total_freight"] = np.asarray(w(d), float), np.asarray(f(d), float)
    return _by_kind(ins.generate_insights(d, 100), "weight_freight")[0]


def test_weight_freight_closing_clause_is_built_from_the_computed_associations():
    rng = np.random.default_rng(5)
    dur = lambda d: d["delivery_duration_days"]
    pos = _wf_text(lambda d: dur(d) * 100 + rng.normal(0, 500, len(d)), lambda d: dur(d) * 2 + rng.normal(0, 1, len(d)))
    neg = _wf_text(lambda d: -dur(d) * 100 + rng.normal(0, 500, len(d)), lambda d: -dur(d) + rng.normal(0, 1, len(d)))
    none = _wf_text(lambda d: rng.normal(0, 1, len(d)), lambda d: rng.normal(0, 1, len(d)))
    mixed = _wf_text(lambda d: rng.normal(0, 1, len(d)), lambda d: dur(d) * 2 + rng.normal(0, 1, len(d)))
    for item in (pos, neg, none, mixed):
        assert "associated with these delivery times" not in item.text  # the old fixed claim is gone
        assert "does not show that weight or freight cost causes" in item.text
    for item, sign in ((pos, "positive"), (neg, "negative")):
        w, f = item.metrics["weight_vs_duration"], item.metrics["freight_vs_duration"]
        assert np.sign(w["rho"]) == (1 if sign == "positive" else -1)
        assert f"Order weight shows a {w['strength_label']} {sign} rank association" in item.text
        assert f"freight cost shows a {f['strength_label']} {sign} rank association with delivery time" in item.text
    assert none.metrics["weight_vs_duration"]["strength_label"] == "negligible"
    assert none.metrics["freight_vs_duration"]["strength_label"] == "negligible"
    assert "Neither weight nor freight cost shows more than a negligible rank association" in none.text
    assert "positive rank association" not in none.text and "negative rank association" not in none.text
    f = mixed.metrics["freight_vs_duration"]["strength_label"]
    assert mixed.metrics["weight_vs_duration"]["strength_label"] == "negligible" and f != "negligible"
    assert "Order weight shows a negligible rank association and freight cost shows a" in mixed.text


# ---- insights: screens, gates and ordering -----------------------------------------------------
def test_month_screen_excludes_unreliable_months_from_tests_but_keeps_them_in_the_rest_pool():
    rows = [("SP", "SP", f"2017-{m:02d}", "toys", 300, 30) for m in range(1, 5)]
    rows.append(("SP", "SP", "2017-05", "toys", 50, 40))  # n = 50 < min_orders: heavily late, but unreliable
    d = _blocks(rows)
    scr = ins._month_screen(an.monthly_performance(d, 100), 0.05)
    assert list(scr["month"]) == ["2017-01", "2017-02", "2017-03", "2017-04"]  # the n = 50 month is not tested
    jan = scr.set_index("month").loc["2017-01"]
    assert jan["rest_n"] == 1250 - 300 == 950  # ... but its orders ARE part of every month's comparison pool
    assert jan["rest_late_rate"] == pytest.approx((160 - 30) / 950)
    out = ins.generate_insights(d, 100)
    mh = _by_kind(out, "month_high")[0]
    assert mh.metrics["n_tested"] == 4 and "None of the 4 months" in mh.text
    assert _by_kind(out, "month_extremes")[0].metrics["worst_month"] != "2017-05"


def test_generate_insights_at_exactly_min_orders_still_compares(big):
    out = ins.generate_insights(big, min_orders=len(big))  # n == min_orders is enough (strict <)
    kinds = {i.kind for i in out}
    assert "weight_freight" in kinds and "processing_time" in kinds
    sections = _by_kind(out, "insufficient_data")[0].metrics["sections"]
    assert sections != ["all comparisons"] and "regional comparison" in sections
    below = ins.generate_insights(big, min_orders=len(big) + 1)
    assert [i.kind for i in below] == ["overview", "insufficient_data"]
    assert below[1].metrics["sections"] == ["all comparisons"]


def test_category_insight_flagged_vs_displayed_and_screen_counts():
    # c1..c3 have the highest late rates (17%, n = 100: interval contains the overall rate) and are displayed;
    # c4 (13.5%, n = 3000) is reliably above the overall rate but not among the top 3 by rate.
    d = _blocks([("SP", "SP", "2017-01", "c1", 100, 17), ("SP", "SP", "2017-01", "c2", 100, 17),
                 ("SP", "SP", "2017-02", "c3", 100, 17), ("SP", "SP", "2017-02", "c4", 3000, 405),
                 ("SP", "SP", "2017-03", "base", 3000, 285), ("SP", "SP", "2017-03", "tiny", 20, 2),
                 ("SP", "SP", "2017-03", "unknown", 300, 30)])
    out = ins.generate_insights(d, 100)
    cd = _by_kind(out, "category_delay")[0]
    assert [s["product_category"] for s in cd.metrics["segments"]] == ["c1", "c2", "c3"]
    assert cd.metrics["n_reliably_above"] == 1 and cd.metrics["flagged"] == []  # c4 flagged, not displayed
    assert cd.severity == "info" and "[reliably above overall]" not in cd.text
    assert cd.metrics["n_screened"] == 5  # c1-c4 + base; 'unknown' and the n = 20 'tiny' are not screened
    assert "across 5 segments screened" in cd.text
    assert not [r for r in ins.recommendations(out) if "reliably higher late rates" in r]
    # two displayed + flagged categories keep the late-rate order (not set order)
    d2 = _blocks([("SP", "SP", "2017-01", "zz", 1000, 250), ("SP", "SP", "2017-01", "aa", 1000, 200),
                  ("SP", "SP", "2017-02", "mm", 1000, 150), ("SP", "SP", "2017-02", "base", 3000, 150)])
    cd2 = _by_kind(ins.generate_insights(d2, 100), "category_delay")[0]
    assert cd2.metrics["flagged"] == ["zz", "aa", "mm"] and cd2.severity == "watch"


def test_category_freight_skips_unknown_and_unreliable_categories():
    d = _blocks(CAT_BLOCKS + [("SP", "SP", "2017-03", "tiny", 20, 2)]).copy()
    d.loc[d["product_category"] == "unknown", "total_freight"] += 1000.0  # highest freight of all
    d.loc[d["product_category"] == "tiny", "total_freight"] += 500.0  # second highest, but n = 20 < 100
    cf = _by_kind(ins.generate_insights(d, 100), "category_freight")[0]
    names = [s["product_category"] for s in cf.metrics["segments"]]
    assert "unknown" not in names and "tiny" not in names and len(names) == 3
    assert "unknown" not in cf.text and "tiny" not in cf.text


def test_processing_time_uses_strict_comparison_with_the_upper_quartile():
    d = _blocks([("SP", "SP", "2017-01", "toys", 101, 10)])
    d["processing_days"] = np.arange(101.0)  # q75 = 75 exactly (a tie with an observed value)
    pt = _by_kind(ins.generate_insights(d, 100), "processing_time")[0]
    assert pt.metrics["q75_processing_days"] == pytest.approx(75.0)
    assert pt.metrics["slow_n"] == 25 and pt.metrics["rest_n"] == 76  # strict >: 76..100 only; median would give 50
    assert pt.n == 101 and "(slowest quarter)" in pt.text


def test_insight_min_orders_gates_use_the_available_pairs():
    d = _blocks([("SP", "SP", "2017-01", "toys", 101, 10)])

    def kinds_and_sections(frame):
        out = ins.generate_insights(frame, 100)
        sec = _by_kind(out, "insufficient_data")
        return {i.kind for i in out}, (sec[0].metrics["sections"] if sec else [])

    # processing_days: 100 usable values == min_orders passes, 99 does not
    p100, p99 = d.copy(), d.copy()
    p100.loc[p100.index[:1], "processing_days"] = np.nan
    p99.loc[p99.index[:2], "processing_days"] = np.nan
    assert "processing_time" in kinds_and_sections(p100)[0]
    k99, s99 = kinds_and_sections(p99)
    assert "processing_time" not in k99 and "processing-time comparison" in s99
    # weight and freight each need min_orders usable pairs with delivery time
    for col in ("total_weight_g", "total_freight"):
        a, b = d.copy(), d.copy()
        a.loc[a.index[:1], col] = np.nan
        b.loc[b.index[:2], col] = np.nan
        assert "weight_freight" in kinds_and_sections(a)[0], col
        kb, sb = kinds_and_sections(b)
        assert "weight_freight" not in kb and "weight / freight association" in sb, col


def _ins(kind, severity="info", n=100, **metrics):
    return ins.Insight(kind, kind, "text", metrics, n, severity)


def test_recommendation_gates_follow_the_insights():
    region = dict(n_flagged=2, n_screened=5, overall_late_rate=0.1,
                  segments=[{"customer_state": "RJ", "late_rate": 0.2, "n_orders": 300}])
    assert any("Investigate delivery performance" in r and "unadjusted screen of 5 regions" in r
               for r in ins.recommendations([_ins("region_high", **region)]))
    assert ins.recommendations([_ins("region_high", **{**region, "n_flagged": 0})]) == []
    month = dict(n_flagged=1, months=[{"month": "2017-03"}])
    assert any("Review capacity" in r and "2017-03" in r for r in ins.recommendations([_ins("month_high", **month)]))
    assert ins.recommendations([_ins("month_high", n_flagged=0, months=[])]) == []
    pair = ins.recommendations([_ins("pair_high", n_screened=70, segments=[{"state_pair": "SP→RJ"}])])
    assert "SP→RJ" in pair[0] and "unadjusted screen of 70 routes" in pair[0] and "treat them as leads" in pair[0]
    assert "reliably above average" not in pair[0]
    cat = lambda sev, flagged: ins.recommendations([_ins("category_delay", sev, flagged=flagged, n_screened=9)])
    assert "unadjusted screen of 9 categories" in cat("watch", ["a"])[0]
    assert cat("info", ["a"]) == [] and cat("watch", []) == []
    assert ins.recommendations([_ins("category_duration", segments=[{"product_category": "a"}])])
    same = lambda sev: ins.recommendations([_ins("same_state", sev, cross_late_rate=0.2, same_late_rate=0.1)])
    assert "cross-state" in same("watch")[0] and same("info") == []
    proc = lambda sev: ins.recommendations([_ins("processing_time", sev)])
    assert "retrospective" in proc("watch")[0] and proc("info") == []
    miss = ins.recommendations([_ins("insufficient_data", sections=["regional comparison"])])
    assert "regional comparison" in miss[0]
    for recs in (pair, miss):  # the observational-data caveat closes any non-empty list
        assert recs[-1].startswith("Treat all findings as associations")
    assert ins.recommendations([_ins("overview"), _ins("region_low")]) == []


# ---- summary report ---------------------------------------------------------------------------
def test_summary_report_exact_key_figures_dates_caveats_and_n_lines(big):
    out = ins.generate_insights(big, min_orders=100)
    rep = ins.summary_report(big, "x", out)
    ref = stats.binomtest(208, 1520).proportion_ci(confidence_level=0.95, method="wilson")
    assert "| Delivered orders (n) | 1,520 |" in rep
    assert (f"| Late rate | 13.7% (95% CI {100 * ref.low:.1f}% to {100 * ref.high:.1f}%; 208 late orders) |") in rep
    assert "| On-time rate | 86.3% |" in rep
    assert "**Purchase dates covered:** 2017-01-15 to 2017-03-15" in rep
    assert "Many segments are screened at once, so a flagged segment is a lead to investigate, not a confirmed effect." in rep
    assert "Rates carry 95% Wilson confidence intervals" in rep
    for i in out:
        if i.kind != "overview":
            assert f"### {i.title} [" in rep and f"*Based on n = {i.n:,} orders.*" in rep
    region_test = _by_kind(out, "region_test")[0]
    assert f"{region_test.text}\n\n*Based on n = {region_test.n:,} orders.*" in rep
    assert rep.count("*Based on n = ") == len([i for i in out if i.kind != "overview"])


def test_summary_report_key_figures_ci_uses_the_exact_n():
    d = _blocks([("SP", "SP", "2017-01", "toys", 40, 4)])  # small n: one extra order would change the rounded CI
    rep = ins.summary_report(d, "x", ins.generate_insights(d, 10))
    ref = stats.binomtest(4, 40).proportion_ci(confidence_level=0.95, method="wilson")
    assert f"| Late rate | 10.0% (95% CI {100 * ref.low:.1f}% to {100 * ref.high:.1f}%; 4 late orders) |" in rep
    assert "| Delivered orders (n) | 40 |" in rep and "4.0% to 23.1%" in rep


def test_summary_report_confidence_level_follows_the_insights_alpha(big):
    out99 = ins.generate_insights(big, 100, alpha=0.01)
    rep = ins.summary_report(big, "x", out99)  # level is taken from the insights' alpha
    lo, hi = an.wilson_ci(208, 1520, 0.01)
    assert f"99% CI {100 * lo:.1f}% to {100 * hi:.1f}%" in rep and "Rates carry 99% Wilson confidence intervals" in rep
    assert "95%" not in rep
    assert _by_kind(out99, "overview")[0].metrics["alpha"] == 0.01
    explicit = ins.summary_report(big, "x", ins.generate_insights(big, 100), alpha=0.1)
    assert "90% CI" in explicit and "Rates carry 90% Wilson" in explicit
    assert "Rates carry 95% Wilson" in ins.summary_report(big, "x", [])  # no insights -> default level


def test_regional_report_flags_use_alpha_and_keep_reliable_separate_from_the_flag():
    d = _blocks([("SP", "SP", "2017-01", "toys", 1000, 100), ("RJ", "SP", "2017-01", "toys", 150, 22)])
    strict = ins.regional_report(d, 100).set_index("customer_state")
    assert strict.loc["RJ", "reliable"] and strict.loc["RJ", "flag"] == "Interval includes overall"
    lax = ins.regional_report(d, 100, alpha=0.5).set_index("customer_state")
    assert lax.loc["RJ", "flag"] == "Above overall (interval excludes it)"
    assert lax.loc["RJ", "late_ci_low"] > strict.loc["RJ", "late_ci_low"]
    assert not any("Reliably" in f for f in lax["flag"])  # "reliable" is the n >= min_orders column only


# --------------------------------------------------------------------------- #
# Audit round 2: regression tests for the confirmed findings
# --------------------------------------------------------------------------- #
def test_holm_test_count_ignores_tests_that_could_not_run(big):
    res = sa.run_all_tests(big[big["customer_state"] == "SP"])  # one state: two tests are undefined
    summ = res["summary"]
    n_run = int(summ["p_value"].notna().sum())
    assert n_run == len(summ) - summ["p_value"].isna().sum() < len(summ)
    assert summ.loc[summ["variables"] == "customer_state x is_late", "p_value"].isna().all()
    every = [*res["chi_square"].values(), *res["spearman"].values(), *res["group_comparison"].values()]
    assert {r["n_tests"] for r in every} == {n_run}
    # the quoted count is the one Holm really used: the multiplier equals the rank among n_run values
    assert summ["p_holm"].dropna().tolist() == pytest.approx(
        sa.holm_adjust(summ["p_value"].dropna().to_numpy()).tolist())
    txt = sa.plain_language(res["chi_square"]["product_category"])
    assert f"after Holm adjustment across the {n_run} tests run together" in txt
    full = sa.run_all_tests(big)
    assert {r["n_tests"] for r in full["chi_square"].values()} == {int(full["summary"]["p_value"].notna().sum())}


def test_time_series_are_chronological_even_when_the_input_is_not(fx):
    rev = fx.sort_values("purchase_date", ascending=False)
    assert rev["purchase_year_month"].iloc[0] == "2017-02"  # the input really is out of order
    m = an.monthly_performance(rev)
    pd.testing.assert_frame_equal(m, an.monthly_performance(fx))
    assert m["purchase_year_month"].is_monotonic_increasing
    for freq in ("M", "W", "D"):
        t = an.delay_trend(rev, freq)
        pd.testing.assert_frame_equal(t, an.delay_trend(fx, freq))
        assert t["period"].is_monotonic_increasing and t["period"].is_unique


def test_chi_level_summary_is_sorted_by_absolute_residual():
    # A is far BELOW the overall late rate (the largest |residual|, negative); B and C are mildly above
    res = sa.chi_square_test(_df_from_counts({"A": (980, 20), "B": (900, 100), "C": (890, 110)}), "g")
    ls = res["level_summary"]
    assert ls["adj_residual"].abs().is_monotonic_decreasing
    assert ls["level"].iloc[0] == "A" and ls["adj_residual"].iloc[0] < 0
    assert ls["adj_residual"].iloc[0] < -ls["adj_residual"].iloc[1:].max()  # |A| really is the largest
    assert "'A'" in sa.plain_language(res).split("departs most from independence is")[1][:6]


def test_tables_report_mean_weight_median_duration_and_same_state_means(fx):
    st = an.state_performance(fx).set_index("customer_state")
    assert st.loc["SP", "mean_weight_g"] == pytest.approx(4550 / 6)  # 500+800+1500+300+250+1200
    assert st.loc["RJ", "mean_weight_g"] == pytest.approx(6350 / 4)
    assert st.loc["SP", "median_duration"] == pytest.approx(6.5)  # 4,5,6,7,9,12 (mean is 43/6)
    assert st.loc["RJ", "median_duration"] == pytest.approx(14.0)  # 6,8,20,25 (mean is 14.75)
    cat = an.category_performance(fx).set_index("product_category")
    assert cat.loc["toys", "mean_weight_g"] == pytest.approx(2050.0)
    assert cat.loc["books", "mean_weight_g"] == pytest.approx(6100 / 6)
    pair = an.state_pair_performance(fx).set_index("state_pair")
    assert pair.loc["MG→SP", "mean_weight_g"] == pytest.approx(1350.0)
    assert pair.loc["MG→SP", "mean_freight"] == pytest.approx(17.5)
    ss = an.same_state_comparison(fx).set_index("group")
    assert ss.loc["cross_state", "mean_duration"] == pytest.approx(118 / 6)  # 12,9,20,25,30,22
    assert ss.loc["cross_state", "median_duration"] == pytest.approx(21.0)  # mean differs from median
    assert ss.loc["same_state", "mean_duration"] == pytest.approx(6.0)
    assert ss.loc["same_state", "median_duration"] == pytest.approx(6.0)


def test_state_pair_performance_column_layout(fx):
    for t in (an.state_pair_performance(fx), an.state_pair_performance(fx.iloc[0:0])):
        assert list(t.columns[:5]) == ["state_pair", "seller_state", "customer_state", "same_state", "n_orders"]


def test_cramers_v_corrected_is_the_bergsma_estimate():
    for counts in ({"A": (70, 30), "B": (50, 50)}, {"A": (70, 30), "B": (50, 50), "C": (20, 30)}):
        res = sa.chi_square_test(_df_from_counts(counts), "g")
        obs = np.array(list(counts.values()))
        n, (r, c) = int(obs.sum()), obs.shape
        chi2 = stats.chi2_contingency(obs, correction=False)[0]
        phi2c = max(0.0, chi2 / n - (r - 1) * (c - 1) / (n - 1))
        rc, cc = r - (r - 1) ** 2 / (n - 1), c - (c - 1) ** 2 / (n - 1)
        assert res["cramers_v_corrected"] == pytest.approx(math.sqrt(phi2c / min(rc - 1, cc - 1)), rel=1e-12)
        assert res["cramers_v_corrected"] < res["cramers_v"]


def test_weight_freight_insight_needs_a_significant_correlation_for_watch():
    rng = np.random.default_rng(1)
    d = make_blocks(BLOCKS).iloc[::100].head(15).reset_index(drop=True)  # 15 orders
    d["total_weight_g"] = rng.normal(0, 1, len(d))
    d["total_freight"] = rng.normal(0, 1, len(d))
    sw = sa.spearman_test(d, "total_weight_g", "delivery_duration_days")
    sf = sa.spearman_test(d, "total_freight", "delivery_duration_days")
    assert max(abs(sw["rho"]), abs(sf["rho"])) >= 0.3 and not sw["significant"] and not sf["significant"]
    assert _by_kind(ins.generate_insights(d, 10), "weight_freight")[0].severity == "info"


def test_region_high_alert_boundary_is_inclusive():
    # RJ 40/200 = 20% vs overall 100/1000 = 10%: the ratio is exactly ALERT_RATIO (2.0)
    d = _blocks([("SP", "SP", "2017-01", "toys", 800, 60), ("RJ", "SP", "2017-01", "toys", 200, 40)])
    out = ins.generate_insights(d, 100)
    rh = _by_kind(out, "region_high")[0]
    assert rh.metrics["segments"][0]["rate_ratio"] == ins.ALERT_RATIO and rh.severity == "alert"
    assert _by_kind(out, "pair_high")[0].severity == "alert"


def test_scatter_sample_seed_and_size_boundaries(fx, big):
    x, y = "total_weight_g", "delivery_duration_days"
    got = an.scatter_sample(big, x, y, n=200)
    ref = big[[x, y, "is_late"]].sample(n=200, random_state=config.RANDOM_STATE).sort_index().reset_index(drop=True)
    pd.testing.assert_frame_equal(got, ref)
    assert len(an.scatter_sample(fx, x, y, n=12)) == 12  # exactly n rows: all kept
    assert len(an.scatter_sample(fx, x, y, n=11)) == 11  # n + 1 rows: still trimmed to n
    assert len(an.scatter_sample(fx, x, y, n=13)) == 12


def test_processing_bucket_default_edges(fx):
    assert tuple(an.PROCESSING_BIN_EDGES_DAYS) == (0, 1, 2, 3, 5, 7, 14, np.inf)
    b = an.late_by_processing_bucket(fx)
    assert b["bin"].tolist() == ["[0, 1]", "(1, 2]", "(2, 3]", "(3, 5]", "(5, 7]"]
    assert b["n_orders"].tolist() == [4, 3, 1, 2, 1]  # processing days 1,1,1,1 | 2,2,2 | 3 | 4,5 | 6


def test_filters_normalise_date_bounds_with_a_time_of_day(fx):
    out = an.apply_filters(fx, (pd.Timestamp("2017-01-20 15:30"), pd.Timestamp("2017-02-01 09:00")))
    assert set(out["order_id"]) == {"o_t02", "o_t12", "o_t03", "o_t04"}  # whole days at both ends
    assert set(an.apply_filters(fx, ("2017-02-28 12:00:00", None))["order_id"]) == {"o_t11"}
    assert set(an.apply_filters(fx, (None, "2017-01-05 23:59:59"))["order_id"]) == {"o_t01"}


def test_insights_say_when_no_route_or_region_reaches_min_orders():
    rows = [(st, "SP", "2017-01", "toys", 30, 3) for st in ("AA", "BB", "CC", "DD", "EE")]
    out = ins.generate_insights(_blocks(rows), 50)  # 150 orders overall, 30 per route / region
    sections = _by_kind(out, "insufficient_data")[0].metrics["sections"]
    assert "seller-to-customer route comparison" in sections and "regional comparison" in sections
    assert not {"pair_high", "pair_low", "region_high"} & {i.kind for i in out}


def test_kruskal_all_values_identical_is_undefined_not_an_error():
    same = pd.DataFrame({"v": [7.0] * 9, "g": list("aaabbbccc")})
    r = sa.kruskal_or_mannwhitney(same, "v", "g")
    assert math.isnan(r["p_value"]) and r["significant"] is False and r["note"] == "all values identical"
    assert "could not be run" in sa.plain_language(r)


def test_wilson_arrays_are_nan_where_n_is_not_positive():
    lo, hi = an.wilson_ci_arrays(np.array([1, 0, 3]), np.array([0, -2, 10]))
    assert np.isnan(lo[:2]).all() and np.isnan(hi[:2]).all() and np.isfinite([lo[2], hi[2]]).all()


def test_weight_freight_text_omits_an_undefined_confidence_interval():
    d = make_blocks(BLOCKS).head(200).copy()
    d["total_weight_g"] = np.where(np.arange(len(d)) % 2 == 0, 400.0, 900.0)  # two values only
    assert math.isnan(sa.spearman_test(d, "total_weight_g", "delivery_duration_days")["ci_low"])
    wf = _by_kind(ins.generate_insights(d, 100), "weight_freight")[0]
    assert "nan" not in wf.text.lower().replace("finance", "")
    assert wf.text.count("approx. 95% CI") == 1  # only the continuous freight variable has one
    assert "order weight vs delivery time: rho = " in wf.text


def test_report_does_not_cut_a_long_rationale_mid_sentence(big):
    sentences = [f"Sentence number {i} explains the selection step in some detail." for i in range(60)]
    rationale = " ".join(sentences)  # ~3,500 characters
    out = ins.generate_insights(big.iloc[:1], min_orders=100)
    short = " ".join(sentences[:20])
    assert short in ins.summary_report(big, "x", out, {"selected_model": {"name": "m", "rationale": short}})
    rep = ins.summary_report(big, "x", out, {"selected_model": {"name": "m", "rationale": rationale}})
    line = next(ln for ln in rep.splitlines() if ln.startswith("- **Selection rationale:**"))
    body = line.split("**Selection rationale:** ")[1]
    assert body.endswith("detail.") and not body.endswith("...") and rationale.startswith(body)
    assert len(body) < len(rationale)


@pytest.mark.skipif(not config.MODEL_METRICS_PATH.exists(), reason="models/model_metrics.json missing")
def test_report_quotes_the_real_model_rationale_in_full(big):
    mm = json.loads(config.MODEL_METRICS_PATH.read_text())
    out = ins.generate_insights(big.iloc[:1], min_orders=100)
    rep = ins.summary_report(big, "x", out, mm)
    rationale = mm["selected_model"]["rationale"]
    assert rationale.strip() in rep  # nothing dropped, in particular the test-window sentence


def test_underflowing_p_values_are_floored_not_reported_as_zero():
    huge = _df_from_counts({"A": (50000, 0), "B": (0, 50000)})
    chi = sa.chi_square_test(huge, "g")
    assert chi["p_value"] == sa.P_FLOOR > 0 and chi["significant"] is True
    mono = pd.DataFrame({"x": np.arange(5000.0), "y": np.arange(5000.0)})
    assert sa.spearman_test(mono, "x", "y")["p_value"] == sa.P_FLOOR
    sep = pd.DataFrame({"v": np.r_[np.zeros(5000), np.ones(5000)], "g": np.repeat(["a", "b"], 5000)})
    assert sa.kruskal_or_mannwhitney(sep, "v", "g")["p_value"] == sa.P_FLOOR
    assert sa.two_proportion_ztest(0, 10_000, 10_000, 10_000)["p_value"] == sa.P_FLOOR
    assert "below floating-point precision" in sa.plain_language(chi)
    assert sa._fmt_p(12 * sa.P_FLOOR) == sa._fmt_p(0.0)  # a Holm multiple of the floor reads the same
    assert not (sa.run_all_tests(make_blocks(BLOCKS))["summary"]["p_value"] == 0).any()
    assert math.isnan(sa.spearman_test(mono.iloc[:3], "x", "y")["p_value"])  # NaN is not floored


def test_purchase_month_chi_square_carries_a_pooling_caveat(big):
    res = sa.run_all_tests(big)
    pm = res["chi_square"]["purchase_month"]
    assert pm["caveat"] == sa.POOLED_MONTH_CAVEAT
    assert "does not isolate seasonality" in sa.plain_language(pm)
    for col in ("customer_state", "product_category", "same_state"):
        assert "caveat" not in res["chi_square"][col]
        assert "isolate seasonality" not in sa.plain_language(res["chi_square"][col])
    assert math.isfinite(pm["chi2"]) and pm["chi2"] > 0  # the test itself is unchanged


# --------------------------------------------------------------------------- #
# Real-data integration tests (skipped when the parquet is absent)
# --------------------------------------------------------------------------- #
needs_data = pytest.mark.skipif(
    not config.ORDERS_DELIVERED_PATH.exists(),
    reason="processed parquet missing - run `.venv/bin/python -m src.preprocessing`")


@pytest.fixture(scope="module")
def real() -> pd.DataFrame:
    from src.preprocessing import load_processed
    return load_processed()


@pytest.fixture(scope="module")
def quality() -> dict:
    return json.loads(config.QUALITY_REPORT_PATH.read_text())


@needs_data
def test_real_kpis_match_quality_report(real, quality):
    k = an.kpi_summary(real)
    assert k["n_orders"] == quality["n_eligible"] == len(real)
    assert k["n_late"] == quality["n_late_eligible"]
    assert k["late_rate"] == pytest.approx(quality["late_rate_eligible"], abs=1e-12)
    assert k["on_time_rate"] + k["late_rate"] == pytest.approx(1.0)
    assert 0 < k["mean_late_days"] and k["mean_duration"] > k["median_duration"] > 0  # right-skewed durations
    assert k["n_states"] == real["customer_state"].nunique()


@needs_data
def test_real_filter_consistency(real):
    f = an.apply_filters(real, ("2017-01-01", "2017-12-31"), None, None, None, None)
    assert f["purchase_date"].min() >= pd.Timestamp("2017-01-01") and f["purchase_date"].max() <= pd.Timestamp("2017-12-31")
    assert an.kpi_summary(f)["n_orders"] == len(f)
    assert an.state_performance(f)["n_orders"].sum() == len(f)
    assert an.category_performance(f)["n_orders"].sum() == len(f)
    assert an.state_pair_performance(f)["n_orders"].sum() == len(f)
    assert an.monthly_performance(f)["n_orders"].sum() == len(f)
    assert an.delay_trend(f, "W")["n_orders"].sum() == len(f)
    late = an.apply_filters(real, statuses=["late"])
    on_time = an.apply_filters(real, statuses=["on_time"])
    assert len(late) + len(on_time) == len(real) and len(late) == int(real["is_late"].sum())
    sp = an.apply_filters(real, states=["SP"])
    assert (sp["customer_state"] == "SP").all()
    assert an.kpi_summary(sp)["late_rate"] == pytest.approx(real.loc[real["customer_state"] == "SP", "is_late"].astype(float).mean())
    assert an.weight_bins(real)["n_orders"].sum() == real["total_weight_g"].notna().sum()
    assert an.freight_bins(real)["n_orders"].sum() == len(real)


@needs_data
def test_real_state_table_matches_independent_groupby(real):
    t = an.state_performance(real, min_orders=config.DEFAULT_MIN_ORDERS).set_index("customer_state")
    ref = real.groupby("customer_state", observed=True)["is_late"].agg(lambda s: s.astype(float).mean())
    assert t["late_rate"].sort_index().to_numpy() == pytest.approx(ref.sort_index().to_numpy())
    hi = an.high_delay_segments(t.reset_index(), config.DEFAULT_MIN_ORDERS, an.kpi_summary(real)["late_rate"])
    assert (hi["late_ci_low"] > an.kpi_summary(real)["late_rate"]).all() and (hi["n_orders"] >= 100).all()


@needs_data
def test_real_statistics_and_insights(real):
    res = sa.run_all_tests(real)
    chi = res["chi_square"]["customer_state"]
    assert chi["n"] == len(real) and chi["p_value"] < 1e-6 and chi["min_expected_after"] >= 5
    assert 0 < chi["cramers_v"] < 1
    ref = stats.spearmanr(real["total_freight"], real["delivery_duration_days"])
    assert res["spearman"]["total_freight~delivery_duration_days"]["rho"] == pytest.approx(ref.statistic)
    out = ins.generate_insights(real)
    assert {"overview", "region_high", "region_test", "month_high", "weight_freight", "processing_time"} <= {i.kind for i in out}
    overview = _by_kind(out, "overview")[0]
    assert f"{len(real):,}" in overview.text
    rep = ins.summary_report(real, "all data", out)
    assert "Methodology" in rep and ins.regional_report(real, 100)["n_orders"].sum() == len(real)


@needs_data
def test_real_runtime_budget(real):
    """Each public function stays well under the interactive budget on the full 96k-row frame."""
    k = an.kpi_summary(real)
    states = an.state_performance(real, 100)
    calls = {
        "apply_filters": lambda: an.apply_filters(real, ("2017-01-01", "2018-06-30"), ["SP", "RJ"], None, ["late"], "sao"),
        "kpi_summary": lambda: an.kpi_summary(real),
        "monthly_performance": lambda: an.monthly_performance(real),
        "delay_trend": lambda: an.delay_trend(real, "W"),
        "duration_summary": lambda: an.duration_summary(real),
        "state_performance": lambda: an.state_performance(real, 100),
        "category_performance": lambda: an.category_performance(real, 100),
        "state_pair_performance": lambda: an.state_pair_performance(real, 100),
        "high_delay_segments": lambda: an.high_delay_segments(states, 100, k["late_rate"]),
        "binned_stat": lambda: an.binned_stat(real, "total_weight_g", 8),
        "processing_summary": lambda: an.processing_summary(real),
        "late_by_processing_bucket": lambda: an.late_by_processing_bucket(real),
        "scatter_sample": lambda: an.scatter_sample(real, "total_weight_g", "delivery_duration_days"),
        "category_cost_comparison": lambda: an.category_cost_comparison(real, 100),
        "state_matrix": lambda: an.state_matrix(real, 30),
        "chi_square_test": lambda: sa.chi_square_test(real, "product_category"),
        "spearman_test": lambda: sa.spearman_test(real, "total_weight_g", "delivery_duration_days"),
        "run_all_tests": lambda: sa.run_all_tests(real),
        "generate_insights": lambda: ins.generate_insights(real),
        "regional_report": lambda: ins.regional_report(real, 100),
    }
    for name, fn in calls.items():
        t0 = time.perf_counter()
        fn()
        assert time.perf_counter() - t0 < 3.0, f"{name} is too slow"
