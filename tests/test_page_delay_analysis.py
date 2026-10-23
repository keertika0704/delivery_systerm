"""Tests for the Delay Analysis page (views/2_Delay_Analysis.py) and the helpers it added.

Page tests drive the page with ``streamlit.testing.AppTest`` (no sidebar is rendered, so filters are set
through ``st.session_state``) and compare the numbers on screen with an independent pandas / scipy
recomputation from ``orders_delivered.parquet``.
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from scipy import stats
from streamlit.testing.v1 import AppTest

from src import analytics as an
from src import charts, config, ui
from src import delay_narrative as nar
from src import insights as ins

ROOT = Path(__file__).resolve().parents[1]
PAGE = str(ROOT / "views" / "2_Delay_Analysis.py")
TIMEOUT = 120

needs_data = pytest.mark.skipif(not config.ORDERS_DELIVERED_PATH.is_file(),
                                reason="processed data missing: run .venv/bin/python -m src.preprocessing")


@pytest.fixture(autouse=True)
def _clear_caches():
    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


@pytest.fixture(scope="module")
def orders() -> pd.DataFrame:
    return pd.read_parquet(config.ORDERS_DELIVERED_PATH)


def run_page(states=None, categories=None, min_orders=None) -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=TIMEOUT)
    if states is not None:
        at.session_state[ui.K_STATES] = list(states)
    if categories is not None:
        at.session_state[ui.K_CATEGORIES] = list(categories)
    at.run()
    if min_orders is not None:
        at.slider(key="da_min_orders").set_value(min_orders).run()
    return at


def page_html(at: AppTest) -> str:
    parts = ([str(e.value) for e in at.get("html")] + [str(m.value) for m in at.markdown]
             + [str(c.value) for c in at.caption])
    return "\n".join(parts)


def text_of(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


def kpi(at: AppTest, label: str) -> tuple[str, str]:
    """(value, sublabel) of the KPI tile with this label, read from the rendered ``diq-kpi`` html."""
    html = "\n".join(str(e.value) for e in at.get("html"))
    m = re.search(r'diq-kpi-label">' + re.escape(label) + r'</span>.*?diq-kpi-value">(.*?)</div>(?:<div class="diq-kpi-sub">(.*?)</div>)?',
                  html, re.S)
    assert m, f"no KPI tile {label!r}"
    return m.group(1), m.group(2) or ""


def dec(v):
    """Plotly serialises numeric arrays as {dtype, bdata[, shape]}; decode to a numpy array (lists pass through)."""
    if isinstance(v, dict) and "bdata" in v:
        a = np.frombuffer(base64.b64decode(v["bdata"]), dtype=np.dtype(v["dtype"]))
        return a.reshape([int(x) for x in str(v["shape"]).split(",")]) if "shape" in v else a
    return v


def fig_of(at: AppTest, key: str) -> dict:
    el = next(e for e in at.get("plotly_chart") if e.key == key)
    return json.loads(el.proto.spec)


def bar_labels(fig: dict) -> list:
    return [y for t in fig["data"] if t.get("type") == "bar" for y in (dec(t.get("y")) if t.get("y") is not None else [])
            if y is not None]


def frame_with(at: AppTest, column: str) -> pd.DataFrame:
    for d in at.dataframe:
        if column in d.value.columns:
            return d.value
    raise AssertionError(f"no dataframe with column {column!r}")


# --------------------------------------------------------------------------- #
# Pure helpers added for the page
# --------------------------------------------------------------------------- #
def _v_scipy(table: np.ndarray) -> float:
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    return float(np.sqrt(chi2 / (table.sum() * (min(table.shape) - 1))))


def test_cramers_v_ci_point_estimate_matches_statistical_analysis():
    from src import statistical_analysis as sa
    rng = np.random.default_rng(1)
    d = pd.DataFrame({"g": rng.choice(list("ABCD"), 4000, p=[.4, .3, .2, .1])})
    d["is_late"] = (rng.random(4000) < np.where(d["g"] == "A", .2, .08)).astype(int)
    res = sa.chi_square_test(d, "g")
    v, lo, hi = an.cramers_v_bootstrap_ci(res["table"])
    assert v == pytest.approx(res["cramers_v_corrected"], rel=1e-9)
    assert lo <= v <= hi
    assert 0 <= lo < hi


def test_cramers_v_ci_deterministic_and_narrower_with_more_data():
    base = np.array([[300, 60], [280, 90], [320, 40]])
    a = an.cramers_v_bootstrap_ci(base)
    assert a == an.cramers_v_bootstrap_ci(base)
    big = an.cramers_v_bootstrap_ci(base * 10)
    assert (big[2] - big[1]) < (a[2] - a[1])
    assert big[0] == pytest.approx(a[0], abs=0.01)


def test_cramers_v_ci_independent_table_is_near_zero_and_bounded():
    v, lo, hi = an.cramers_v_bootstrap_ci(np.array([[500, 100], [500, 100]]))
    assert v == 0.0 and lo == 0.0 and 0 <= hi < 0.1


@pytest.mark.parametrize("table", [np.zeros((2, 2)), np.array([[5, 3]]), np.array([[4], [6]]), np.array([[0, 0], [7, 2]])])
def test_cramers_v_ci_degenerate_tables_give_nan(table):
    assert all(np.isnan(x) for x in an.cramers_v_bootstrap_ci(table))


def test_cramers_v_uncorrected_matches_scipy():
    t = np.array([[120, 30, 10], [80, 60, 25]])
    assert an._cramers_v_from_counts(t) == pytest.approx(_v_scipy(t))


def test_month_screen_is_public_and_equals_private(orders):
    m = an.monthly_performance(orders, 100)
    pd.testing.assert_frame_equal(ins.month_screen(m, 0.05), ins._month_screen(m, 0.05))
    assert "month_screen" in ins.__all__


def test_month_screen_frames_flags_match_independent_computation(orders):
    m = an.monthly_performance(orders, 100)
    scr, flagged = nar.month_screen_frames(m)
    rel = m[m["reliable"]]
    n_tot, k_tot = m["n_orders"].sum(), m["n_late"].sum()
    expect = set()
    for _, r in rel.iterrows():
        k1, n1, k2, n2 = r["n_late"], r["n_orders"], k_tot - r["n_late"], n_tot - r["n_orders"]
        p = (k1 + k2) / (n1 + n2)
        z = (k1 / n1 - k2 / n2) / np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
        p_two = 2 * stats.norm.sf(abs(z))
        if z > 0 and min(1.0, p_two * len(rel)) < 0.05:
            expect.add(r["purchase_year_month"])
    assert set(flagged["month"]) == expect and len(expect) > 0
    assert len(scr) == len(rel)


def test_month_screen_frames_needs_two_reliable_months():
    m = an.monthly_performance(pd.DataFrame({"purchase_year_month": ["2017-01"] * 5, "is_late": [0, 1, 0, 0, 1],
                                             "delivery_duration_days": [3.0] * 5}), 100)
    scr, flagged = nar.month_screen_frames(m)
    assert scr.empty and flagged.empty
    assert "at least two months" in nar.flagged_text(scr, flagged)
    # the screen never ran: the text must NOT claim that no month stands out (that would be an "all clear" from no test)
    not_run = nar.trend_action(flagged, screen_ran=False)
    assert "too small to test" in not_run and "No month stands out" not in not_run
    # the screen ran and found nothing: that statement is supported
    assert "No month stands out" in nar.trend_action(flagged, screen_ran=True)


def test_flagged_months_chart_colours_and_empty():
    m = pd.DataFrame({"purchase_year_month": ["2017-01", "2017-02", "2017-03", "2017-04"],
                      "n_orders": [500, 500, 500, 5], "n_late": [10, 100, 12, 3]})
    scr = pd.DataFrame({"month": ["2017-01", "2017-02", "2017-03"], "diff": [-.03, .15, -.03],
                        "p_bonferroni": [0.001, 1e-9, 0.002], "rest_late_rate": [.1, .02, .1]})
    fig = charts.flagged_months_chart(m, scr, min_orders=100)
    names = {t.name: t for t in fig.data if t.x is not None and t.x[0] is not None}
    flag = names["Significantly above other months"]
    assert [pd.Timestamp(x).strftime("%Y-%m") for x in flag.x] == ["2017-02"]
    assert flag.marker.color == config.PALETTE["red"]
    assert "Small sample (n < 100), not tested" in names
    assert fig.layout.yaxis.title.text and fig.layout.xaxis.title.text and fig.layout.title.text
    # the 60% month with n = 5 must not stretch the axis beyond the testable months
    assert fig.layout.yaxis.range[1] < 0.5
    for bad in (None, m.head(0)):
        e = charts.flagged_months_chart(bad)
        assert e.layout.annotations and charts.NO_DATA_MESSAGE in e.layout.annotations[0].text


def test_state_duration_bar_and_empty(orders):
    sp = an.state_performance(orders, 100)
    fig = charts.state_duration_bar(sp, min_orders=100)
    assert fig.data and "state" in fig.layout.title.text.lower()
    assert charts.NO_DATA_MESSAGE in charts.state_duration_bar(None).layout.annotations[0].text
    assert charts.NO_DATA_MESSAGE in charts.state_duration_bar(sp.head(0)).layout.annotations[0].text


def test_delay_trend_chart_without_reliable_period_has_sane_axis():
    tiny = an.delay_trend(pd.DataFrame({"purchase_date": pd.to_datetime(["2017-05-01", "2017-06-03"]), "is_late": [0, 1],
                                        "delivery_duration_days": [3.0, 4.0]}), "M", 100)
    fig = charts.delay_trend_chart(tiny, min_orders=100)
    assert list(fig.layout.yaxis.range) == [0, 0.1]


# --------------------------------------------------------------------------- #
# Narrative functions
# --------------------------------------------------------------------------- #
def test_trend_observed_uses_computed_numbers_and_n(orders):
    m = an.monthly_performance(orders, 100)
    txt = nar.trend_observed(m, 100)
    rel = m[m["reliable"]]
    hi = rel.loc[rel["late_rate"].idxmax()]
    assert f"{100 * hi['late_rate']:.1f}%" in txt and f"n = {int(hi['n_orders']):,}" in txt
    assert f"{len(rel)} months" in txt
    assert nar.trend_observed(m.head(0), 100) is None


def test_trend_observed_when_no_month_is_reliable():
    m = an.monthly_performance(pd.DataFrame({"purchase_year_month": ["2017-01"] * 4, "is_late": [0, 1, 0, 0],
                                             "delivery_duration_days": [3.0] * 4}), 100)
    assert "too noisy" in nar.trend_observed(m, 100)


def test_sparse_caveat_lists_small_months_with_n(orders):
    m = an.monthly_performance(orders, 100)
    txt = nar.sparse_caveat(m, 100)
    sparse = m[~m["reliable"]]
    assert f"{len(sparse)} month(s)" in txt and "2016" in txt and "n = 1" in txt
    assert nar.sparse_caveat(an.monthly_performance(orders, 0), 0) is None


def test_duration_observed_matches_summary(orders):
    s = an.duration_summary(orders)
    txt = nar.duration_observed(s)
    assert f"{s['median']:,.1f} days" in txt and f"{s['p95']:,.1f}" in txt and f"n = {s['n']:,}" in txt
    assert nar.duration_observed({"n": 0}) is None


def test_segment_duration_text_handles_no_reliable_segment(orders):
    cp = an.category_performance(orders.head(50), 100)
    assert "No category reaches 100 orders" in nar.segment_duration_text(cp, "product_category", an.duration_summary(orders), 100)


def test_binned_observed_requires_two_reliable_bins(orders):
    wb = an.weight_bins(orders, min_orders=100)
    ok = nar.binned_observed(wb, "weight", "g", 100, 0.07)
    assert f"n = {int(wb.iloc[0]['n_orders']):,}" in ok
    assert "Fewer than two" in nar.binned_observed(wb, "weight", "g", 10**9, 0.07)
    assert nar.binned_observed(wb.head(0), "weight", "g", 100, 0.07) is None


def test_category_observed_reports_flags_with_n_and_chance_note(orders):
    perf = an.category_performance(orders, 100)
    overall = an.kpi_summary(orders)["late_rate"]
    obs, assoc, hi, lo = nar.category_observed(perf, overall, 100)
    expect = an.high_delay_segments(perf[perf["product_category"] != "unknown"], 100, overall, 0.05, "high")
    assert list(hi["product_category"]) == list(expect["product_category"])
    assert "by chance" in assoc and f"{len(hi)} of" in assoc
    for c in hi["product_category"].head(2):
        assert nar.cat_name(c) in assoc
    o2, a2, h2, l2 = nar.category_observed(perf.head(0), overall, 100)
    assert o2 is None and h2.empty
    assert "No named category" in nar.category_observed(perf, overall, 10**9)[0]


def test_processing_observed_uses_summary_and_buckets(orders):
    ps = an.processing_summary(orders)
    b = an.late_by_processing_bucket(orders, min_orders=100)
    txt = nar.processing_observed(ps, b, 100)
    assert f"{ps['median']:,.1f} days" in txt and f"n = {ps['n']:,}" in txt
    assert f"{100 * b.iloc[0]['late_rate']:.1f}%" in txt
    assert nar.processing_observed({"n": 0}, b.head(0), 100) is None


def test_tests_headline_counts_and_no_significance_branch():
    summ = pd.DataFrame({"test": ["chi-square", "spearman"], "variables": ["a x is_late", "b ~ c"], "statistic": [1.0, 0.1],
                         "p_value": [0.5, 0.4], "p_holm": [0.9, 0.9], "effect_size": [0.01, 0.01],
                         "effect_label": ["negligible", "negligible"], "n": [50, 50], "significant_holm": [False, False]})
    assert "None of the 2 tests" in nar.tests_headline({"summary": summ, "n_orders": 50})
    summ2 = summ.assign(p_value=[1e-9, 1e-9], p_holm=[1e-9, 1e-9], significant_holm=[True, True],
                        effect_label=["small", "negligible"])
    txt = nar.tests_headline({"summary": summ2, "n_orders": 50})
    assert "2 of 2 tests are" in txt and "only 1 of them has" in txt and "a x late delivery (small)" in txt
    assert "is_late" not in txt and "of them have" not in txt
    one = nar.tests_headline({"summary": summ2.head(1), "n_orders": 50})
    assert "1 of 1 test is" in one
    # the "tiny differences reach p < 0.05" sentence only holds for large n
    assert "tiny and practically irrelevant" not in txt
    big = nar.tests_headline({"summary": summ2, "n_orders": 50_000}, min_orders=100)
    assert "tiny and practically irrelevant" in big
    # a view smaller than the minimum is described as unstable and the effect labels are not listed as findings
    few = nar.tests_headline({"summary": summ2, "n_orders": 50}, min_orders=100)
    assert "indicative only" in few and "small)" not in few and "tiny" not in few
    assert nar.tests_headline({"summary": summ.head(0)}) is None


def test_friendly_names_plural_and_lists():
    assert nar.friendly("customer_state x is_late") == "customer state x late delivery"
    assert nar.friendly("total_weight_g ~ delivery_duration_days") == "order weight ~ delivery time"
    assert nar.friendly("delivery_duration_days by same_state") == "delivery time by same-state route"
    assert "_" not in nar.friendly("estimated_delivery_days freight_ratio purchase_month product_category total_freight")
    assert nar.plural(1, "month") == "1 month" and nar.plural(3, "month") == "3 months"
    assert nar.plural(1, "category", "categories") == "1 category" and nar.plural(2, "category", "categories") == "2 categories"
    assert nar.and_list(["2017"]) == "2017" and nar.and_list(["2017", "2018"]) == "2017 and 2018"
    assert nar.and_list(["2016", "2017", "2018"]) == "2016, 2017 and 2018"


def test_segment_duration_text_needs_two_segments(orders):
    sp = an.state_performance(orders[orders["customer_state"] == "SP"], 100)
    assert len(sp) == 1
    assert nar.segment_duration_text(sp, "customer_state", an.duration_summary(orders), 100) is None   # not "1 states ... 1.0"
    two = an.state_performance(orders[orders["customer_state"].isin(["SP", "RJ"])], 100)
    txt = nar.segment_duration_text(two, "customer_state", an.duration_summary(orders), 100)
    assert "Among 2 states" in txt


def test_category_action_does_not_claim_all_clear_when_nothing_was_ranked():
    none_ranked = nar.category_action(pd.DataFrame(), ranked=False)
    assert "no category-level conclusion" in none_ranked and "not supported" not in none_ranked
    assert "not supported by this view" in nar.category_action(pd.DataFrame(), ranked=True)


def test_category_observed_pluralises_and_uses_half_alpha_chance_count(orders):
    perf = an.category_performance(orders, 100)
    overall = an.kpi_summary(orders)["late_rate"]
    rel = perf[(perf["product_category"] != "unknown") & (perf["n_orders"] >= 100)]
    _, assoc, hi, _ = nar.category_observed(perf, overall, 100)
    assert f"about {len(rel) * 0.05 / 2:.1f} would be expected" in assoc      # one-sided: n * alpha / 2
    one = perf[perf["product_category"] == perf.sort_values("n_orders").iloc[-1]["product_category"]]
    o, a, h1, l1 = nar.category_observed(one, overall, 100)
    # a single reliable category cannot be compared with anything: no "ranges from X to X" prose, no association claim
    assert "Only one category" in o and "cannot be compared" in o and "ranges from" not in o
    assert a is None and h1.empty and l1.empty
    two = perf[perf["product_category"].isin(perf.sort_values("n_orders").iloc[-2:]["product_category"])]
    o2, a2, _, _ = nar.category_observed(two, overall, 100)
    assert "Among 2 categories" in o2 and "1 categories" not in o2 + (a2 or "")


def test_sparse_caveat_joins_years_with_and():
    m = pd.DataFrame({"purchase_year_month": ["2017-01", "2018-02"], "n_orders": [3, 4], "n_late": [0, 1], "reliable": [False, False]})
    assert "all in 2017 and 2018" in nar.sparse_caveat(m, 100)


def test_binned_observed_lightest_and_heaviest_are_first_and_last_bins(orders):
    wb = an.weight_bins(orders, min_orders=100)
    txt = nar.binned_observed(wb, "weight", "g", 100, 0.07)
    first, last = wb.iloc[0], wb.iloc[-1]
    assert f"lightest group ({nar.friendly_bin(first['bin'], 'g')}, n = {int(first['n_orders']):,}) has a late rate of **{100 * first['late_rate']:.1f}%**" in txt
    assert f"heaviest group ({nar.friendly_bin(last['bin'], 'g')}, n = {int(last['n_orders']):,}) has **{100 * last['late_rate']:.1f}%**" in txt


def test_friendly_bin_reads_without_interval_notation():
    f = nar.friendly_bin
    assert f("[0, 500]", "g") == "0 to 500 g" and f("(500, 1000]", "g") == "over 500 to 1,000 g"
    assert f("(10000, 30000]", "g") == "over 10,000 to 30,000 g" and f("> 30000", "g") == "over 30,000 g"
    assert f("[0, 10]", "R$") == "R$ 0 to 10" and f("> 100", "R$") == "over R$ 100"
    assert f("something else", "g") == "something else"
    wb = an.weight_bins(pd.read_parquet(config.ORDERS_DELIVERED_PATH), min_orders=100) if config.ORDERS_DELIVERED_PATH.is_file() else None
    if wb is not None:
        txt = nar.binned_observed(wb, "weight", "g", 100, 0.07)
        assert "[" not in txt and "(0," not in txt and "]" not in txt


def test_spearman_sentence_with_and_without_ci():
    r = {"rho": 0.2, "ci_low": 0.1, "ci_high": 0.3, "n": 99, "strength_label": "weak", "direction": "positive"}
    assert "95% CI 0.10 to 0.30" in nar.spearman_sentence(r, "x", "y")
    assert "CI" not in nar.spearman_sentence({**r, "ci_low": np.nan, "ci_high": np.nan}, "x", "y")
    assert "could not be computed" in nar.spearman_sentence({"rho": np.nan, "note": "too few"}, "x", "y")


# --------------------------------------------------------------------------- #
# The page itself
# --------------------------------------------------------------------------- #
@needs_data
def test_default_page_renders_without_exceptions_and_has_six_tabs():
    at = run_page()
    assert not at.exception
    assert [t.label for t in at.tabs] == [
        ":material/trending_up: Delay trends", ":material/schedule: Delivery duration", ":material/scale: Weight & freight",
        ":material/category: Product categories", ":material/local_shipping: Processing time",
        ":material/functions: Statistical tests"]
    html = page_html(at)
    assert "Delay Analysis" in html
    for label in ("Observed finding", "Statistical association", "Suggested action", "Data caveat"):
        assert label in html
    # every widget has a key (unique) and the main controls exist
    assert at.slider(key="da_min_orders").value == config.DEFAULT_MIN_ORDERS
    assert at.radio(key="da_trend_gran").value == "Month"


@needs_data
def test_kpis_match_pandas_recomputation(orders):
    at = run_page()
    txt = text_of(page_html(at))
    n, late = len(orders), int(orders["is_late"].sum())
    assert f"{n:,}" in txt
    assert f"{100 * late / n:.1f}%" in txt and f"{late:,} late" in txt
    assert f"{orders['delivery_duration_days'].mean():.1f} days" in txt
    lo, hi = an.wilson_ci(late, n)
    assert f"{100 * lo:.1f}% to {100 * hi:.1f}%" in txt
    late_days = orders.loc[orders["is_late"] == 1, "delivery_delay_days"].mean()
    assert f"{late_days:.1f} days" in txt


@needs_data
def test_flagged_months_table_matches_independent_screen(orders):
    at = run_page()
    t = frame_with(at, "Adjusted p")
    m = an.monthly_performance(orders, 100)
    _, flagged = nar.month_screen_frames(m)
    assert list(t["Month"]) == [nar.month_name(x) for x in flagged["month"]]
    for _, row in t.iterrows():
        month = pd.Timestamp(row["Month"]).strftime("%Y-%m")
        d = orders[orders["purchase_year_month"] == month]
        assert row["Orders (n)"] == len(d)
        assert row["Late orders"] == int(d["is_late"].sum())
        assert row["Late rate (%)"] == pytest.approx(100 * d["is_late"].mean())
    txt = text_of(page_html(at))
    assert "significantly higher" in txt and "Bonferroni" in txt


@needs_data
def test_category_table_matches_groupby_and_respects_min_orders(orders):
    at = run_page()
    t = frame_with(at, "Category")
    assert (t["Orders (n)"] >= 100).all() and t["Reliable"].all()
    assert "Unknown" not in set(t["Category"])
    g = orders.groupby("product_category").agg(n=("is_late", "size"), late=("is_late", "mean"), days=("delivery_duration_days", "mean"))
    for _, r in t.head(8).iterrows():
        key = r["Category"].lower().replace(" ", "_")
        match = g[g.index.str.replace("_", " ").str.lower() == r["Category"].lower()]
        assert len(match) == 1, key
        assert r["Orders (n)"] == match["n"].iloc[0]
        assert r["Late rate (%)"] == pytest.approx(100 * match["late"].iloc[0])
        assert r["Mean days"] == pytest.approx(match["days"].iloc[0])
        assert r["CI low (%)"] <= r["Late rate (%)"] <= r["CI high (%)"]
    # sorted by late rate (highest first) by default
    assert list(t["Late rate (%)"]) == sorted(t["Late rate (%)"], reverse=True)


@needs_data
def test_raising_min_orders_filters_the_category_table():
    at = run_page(min_orders=1000)
    assert not at.exception
    t = frame_with(at, "Category")
    assert len(t) > 0 and (t["Orders (n)"] >= 1000).all()


@needs_data
def test_category_toggle_shows_small_categories_with_reliable_flag():
    at = run_page()
    at.toggle(key="da_cat_reliable").set_value(False).run()
    assert not at.exception
    t = frame_with(at, "Category")
    assert (~t["Reliable"]).any() and (t["Orders (n)"] < 100).equals(~t["Reliable"])


@needs_data
def test_state_filter_changes_numbers(orders):
    sp = orders[orders["customer_state"] == "SP"]
    at = run_page(states=["SP"])
    assert not at.exception
    txt = text_of(page_html(at))
    assert f"{len(sp):,}" in txt and f"{100 * sp['is_late'].mean():.1f}%" in txt
    # the same-state / customer-state chi-square is degenerate for one state: shown as 'Not run', not as a traceback
    assert "Not run" in txt


@needs_data
def test_tiny_filtered_sample_never_raises():
    at = run_page(states=["SP"], categories=["flowers"])
    assert not at.exception
    txt = text_of(page_html(at))
    assert "Small sample" in txt or "fewer than 100" in txt


@needs_data
def test_empty_filter_result_shows_empty_state_in_every_tab():
    at = run_page(states=["RR"], categories=["flowers"])
    assert not at.exception
    html = page_html(at)
    assert html.count("No orders match the current filters") >= 6
    assert "Orders analysed" not in html          # KPI row is skipped when there is nothing to report
    assert "Retrospective analysis" in html       # the banner stays visible
    assert "significance is not practical importance" in text_of(html)


@needs_data
def test_processing_tab_banner_and_numbers(orders):
    at = run_page()
    txt = text_of(page_html(at))
    assert "Carrier handoff time is NOT known at purchase and is not a model input" in txt
    ps = an.processing_summary(orders)
    assert f"{ps['median']:.1f} days" in txt and f"{ps['n_missing']:,}" in txt


@needs_data
def test_statistical_tests_tab_shows_effect_sizes_and_caveat(orders):
    from src import statistical_analysis as sa
    at = run_page()
    txt = text_of(page_html(at))
    assert "Statistical significance is not practical importance" in txt
    res = sa.run_all_tests(orders)
    ss = res["chi_square"]["same_state"]
    assert f"V = {ss['cramers_v']:.3f}" in txt
    assert f"chi-square = {ss['chi2']:,.1f}" in txt and f"df = {ss['dof']}" in txt and f"n = {ss['n']:,}" in txt
    sp = res["spearman"]["total_freight~delivery_duration_days"]
    assert f"rho = {sp['rho']:.3f}" in txt and f"{sp['ci_low']:.3f} to {sp['ci_high']:.3f}" in txt
    # independent check of Cramer's V with scipy
    table = pd.crosstab(orders["same_state"], orders["is_late"]).to_numpy()
    assert ss["cramers_v"] == pytest.approx(_v_scipy(table))
    summary = frame_with(at, "Holm-adjusted p")
    assert len(summary) == len(res["summary"]) == 12
    assert "Significant after Holm" in summary.columns
    # plain-language readings sit in expanders
    assert len(at.expander) >= 12


@needs_data
def test_weight_freight_route_filter_and_scatter_toggle(orders):
    at = run_page()
    at.selectbox(key="da_wf_route").set_value("Same state (seller and customer)").run()
    assert not at.exception
    same = orders[orders["same_state"] == 1]
    assert f"n = {len(same):,}" in text_of(page_html(at))
    at.checkbox(key="da_wf_scatter").set_value(True).run()
    at.radio(key="da_wf_bins").set_value("Equal-count groups").run()
    assert not at.exception


@needs_data
def test_trend_granularity_and_window_controls_do_not_break():
    at = run_page()
    for gran in ("Week", "Quarter", "Month"):
        at.radio(key="da_trend_gran").set_value(gran).run()
        assert not at.exception
    at.slider(key="da_trend_window").set_value(4).run()
    assert not at.exception


@needs_data
def test_no_deprecated_use_container_width_in_page_source():
    src = Path(PAGE).read_text()
    assert "use_container_width" not in src
    assert "st.cache_data" in src and "get_filtered_orders" in src
    assert "read_csv" not in src and "read_parquet" not in src


# --------------------------------------------------------------------------- #
# Review fixes: honest small-sample behaviour, exact KPI values, friendly names, state persistence
# --------------------------------------------------------------------------- #
RAW_NAMES = re.compile(r"\b(total_weight_g|total_freight|freight_ratio|delivery_duration_days|estimated_delivery_days|"
                       r"customer_state|product_category|purchase_month|same_state|is_late)\b")


@needs_data
def test_headline_kpi_tiles_show_exact_values(orders):
    at = run_page()
    n, late = len(orders), int(orders["is_late"].sum())
    assert kpi(at, "Late rate")[0] == f"{100 * late / n:.1f}%"                 # not 100 - rate, not the mean
    assert kpi(at, "Orders analysed")[0] == f"{n:,}"
    assert kpi(at, "Mean delivery time")[0] == f"{orders['delivery_duration_days'].mean():,.1f} days"
    assert kpi(at, "Mean days late")[0] == f"{orders.loc[orders['is_late'] == 1, 'delivery_delay_days'].mean():,.1f} days"
    # the processing tab: median handoff (not the mean)
    assert kpi(at, "Median handoff")[0] == f"{orders['processing_days'].median():,.1f} days"
    assert kpi(at, "Mean handoff")[0] == f"{orders['processing_days'].mean():,.1f} days"


@needs_data
def test_small_view_is_flagged_in_kpis_and_not_in_a_big_view(orders):
    big = page_html(run_page())
    assert 'diq-callout-title">Small sample' not in big and "indicative only" not in kpi(run_page(), "Mean days late")[1]
    at = run_page(states=["AP"])
    assert not at.exception
    html = page_html(at)
    n_ap = int((orders["customer_state"] == "AP").sum())
    assert 'diq-callout-title">Small sample' in html and f"Only n = {n_ap:,}" in text_of(html)
    assert "indicative only" in kpi(at, "Mean days late")[1]


@needs_data
def test_month_screen_that_could_not_run_gives_no_all_clear():
    at = run_page(states=["AP"])
    txt = text_of(page_html(at))
    assert "needs at least two months with enough orders" in txt
    assert "too small to test for problem months" in txt
    assert "No month stands out" not in txt and "no period-specific problem" not in txt
    # the healthy full-data page still reports a screen result
    full = text_of(page_html(run_page()))
    assert "too small to test for problem months" not in full


@needs_data
def test_state_chart_lists_every_state_with_orders_and_a_true_subtitle(orders):
    cat = "health_beauty"
    d = orders[orders["product_category"] == cat]
    n_states = d["customer_state"].nunique()
    at = run_page(categories=[cat], min_orders=1000)       # most states are "small" here, far above the old cap of 9
    assert not at.exception
    fig = fig_of(at, "da_dur_state_chart")
    drawn = bar_labels(fig)
    assert len(drawn) == n_states and len(set(drawn)) == n_states
    assert f"all {n_states} states with orders" in text_of(page_html(at))
    assert "hidden" not in (fig["layout"]["title"].get("subtitle", {}).get("text", ""))


@needs_data
def test_single_state_view_has_no_meaningless_state_comparison():
    at = run_page(states=["SP"])
    txt = text_of(page_html(at))
    assert "1 states" not in txt and "a factor of 1.0" not in txt
    assert "Compare the slowest states" not in txt
    assert "Compare the slowest states" in text_of(page_html(run_page()))


@needs_data
def test_tiny_view_tests_tab_is_marked_indicative_and_reports_true_n(orders):
    sub = orders[(orders["customer_state"] == "SP") & (orders["product_category"] == "flowers")]
    at = run_page(states=["SP"], categories=["flowers"])
    assert not at.exception
    txt = text_of(page_html(at))
    assert "Too few orders to test reliably" in txt
    assert "tiny and practically irrelevant" not in txt                      # false at this n
    assert "very strong effect" not in txt and "large effect" not in txt
    assert f"n = {len(sub):,} orders in this view" in txt and "n = 0 orders" not in txt
    assert "indicative only" in txt
    eff = frame_with(at, "Holm-adjusted p")["Effect"]
    assert set(eff) == {"indicative only"}
    # a big view still shows the real effect labels
    big = text_of(page_html(run_page()))
    assert "Too few orders to test reliably" not in big and "moderate effect" in big


@needs_data
def test_no_raw_column_names_reach_the_screen():
    for kwargs in ({}, {"states": ["SP"]}):
        at = run_page(**kwargs)
        txt = text_of(page_html(at))
        assert not RAW_NAMES.search(txt), RAW_NAMES.search(txt).group(0)
        glance = frame_with(at, "Holm-adjusted p")
        for v in list(glance["Variables"]) + list(glance["Test"]):
            assert "_" not in v, v
    assert "customer state x late delivery" in text_of(page_html(run_page()))


@needs_data
def test_tab_controls_survive_a_filter_with_no_orders():
    at = run_page()
    at.radio(key="da_trend_gran").set_value("Week").run()
    at.slider(key="da_trend_window").set_value(4).run()
    at.selectbox(key="da_cat_sort").set_value("Orders (most first)").run()
    at.toggle(key="da_cat_unknown").set_value(True).run()
    at.radio(key="da_dur_stat").set_value("Median").run()
    at.session_state[ui.K_STATES] = ["RR"]
    at.session_state[ui.K_CATEGORIES] = ["flowers"]
    at.run()
    assert not at.exception and "No orders match the current filters" in page_html(at)
    at.session_state[ui.K_STATES] = []
    at.session_state[ui.K_CATEGORIES] = []
    at.run()
    assert not at.exception
    assert at.radio(key="da_trend_gran").value == "Week" and at.slider(key="da_trend_window").value == 4
    assert at.selectbox(key="da_cat_sort").value == "Orders (most first)"
    assert at.toggle(key="da_cat_unknown").value is True and at.radio(key="da_dur_stat").value == "Median"


@needs_data
def test_changing_filters_recomputes_cached_results(orders):
    at = run_page(states=["SP"])
    sp = orders[orders["customer_state"] == "SP"]
    assert kpi(at, "Late rate")[0] == f"{100 * sp['is_late'].mean():.1f}%"
    at.session_state[ui.K_STATES] = ["RJ"]
    at.run()
    rj = orders[orders["customer_state"] == "RJ"]
    assert kpi(at, "Late rate")[0] == f"{100 * rj['is_late'].mean():.1f}%"
    t = frame_with(at, "Category")
    top = t.iloc[0]
    g = rj[rj["product_category"].map(lambda c: c.replace("_", " ").lower()) == top["Category"].lower()]
    assert top["Orders (n)"] == len(g)


@needs_data
def test_category_table_vs_overall_column_is_late_rate_minus_overall(orders):
    at = run_page()
    t = frame_with(at, "vs overall (pp)")
    overall = 100 * orders["is_late"].mean()
    assert np.allclose(t["vs overall (pp)"], t["Late rate (%)"] - overall, atol=1e-9)
    assert t.iloc[0]["vs overall (pp)"] > 0          # highest late rate first: above the overall rate


@needs_data
def test_rolling_average_line_matches_independent_pooled_rolling_rate(orders):
    at = run_page()
    at.slider(key="da_trend_window").set_value(3).run()
    fig = fig_of(at, "da_trend_chart")
    tr = next(t for t in fig["data"] if t.get("name") == "3-month rolling late rate")
    m = an.monthly_performance(orders, 100)
    m = m.set_index(pd.PeriodIndex(m["purchase_year_month"], freq="M").to_timestamp())
    ok = m["n_orders"] >= 100
    exp = (m["n_late"].where(ok).rolling(3, min_periods=3).sum() / m["n_orders"].where(ok).rolling(3, min_periods=3).sum())
    got = pd.Series(np.asarray(dec(tr["y"]), dtype="float64"), index=pd.to_datetime(dec(tr["x"])))
    both = exp.rename("e").to_frame().join(got.rename("g"), how="inner").dropna()
    assert len(both) >= 15 and np.allclose(both["e"], both["g"], atol=1e-9)


@needs_data
def test_equal_count_toggle_gives_equal_sized_weight_bins_and_fixed_does_not():
    at = run_page()
    def bin_ns(a):
        bar = next(t for t in fig_of(a, "da_w_late")["data"] if t.get("type") == "bar" and t.get("customdata") is not None)
        return [float(c[1]) for c in dec(bar["customdata"])]
    fixed = bin_ns(at)
    assert max(fixed) / min(fixed) > 20                    # round-number ranges: very unequal bins
    at.radio(key="da_wf_bins").set_value("Equal-count groups").run()
    eq = bin_ns(at)
    assert max(eq) / min(eq) < 1.2                         # quantile bins: (almost) the same size


@needs_data
def test_kruskal_note_counts_only_states_with_enough_orders(orders):
    d = orders[orders["product_category"] == "health_beauty"]
    n_ok = int((d["customer_state"].value_counts() >= 30).sum())
    assert 2 <= n_ok < d["customer_state"].nunique()
    txt = text_of(page_html(run_page(categories=["health_beauty"])))
    assert f"Kruskal-Wallis test across {n_ok} states (each with at least 30 orders)" in txt


@needs_data
def test_category_charts_never_include_the_unknown_category(orders):
    assert (orders["product_category"] == "unknown").sum() >= 100          # it would qualify as "reliable"
    at = run_page()
    for key in ("da_cat_late", "da_cat_dur", "da_cat_freight"):
        labels = [str(y) for y in bar_labels(fig_of(at, key))]
        assert labels and not any("nknown" in x for x in labels), key
    # the duration chart on the "Delivery duration" tab: show the 30 longest AND the 30 shortest so 'unknown' would surface
    cp = an.category_performance(orders, 100)
    assert int((cp["n_orders"] >= 100).sum()) <= 60
    at.slider(key="da_dur_topn").set_value(30).run()
    for order in ("Longest", "Shortest"):
        at.radio(key="da_dur_order").set_value(order).run()
        labels = [str(y) for y in bar_labels(fig_of(at, "da_dur_cat_chart"))]
        assert len(labels) > 20 and not any("nknown" in x for x in labels), order


# --- chart helpers used by the page ------------------------------------------------------
def test_binned_late_bars_use_compact_horizontal_tick_labels_and_chips_for_values():
    b = pd.DataFrame({"bin": ["[0, 500]", "(500, 1000]", "(1000, 2800]", "> 30000"], "bin_low": [0.0, 500.0, 1000.0, 30000.0],
                      "bin_high": [500.0, 1000.0, 2800.0, np.inf], "n_orders": [900, 500, 300, 150], "n_late": [60, 30, 25, 20],
                      "late_rate": [.066, .06, .083, .133], "late_ci_low": [.05, .04, .06, .09], "late_ci_high": [.08, .08, .11, .18]})
    fig = charts.weight_late_rate_bars(b, overall_rate=0.07, min_orders=100)
    cats = list(fig.data[0].x)
    assert [c.split("<br>")[0] for c in cats] == ["0–500", "500–1k", "1k–2.8k", "> 30k"]
    assert fig.layout.xaxis.tickangle == 0
    assert len([a for a in fig.layout.annotations if a.bgcolor]) == 4        # one white-backed value label per bar
    assert "(1000, 2800]" in fig.data[0].customdata[2][0]                     # the hover still shows the exact bin


def test_log_ticks_are_few_and_readable():
    for lo, hi in ((300, 1000), (20, 60_000), (1.5, 9000)):
        t = charts._log_ticks(lo, hi)
        assert 2 <= len(t) <= 9 and all(lo <= v <= hi for v in t) and t == sorted(t)
    assert charts._log_ticks(0, 10) == [] and charts._log_ticks(5, 5) == []


def test_weight_vs_duration_log_axis_uses_explicit_ticks():
    b = pd.DataFrame({"bin": list("abcd"), "mean_value": [310.0, 420.0, 650.0, 900.0], "mean_duration": [20.0, 22.0, 25.0, 24.0],
                      "n_orders": [4, 5, 3, 2], "late_rate": [0.0, .1, .2, 0.0]})
    fig = charts.weight_vs_duration(b, min_orders=100)
    assert fig.layout.xaxis.tickmode == "array" and 3 <= len(fig.layout.xaxis.tickvals) <= 7


def test_trend_chart_without_reliable_period_says_so_inside_the_rate_panel():
    tiny = an.delay_trend(pd.DataFrame({"purchase_date": pd.to_datetime(["2017-05-01", "2017-06-03"]), "is_late": [0, 1],
                                        "delivery_duration_days": [3.0, 4.0]}), "M", 100)
    fig = charts.delay_trend_chart(tiny, min_orders=100)
    assert any("No period has 100 or more orders" in a.text for a in fig.layout.annotations)
    ok = charts.delay_trend_chart(an.delay_trend(pd.DataFrame({"purchase_date": pd.date_range("2017-01-01", periods=400),
                                                               "is_late": [0, 1] * 200, "delivery_duration_days": [5.0] * 400}), "M", 20), min_orders=20)
    assert not any("No period has" in a.text for a in ok.layout.annotations)


def test_state_duration_bar_max_small_draws_all_small_states_or_counts_the_hidden_ones():
    sp = pd.DataFrame({"customer_state": [f"S{i}" for i in range(20)], "n_orders": [500] * 2 + [10] * 18,
                       "mean_duration": np.linspace(8, 30, 20), "median_duration": np.linspace(7, 28, 20), "late_rate": [.05] * 20})
    allbars = charts.state_duration_bar(sp, top_n=20, min_orders=100, max_small=20)
    drawn = lambda f: sum(len(t.y) for t in f.data if t.type == "bar" and t.y is not None and t.y[0] is not None)
    assert drawn(allbars) == 20
    capped = charts.state_duration_bar(sp, top_n=20, min_orders=100)
    assert drawn(capped) == 2 + 6                                   # old cap max(3, 20 // 3) = 6 small states
    assert "12 segments with n < 100 hidden" in capped.layout.title.subtitle.text


# --------------------------------------------------------------------------- #
# Second review round: statistic formatting, table layout, narrative counts, single category, level labels
# --------------------------------------------------------------------------- #
def test_stat_text_keeps_large_statistics_whole():
    assert nar.stat_text("mann_whitney", 1709892180.5) == "1,709,892,180"       # %.4g printed 710000000
    assert nar.stat_text("kruskal", 24168.31) == "24,168"
    assert nar.stat_text("chi-square", 1782.376) == "1,782.4"
    assert nar.stat_text("spearman", 0.08697838) == "0.087" and nar.stat_text("spearman", -0.02514) == "-0.025"
    assert nar.stat_text("spearman", np.nan) == "n/a" and nar.stat_text("chi-square", None) == "n/a"


@needs_data
def test_glance_table_statistics_agree_with_the_cards_and_are_not_clipped(orders):
    from src import statistical_analysis as sa
    at = run_page()
    t = frame_with(at, "Holm-adjusted p")
    res = sa.run_all_tests(orders)
    stat = dict(zip(t["Variables"], t["Statistic"]))
    u = res["group_comparison"]["delivery_duration_days~same_state"]["statistic"]
    h = res["group_comparison"]["delivery_duration_days~customer_state"]["statistic"]
    chi = res["chi_square"]["customer_state"]["chi2"]
    assert stat["delivery time by same-state route"] == f"{u:,.0f}" == "1,709,892,180"
    assert stat["delivery time by customer state"] == f"{h:,.0f}"
    assert stat["customer state x late delivery"] == f"{chi:,.1f}"
    assert stat["order weight ~ delivery time"] == f"{res['spearman']['total_weight_g~delivery_duration_days']['rho']:.3f}"
    txt = text_of(page_html(at))
    # the same numbers appear on the cards below the table
    assert f"U = {u:,.0f}" in txt and f"H = {h:,.0f}" in txt and f"chi-square = {chi:,.1f}" in txt


@needs_data
def test_wide_tables_fit_at_1440px_and_keep_the_key_flags_next_to_their_values():
    at = run_page()
    for column, last_col in (("Holm-adjusted p", "n"), ("Freight / price (%)", "Reliable")):
        el = next(d for d in at.dataframe if column in d.value.columns)
        cfg = json.loads(el.proto.columns)
        cols = list(el.value.columns)
        assert cols[-1] == last_col
        widths = [cfg[c]["width"] for c in cols]
        assert all(isinstance(w, int) for w in widths), widths          # explicit pixel widths, no "large" columns
        assert sum(widths) <= 1060, sum(widths)                          # measured content width of a 1440 px window with the sidebar open
    glance = frame_with(at, "Holm-adjusted p")
    cols = list(glance.columns)
    assert cols.index("Significant after Holm") == cols.index("Holm-adjusted p") + 1      # the flag sits next to its p-value


@needs_data
def test_category_flag_lists_name_four_and_count_the_rest(orders):
    perf = an.category_performance(orders, 100)
    overall = an.kpi_summary(orders)["late_rate"]
    _, assoc, hi, lo = nar.category_observed(perf, overall, 100)
    assert len(hi) > 4 and len(lo) > 4
    assert f"and {len(hi) - 4} more" in assoc and f"Reliably below average ({len(lo)})" in assoc
    assert f"and {len(lo) - 4} more" in assoc
    named = sum(nar.cat_name(c) in assoc for c in hi["product_category"])
    assert named == 4                                                    # four named, the remainder only counted
    act = nar.category_action(hi)
    assert f"and {len(hi) - 4} more" in act                              # same four + same remainder as the association note
    for c in list(hi["product_category"])[:4]:
        assert nar.cat_name(c) in act
    # the page shows the same sentence
    assert f"and {len(hi) - 4} more" in text_of(page_html(run_page()))


@needs_data
def test_single_reliable_category_view_says_nothing_can_be_compared():
    at = run_page(categories=["health_beauty"])
    assert not at.exception
    txt = text_of(page_html(at))
    assert "Only one category (Health beauty) has at least 100 orders" in txt and "cannot be compared" in txt
    assert "ranges from" not in txt and "Reliably below average" not in txt
    assert "No category is reliably above average" not in txt             # the old degenerate advice
    keys = {e.key for e in at.get("plotly_chart")}
    assert not {"da_cat_late", "da_cat_dur", "da_cat_freight"} & keys    # no one-bar rankings
    assert len(frame_with(at, "Category")) == 1                          # the table still lists the category
    # the shipping-cost choice is remembered although its radio is not drawn here
    base = run_page()
    base.radio(key="da_cat_fmetric").set_value("Freight as % of item price").run()
    base.session_state[ui.K_CATEGORIES] = ["health_beauty"]
    base.run()
    assert not base.exception
    base.session_state[ui.K_CATEGORIES] = []
    base.run()
    assert base.radio(key="da_cat_fmetric").value == "Freight as % of item price"


def test_category_action_for_exactly_one_compared_category():
    txt = nar.category_action(pd.DataFrame(), ranked=True, n_compared=1)
    assert "Only one category" in txt and "not supported" not in txt and "no category-level conclusion" not in txt
    assert "not supported" in nar.category_action(pd.DataFrame(), ranked=True, n_compared=5)


def test_level_labels_and_relabelled_reading():
    assert nar.level_label("same_state", "0") == "Different states" and nar.level_label("same_state", 1) == "Same state"
    assert nar.level_label("purchase_month", "3") == "March" and nar.level_label("purchase_month", "12") == "December"
    assert nar.level_label("purchase_month", "x") == "x" and nar.level_label("customer_state", "SP") == "SP"
    assert nar.level_label("product_category", "health_beauty") == "Health beauty"
    txt = "The level that departs most from independence is '0' (late rate 8.0%, n = 61,774)."
    assert nar.relabel_reading(txt, "same_state", "0") == txt.replace("'0'", "'Different states'")


@needs_data
def test_chi_square_readings_and_level_tables_show_labels_not_codes():
    at = run_page()
    txt = text_of(" ".join(str(m.value) for e in at.expander for m in e.markdown))       # the readings sit inside expanders
    assert "departs most from independence is 'Different states'" in txt
    assert re.search(r"departs most from independence is '(January|February|March|April|May|June|July|August|September|October|"
                     r"November|December)'", txt)
    assert not re.search(r"departs most from independence is '\d+'", txt)
    levels = [set(d.value["Level"]) for d in at.dataframe if "Adjusted residual" in d.value.columns]
    assert len(levels) == 4
    flat = set().union(*levels)
    assert {"Different states", "Same state"} <= flat and not {"0", "1"} & flat
    months = {"January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"}
    assert any(s <= months for s in levels)
    groups = [d.value for d in at.dataframe if "Group" in d.value.columns]
    assert any(set(g["Group"]) == {"Different states", "Same state"} for g in groups)


# --- coverage gaps found by mutation testing ---------------------------------------------
@needs_data
@pytest.mark.parametrize("route,mask", [("All orders", None), ("Same state (seller and customer)", 1), ("Different states", 0)])
def test_route_selector_filters_orders_and_weight_caption(orders, route, mask):
    at = run_page()
    at.selectbox(key="da_wf_route").set_value(route).run()
    assert not at.exception
    d = orders if mask is None else orders[orders["same_state"] == mask]
    cap = next(c.value for c in at.caption if "orders without a usable weight" in str(c.value))
    assert f"{len(d):,} orders" in cap and f"overall late rate {100 * d['is_late'].mean():.1f}%" in cap
    assert f"{int(d['total_weight_g'].isna().sum()):,} orders without a usable weight" in cap
    # the correlations on the page are computed on the selected orders as well
    from scipy import stats as st_
    rho = st_.spearmanr(d["total_weight_g"], d["delivery_duration_days"], nan_policy="omit")[0]
    assert f"order weight with delivery time: rho = {rho:.2f}" in text_of(page_html(at))
    assert f"n = {int(d['total_weight_g'].notna().sum()):,}" in text_of(page_html(at))


@needs_data
def test_freight_ratio_late_correlation_uses_the_freight_ratio_column(orders):
    from scipy import stats as st_
    at = run_page()
    at.selectbox(key="da_wf_route").set_value("Same state (seller and customer)").run()
    d = orders[orders["same_state"] == 1]
    ratio = st_.spearmanr(d["freight_ratio"], d["is_late"], nan_policy="omit")[0]
    freight = st_.spearmanr(d["total_freight"], d["is_late"])[0]
    assert f"{ratio:.2f}" != f"{freight:.2f}"                          # the check is able to tell the two variables apart
    assert f"freight-to-price ratio with being late (0/1): rho = {ratio:.2f}" in text_of(page_html(at))


@needs_data
def test_scatter_overlay_is_drawn_only_when_switched_on():
    at = run_page()
    assert not any("Sampled order" in str(t.get("name")) for t in fig_of(at, "da_w_dur")["data"])
    assert not any("Sampled order" in str(t.get("name")) for t in fig_of(at, "da_f_dur")["data"])
    at.checkbox(key="da_wf_scatter").set_value(True).run()
    for key in ("da_w_dur", "da_f_dur"):
        names = [str(t.get("name")) for t in fig_of(at, key)["data"]]
        assert "Sampled order: on time" in names and "Sampled order: late" in names, key
    at.checkbox(key="da_wf_scatter").set_value(False).run()
    assert not any("Sampled order" in str(t.get("name")) for t in fig_of(at, "da_w_dur")["data"])


@needs_data
def test_min_orders_boundary_keeps_a_category_with_exactly_that_many_orders(orders):
    n = int((orders["product_category"] == "home_confort").sum())
    assert n % 10 == 0 and 10 <= n <= 1000                               # reachable with the slider (step 10)
    at = run_page(min_orders=n)
    t = frame_with(at, "Category")
    assert "Home confort" in set(t["Category"]) and (t["Orders (n)"] >= n).all() and t["Reliable"].all()
    sizes = orders["product_category"].value_counts()
    named = sizes[sizes.index != "unknown"]
    assert len(t) == int((named >= n).sum())
    at2 = run_page(min_orders=n + 10)
    assert "Home confort" not in set(frame_with(at2, "Category")["Category"])
    # the bar charts apply the same rule (>= n)
    assert any("Home confort" in str(y) for y in bar_labels(fig_of(at, "da_cat_late")))


@needs_data
def test_category_table_late_orders_column_and_sort_options(orders):
    at = run_page()
    t = frame_with(at, "Category")
    late = orders.groupby("product_category")["is_late"].agg(["size", "sum", "mean"])
    late.index = [ui.category_label(c) for c in late.index]
    for _, r in t.iterrows():
        assert r["Late orders"] == late.loc[r["Category"], "sum"] and r["Orders (n)"] == late.loc[r["Category"], "size"]
    named = orders[orders["product_category"] != "unknown"]
    g = named.groupby("product_category").agg(n=("is_late", "size"), days=("delivery_duration_days", "mean"),
                                              freight=("total_freight", "mean"))
    g = g[g["n"] >= 100]
    expected_first = {"Orders (most first)": g["n"].idxmax(), "Mean delivery time (longest first)": g["days"].idxmax(),
                      "Mean freight (highest first)": g["freight"].idxmax()}
    for option, code in expected_first.items():
        at.selectbox(key="da_cat_sort").set_value(option).run()
        t = frame_with(at, "Category")
        assert t.iloc[0]["Category"] == ui.category_label(code), option
        assert len(t) == len(g)
    at.selectbox(key="da_cat_sort").set_value("Orders (most first)").run()
    t = frame_with(at, "Category")
    assert list(t["Orders (n)"]) == sorted(t["Orders (n)"], reverse=True)
    at.selectbox(key="da_cat_sort").set_value("Category name").run()
    t = frame_with(at, "Category")
    assert list(t["Category"]) == [ui.category_label(c) for c in sorted(g.index)]


@needs_data
def test_csv_download_is_the_full_table_on_screen():
    from io import BytesIO
    from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage as Storage
    captured: list[bytes] = []
    original = Storage.load_and_get_id

    def spy(self, path_or_data, *args, **kwargs):
        if isinstance(path_or_data, bytes):
            captured.append(path_or_data)
        return original(self, path_or_data, *args, **kwargs)

    Storage.load_and_get_id = spy
    try:
        at = run_page()
        at.selectbox(key="da_cat_sort").set_value("Category name").run()
    finally:
        Storage.load_and_get_id = original
    t = frame_with(at, "Category")
    csv = pd.read_csv(BytesIO(captured[-1]))
    assert list(csv.columns) == list(t.columns) and len(csv) == len(t) > 20
    assert list(csv["Category"]) == list(t["Category"])                 # the download follows the chosen sort
    assert list(csv["Orders (n)"]) == list(t["Orders (n)"]) and list(csv["Late orders"]) == list(t["Late orders"])
    np.testing.assert_allclose(csv["Late rate (%)"], t["Late rate (%)"])


@needs_data
@pytest.mark.parametrize("stat,col", [("Mean", "mean"), ("Median", "median")])
@pytest.mark.parametrize("order", ["Longest", "Shortest"])
def test_duration_chart_respects_statistic_and_order_radios(orders, stat, col, order):
    g = orders[orders["product_category"] != "unknown"].groupby("product_category")["delivery_duration_days"].agg(["size", "mean", "median"])
    g = g[g["size"] >= 100][col].sort_values(ascending=(order == "Shortest"))
    at = run_page()
    at.radio(key="da_dur_stat").set_value(stat).run()
    at.radio(key="da_dur_order").set_value(order).run()
    assert not at.exception
    bar = next(t for t in fig_of(at, "da_dur_cat_chart")["data"] if t.get("type") == "bar")
    got = sorted(float(v) for v in dec(bar["x"]))
    assert len(got) == 15
    np.testing.assert_allclose(got, sorted(g.head(15).to_numpy()), rtol=1e-9)
    # the state chart and the sentence follow the same radios
    word = stat.lower()
    assert f"{word} delivery time runs from" in text_of(page_html(at))


@needs_data
def test_flagged_months_table_difference_is_in_percentage_points(orders):
    at = run_page()
    t = frame_with(at, "Adjusted p")
    assert len(t) > 0
    for _, r in t.iterrows():
        month = pd.Timestamp(r["Month"]).strftime("%Y-%m")
        in_m = orders["purchase_year_month"] == month
        rate, rest = orders.loc[in_m, "is_late"].mean(), orders.loc[~in_m, "is_late"].mean()
        assert r["Other months (%)"] == pytest.approx(100 * rest)
        assert r["Difference (pp)"] == pytest.approx(100 * (rate - rest))      # percentage points, not a fraction
        assert r["CI low (pp)"] < r["Difference (pp)"] < r["CI high (pp)"]
        assert r["Difference (pp)"] > 1                                        # flagged months are several points above the rest


@needs_data
def test_unknown_category_is_left_out_of_charts_even_when_it_ranks_in_the_top(orders):
    sp = orders[orders["customer_state"] == "SP"]
    cp = an.category_performance(sp, 100)
    cp = cp[cp["n_orders"] >= 100]
    top_late = cp.sort_values("late_rate", ascending=False)["product_category"].head(12).tolist()
    top_dur = cp.sort_values("mean_duration", ascending=False)["product_category"].head(15).tolist()
    assert "unknown" in top_late and "unknown" in top_dur                # it WOULD be drawn if it were not excluded
    at = run_page(states=["SP"])
    assert not at.exception
    for key in ("da_cat_late", "da_cat_dur", "da_dur_cat_chart"):
        labels = [str(y) for y in bar_labels(fig_of(at, key))]
        assert labels and not any("nknown" in x for x in labels), key
    # the table keeps it out by default, and shows it when asked
    assert "Unknown" not in set(frame_with(at, "Category")["Category"])
    at.toggle(key="da_cat_unknown").set_value(True).run()
    assert "Unknown" in set(frame_with(at, "Category")["Category"])
