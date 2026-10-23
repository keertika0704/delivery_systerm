"""Tests for the Regional Intelligence page (pages/3_Regional_Intelligence.py).

The page is run with ``streamlit.testing`` ``AppTest`` (standalone, so the sidebar filters are driven
through ``st.session_state``).  Displayed numbers are checked against an independent pandas
recomputation from ``orders_delivered.parquet``.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import analytics as an
from src import config, insights, ui

ROOT = Path(__file__).resolve().parents[1]
PAGE = str(ROOT / "pages" / "3_Regional_Intelligence.py")
TIMEOUT = 120

pytestmark = pytest.mark.skipif(not config.ORDERS_DELIVERED_PATH.is_file(),
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


@pytest.fixture
def downloads(monkeypatch):
    """Capture the data passed to st.download_button (AppTest only exposes a mock URL)."""
    captured: list[dict] = []
    original = st.download_button

    def spy(label, data=None, *args, **kwargs):
        captured.append({"label": label, "data": data, "key": kwargs.get("key"), "file_name": kwargs.get("file_name")})
        return original(label, data, *args, **kwargs)

    monkeypatch.setattr(st, "download_button", spy)
    return captured


def _run(states=None, categories=None, date_range=None) -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=TIMEOUT)
    if states is not None:
        at.session_state[ui.K_STATES] = states
    if categories is not None:
        at.session_state[ui.K_CATEGORIES] = categories
    if date_range is not None:
        at.session_state[ui.K_DATE] = date_range
    return at.run()


def _html(at: AppTest) -> str:
    return "\n".join([str(e.value) for e in at.get("html")] + [str(m.value) for m in at.markdown])


_TABLE_MARKERS = {"reg_table": "state_name", "reg_cmp_table": "assessment", "reg_high_table": "holds",
                  "reg_small_table": "State", "reg_route_table": "route"}


def _table(at: AppTest, key: str) -> pd.DataFrame | None:
    """The st.dataframe whose columns carry the marker of ``key`` (AppTest does not expose dataframe keys)."""
    for d in at.dataframe:
        if _TABLE_MARKERS[key] in d.value.columns:
            return d.value
    return None


def _state_perf(df: pd.DataFrame) -> pd.DataFrame:
    """Independent pandas recomputation of the per-state numbers."""
    g = df.groupby("customer_state", observed=True)
    out = pd.DataFrame({"n": g.size(), "late": g["is_late"].sum(), "dur": g["delivery_duration_days"].mean()})
    out["rate"] = out["late"] / out["n"]
    out.index = out.index.astype(str)
    return out


# --------------------------------------------------------------------------- #
# Default view
# --------------------------------------------------------------------------- #
def test_page_runs_and_has_all_sections():
    at = _run()
    assert not at.exception, [e.value for e in at.exception]
    assert [t.label for t in at.tabs] == ["Map & performance table", "Compare states", "High-delay regions",
                                         "Seller → customer routes"]
    html = _html(at)
    assert "Regional Intelligence" in html
    # one chart per section is rendered (map, comparison, high-delay bar, route bar, heat map)
    assert len(at.get("plotly_chart")) >= 5
    assert at.slider(key="reg_min_orders").value == config.DEFAULT_MIN_ORDERS
    assert at.radio(key="reg_map_metric").value == "Late rate"
    # clearly separated interpretation callouts
    for label in ("Observed", "Statistical association", "Suggested action", "Sample size"):
        assert f">{label}</div>" in html


def test_headline_numbers_match_pandas(orders):
    at = _run()
    html = _html(at)
    perf = _state_perf(orders)
    overall = orders["is_late"].mean()
    assert ui.fmt_pct(overall) in html
    assert f"{int(orders['is_late'].sum()):,} late of {len(orders):,} orders" in html
    rel = perf[perf["n"] >= config.DEFAULT_MIN_ORDERS]
    top = rel["rate"].idxmax()
    assert ui.fmt_pct(rel.loc[top, "rate"]) in html
    assert ui.state_label(top) in html and f"n = {int(rel.loc[top, 'n']):,}" in html
    assert f"{len(perf)}" in html and f"{len(rel)} with at least" in html


def test_regional_table_matches_pandas(orders):
    at = _run()
    tbl = _table(at, "reg_table").set_index("customer_state")
    perf = _state_perf(orders)
    assert set(tbl.index) == set(perf.index) and len(tbl) == orders["customer_state"].nunique()
    pd.testing.assert_series_equal(tbl["n_orders"].astype("int64").sort_index(), perf["n"].astype("int64").sort_index(),
                                   check_names=False)
    np.testing.assert_allclose(tbl["late_rate"].sort_index(), perf["rate"].sort_index(), rtol=1e-9)
    np.testing.assert_allclose(tbl["mean_duration"].sort_index(), perf["dur"].sort_index(), rtol=1e-9)
    np.testing.assert_allclose(tbl["on_time_rate"].sort_index(), (1 - perf["rate"]).sort_index(), rtol=1e-9)
    # reliable states first (descending late rate), insufficient-data states last
    flags = tbl["flag"].tolist()
    first_small = next(i for i, f in enumerate(flags) if f.startswith("Insufficient"))
    assert all(f.startswith("Insufficient") for f in flags[first_small:])
    rel_rates = tbl["late_rate"].iloc[:first_small].tolist()
    assert rel_rates == sorted(rel_rates, reverse=True)
    small_codes = set(tbl.index[first_small:])
    assert small_codes == set(perf.index[perf["n"] < config.DEFAULT_MIN_ORDERS])


def test_flagged_states_agree_between_report_and_high_delay_screen(orders):
    """'Above overall' in the table and 'high-delay segment' use the same rule."""
    overall = an.kpi_summary(orders)["late_rate"]
    perf = an.state_performance(orders, config.DEFAULT_MIN_ORDERS)
    high = an.high_delay_segments(perf, config.DEFAULT_MIN_ORDERS, overall)
    rep = insights.regional_report(orders, config.DEFAULT_MIN_ORDERS)
    above = set(rep.loc[rep["flag"].str.startswith("Above"), "customer_state"])
    assert above == set(high["customer_state"])
    at = _run()
    shown = _table(at, "reg_high_table")
    assert set(shown["customer_state"]) == above
    assert len(shown) == len(high)
    assert (shown["excess_pp"] > 0).all() and (shown["rate_ratio"] > 1).all()
    assert shown["holds"].isin(["Yes", "No"]).all()


# --------------------------------------------------------------------------- #
# Minimum-sample slider
# --------------------------------------------------------------------------- #
def test_min_sample_slider_changes_flags_and_insufficient_list(orders):
    at = _run()
    at.slider(key="reg_min_orders").set_value(1000).run()
    assert not at.exception, [e.value for e in at.exception]
    perf = _state_perf(orders)
    overall = orders["is_late"].mean()
    full = an.state_performance(orders, 1000)
    expected = set(an.high_delay_segments(full, 1000, overall)["customer_state"])
    shown = _table(at, "reg_high_table")
    got = set(shown["customer_state"]) if shown is not None else set()
    assert got == expected
    small = _table(at, "reg_small_table")
    assert set(small["State"]) == set(perf.index[perf["n"] < 1000])
    # insufficient-data states never appear as findings and their rates are not listed
    assert not (set(small["State"]) & got)
    assert "late_rate" not in small.columns and "Late rate" not in small.columns
    reg = _table(at, "reg_table")
    assert reg.loc[reg["customer_state"].isin(small["State"]), "flag"].str.startswith("Insufficient").all()
    assert "multiple" in _html(at).lower() or "comparisons" in _html(at).lower()   # unadjusted-interval caveat


def test_min_sample_above_every_state_shows_friendly_state(orders):
    at = _run(states=["AC"])           # one tiny state: 80 orders < default 100
    assert not at.exception, [e.value for e in at.exception]
    html = _html(at)
    assert "No state meets the minimum sample" in html
    assert "no state has 100+ orders" in html
    assert _table(at, "reg_high_table") is None


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #
def test_state_filter_matches_pandas(orders):
    at = _run(states=["SP"])
    assert not at.exception, [e.value for e in at.exception]
    sp = orders[orders["customer_state"] == "SP"]
    tbl = _table(at, "reg_table")
    assert list(tbl["customer_state"]) == ["SP"]
    assert int(tbl["n_orders"].iloc[0]) == len(sp)
    assert tbl["late_rate"].iloc[0] == pytest.approx(sp["is_late"].mean())
    assert ui.fmt_pct(sp["is_late"].mean()) in _html(at)
    # route table: only SP customers, volume equals pandas
    routes = _table(at, "reg_route_table")
    assert routes["route"].str.endswith("→SP").all()
    n_sp_sp = int(((sp["seller_state"] == "SP")).sum())
    assert int(routes.loc[routes["route"] == "SP→SP", "n_orders"].iloc[0]) == n_sp_sp


def test_state_and_category_filter_matches_pandas(orders):
    cat = orders["product_category"].value_counts().index[0]
    at = _run(states=["RJ", "MG"], categories=[cat])
    assert not at.exception, [e.value for e in at.exception]
    sub = orders[orders["customer_state"].isin(["RJ", "MG"]) & (orders["product_category"] == cat)]
    tbl = _table(at, "reg_table").set_index("customer_state")
    perf = _state_perf(sub)
    assert set(tbl.index) == set(perf.index)
    for code in perf.index:
        assert int(tbl.loc[code, "n_orders"]) == int(perf.loc[code, "n"])
        assert tbl.loc[code, "late_rate"] == pytest.approx(perf.loc[code, "rate"])
    assert f"{len(sub):,}" in _html(at)


def test_date_filter_matches_pandas(orders):
    rng = (pd.Timestamp("2017-03-01").date(), pd.Timestamp("2017-09-30").date())
    at = _run(date_range=rng)
    assert not at.exception
    sub = orders[(orders["purchase_date"] >= pd.Timestamp(rng[0])) & (orders["purchase_date"] <= pd.Timestamp(rng[1]))]
    tbl = _table(at, "reg_table")
    assert int(tbl["n_orders"].sum()) == len(sub)


def test_empty_filter_result_is_friendly_on_all_tabs(orders):
    ac_cats = set(orders.loc[orders["customer_state"] == "AC", "product_category"])
    cat = next(c for c in orders["product_category"].value_counts().index if c not in ac_cats)
    at = _run(states=["AC"], categories=[cat])
    assert not at.exception, [e.value for e in at.exception]
    html = _html(at)
    assert "No orders match the current filters" in html
    assert len(at.tabs) == 0 and len(at.get("plotly_chart")) == 0 and len(at.dataframe) == 0


def test_tiny_sample_view_never_raises(orders):
    """A view with a single small state: tables / charts / look-up all cope."""
    state = "AP"
    at = _run(states=[state])
    assert not at.exception, [e.value for e in at.exception]
    html = _html(at)
    assert "Insufficient data" in html or "insufficient" in html.lower()
    tbl = _table(at, "reg_table")
    assert tbl["flag"].str.startswith("Insufficient").all()


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def test_csv_download_equals_table_shown(orders, downloads):
    at = _run()
    assert not at.exception
    dl = [d for d in downloads if d["key"] == "reg_download"][-1]
    assert dl["file_name"].endswith(".csv")
    csv = pd.read_csv(io.BytesIO(dl["data"]))
    tbl = _table(at, "reg_table")
    assert len(csv) == len(tbl)
    assert list(csv["customer_state"]) == list(tbl["customer_state"])
    np.testing.assert_allclose(csv["late_rate"], tbl["late_rate"])
    np.testing.assert_array_equal(csv["n_orders"], tbl["n_orders"])
    assert [f.split(" (")[0].replace("Interval includes overall", "Overlaps overall") for f in csv["flag"]] == \
        [f.replace("Insufficient data", "Insufficient data") for f in tbl["flag"]]
    # and equals the insights report for the same view / threshold (same rows, different order only)
    rep = insights.regional_report(orders, config.DEFAULT_MIN_ORDERS).set_index("customer_state")
    c2 = csv.set_index("customer_state")
    assert set(c2.index) == set(rep.index)
    for col in ("n_orders", "n_late", "late_rate", "late_ci_low", "late_ci_high", "mean_duration", "flag"):
        pd.testing.assert_series_equal(c2[col].sort_index(), rep[col].sort_index(), check_dtype=False, check_names=False)
    assert not {"order_id", "customer_id", "seller_id"} & set(csv.columns)


def test_csv_follows_filters_threshold_and_search(orders, downloads):
    at = _run(states=["SP", "RJ", "AC"])
    at.slider(key="reg_min_orders").set_value(200).run()
    csv = pd.read_csv(io.BytesIO([d for d in downloads if d["key"] == "reg_download"][-1]["data"]))
    sub = orders[orders["customer_state"].isin(["SP", "RJ", "AC"])]
    rep = insights.regional_report(sub, 200).set_index("customer_state")
    c = csv.set_index("customer_state")
    assert set(c.index) == {"SP", "RJ", "AC"}
    assert c.loc["AC", "flag"].startswith("Insufficient data (n < 200)")
    assert c.loc["SP", "n_orders"] == rep.loc["SP", "n_orders"]
    assert (c["overall_late_rate"].round(12) == round(sub["is_late"].mean(), 12)).all()
    # text search narrows both the table and the CSV
    at.text_input(key="reg_table_search").set_value("rio").run()
    csv2 = pd.read_csv(io.BytesIO([d for d in downloads if d["key"] == "reg_download"][-1]["data"]))
    assert list(csv2["customer_state"]) == ["RJ"]
    assert list(_table(at, "reg_table")["customer_state"]) == ["RJ"]


def test_search_without_match_shows_empty_state_and_disables_download(orders):
    at = _run()
    at.text_input(key="reg_table_search").set_value("zzz-not-a-state").run()
    assert not at.exception
    assert "No state matches this search" in _html(at)
    assert at.get_by_key("reg_download").proto.disabled is True


def test_hide_insufficient_checkbox(orders):
    at = _run()
    at.checkbox(key="reg_table_hide_small").check().run()
    tbl = _table(at, "reg_table")
    assert not tbl["flag"].str.startswith("Insufficient").any()
    perf = _state_perf(orders)
    assert len(tbl) == int((perf["n"] >= config.DEFAULT_MIN_ORDERS).sum())


# --------------------------------------------------------------------------- #
# Map metric + comparison
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("metric", ["Late rate", "Mean delivery time", "Order volume"])
def test_map_metrics_render_with_matching_text(orders, metric):
    at = _run()
    at.radio(key="reg_map_metric").set_value(metric).run()
    assert not at.exception, [e.value for e in at.exception]
    html = _html(at)
    perf = _state_perf(orders)
    rel = perf[perf["n"] >= config.DEFAULT_MIN_ORDERS]
    if metric == "Mean delivery time":
        assert ui.fmt_days(rel["dur"].max()) in html and ui.fmt_days(rel["dur"].min()) in html
    elif metric == "Order volume":
        top = perf["n"].idxmax()
        assert ui.fmt_int(perf.loc[top, "n"]) in html and ui.state_label(top) in html
    else:
        assert ui.fmt_pct(rel["rate"].max()) in html


def test_compare_default_selection_and_numbers(orders):
    at = _run()
    sel = at.multiselect(key="reg_compare_states").value
    perf = _state_perf(orders)
    rel = perf[perf["n"] >= config.DEFAULT_MIN_ORDERS]
    top3 = perf["n"].sort_values(ascending=False).index[:3].tolist()
    worst2 = rel["rate"].sort_values(ascending=False).index[:2].tolist()
    assert sel == list(dict.fromkeys(top3 + worst2))
    cmp_tbl = _table(at, "reg_cmp_table").set_index("customer_state")
    for code in sel:
        assert int(cmp_tbl.loc[code, "n_orders"]) == int(perf.loc[code, "n"])
        assert cmp_tbl.loc[code, "late_rate"] == pytest.approx(perf.loc[code, "rate"])
        assert cmp_tbl.loc[code, "mean_duration"] == pytest.approx(perf.loc[code, "dur"])


def test_compare_selection_edge_cases(orders):
    at = _run()
    at.multiselect(key="reg_compare_states").set_value([]).run()
    assert not at.exception
    assert "Select at least one state to compare" in _html(at)
    at.multiselect(key="reg_compare_states").set_value(["SP"]).run()
    assert not at.exception
    assert "Select a second state" in _html(at)
    at.multiselect(key="reg_compare_states").set_value(["SP", "RJ"]).run()
    assert not at.exception
    html = _html(at)
    perf = _state_perf(orders)
    gap = (perf.loc["RJ", "rate"] - perf.loc["SP", "rate"])
    assert ui.fmt_pp(gap) in html
    assert "two-proportion z-test" in html
    # a tiny state is shown but not compared statistically
    at.multiselect(key="reg_compare_states").set_value(["SP", "AC"]).run()
    assert not at.exception
    assert "Insufficient data" in _html(at)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
def test_route_table_matches_pandas_and_filters(orders):
    at = _run()
    routes = _table(at, "reg_route_table").set_index("route")
    pairs = orders.groupby(["seller_state", "customer_state"], observed=True).agg(
        n=("is_late", "size"), late=("is_late", "mean"), dur=("delivery_duration_days", "mean"))
    pairs.index = [f"{a}→{b}" for a, b in pairs.index]
    big = pairs[pairs["n"] >= config.DEFAULT_MIN_ORDERS]
    assert set(routes.index) == set(big.index)
    for r in ("SP→SP", "SP→RJ"):
        assert int(routes.loc[r, "n_orders"]) == int(pairs.loc[r, "n"])
        assert routes.loc[r, "late_rate"] == pytest.approx(pairs.loc[r, "late"])
        assert routes.loc[r, "mean_duration"] == pytest.approx(pairs.loc[r, "dur"])
    # filter by seller state and customer state
    at.multiselect(key="reg_pair_seller").set_value(["MG"]).run()
    assert not at.exception
    assert _table(at, "reg_route_table")["route"].str.startswith("MG→").all()
    at.multiselect(key="reg_pair_seller").set_value([]).run()
    at.multiselect(key="reg_pair_customer").set_value(["RJ"]).run()
    assert _table(at, "reg_route_table")["route"].str.endswith("→RJ").all()
    # untick the minimum-sample box: small routes appear and are labelled
    at.checkbox(key="reg_pair_only_reliable").uncheck().run()
    all_rj = _table(at, "reg_route_table")
    assert len(all_rj) == int((orders["customer_state"] == "RJ").groupby(orders["seller_state"]).sum().gt(0).sum())
    assert all_rj["status"].str.startswith("Insufficient").any()


def test_route_filters_with_no_match_show_empty_state():
    at = _run()
    at.multiselect(key="reg_pair_seller").set_value(["AC"]).run()      # no seller in AC ships to...
    at.multiselect(key="reg_pair_customer").set_value(["AM"]).run()
    assert not at.exception
    assert "No routes match these filters" in _html(at)


def test_route_lookup_single_pair(orders):
    at = _run()
    at.selectbox(key="reg_lookup_seller").set_value("SP")
    at.selectbox(key="reg_lookup_customer").set_value("RJ").run()
    assert not at.exception
    sub = orders[(orders["seller_state"] == "SP") & (orders["customer_state"] == "RJ")]
    html = _html(at)
    assert ui.fmt_int(len(sub)) in html
    assert ui.fmt_days(sub["delivery_duration_days"].mean()) in html
    assert ui.fmt_pct(sub["is_late"].mean()) in html
    assert f"{int(sub['is_late'].sum()):,} of {len(sub):,} orders were late" in html


def test_route_lookup_pair_without_orders_and_small_pair(orders):
    pair_n = orders.groupby(["seller_state", "customer_state"], observed=True).size()
    sellers = sorted(orders["seller_state"].unique())
    custs = sorted(orders["customer_state"].unique())
    missing = next(((s, c) for s in sellers for c in custs if (s, c) not in pair_n.index), None)
    assert missing is not None
    at = _run()
    at.selectbox(key="reg_lookup_seller").set_value(missing[0])
    at.selectbox(key="reg_lookup_customer").set_value(missing[1]).run()
    assert not at.exception
    assert f"No orders on {missing[0]} → {missing[1]}" in _html(at)
    small = pair_n[(pair_n >= 5) & (pair_n < config.DEFAULT_MIN_ORDERS)].index[0]
    at.selectbox(key="reg_lookup_seller").set_value(small[0])
    at.selectbox(key="reg_lookup_customer").set_value(small[1]).run()
    assert not at.exception
    assert "below the 100-order minimum" in _html(at)


def test_same_vs_cross_state_numbers(orders):
    at = _run()
    html = _html(at)
    same = orders[orders["same_state"] == 1]
    cross = orders[orders["same_state"] == 0]
    assert ui.fmt_pct(same["is_late"].mean()) in html and ui.fmt_pct(cross["is_late"].mean()) in html
    assert f"n = {len(same):,}" in html and f"n = {len(cross):,}" in html
    assert ui.fmt_days(same["delivery_duration_days"].mean()) in html
    multi = (orders["n_sellers"] > 1).mean()
    assert ui.fmt_pct(multi, 2) in html and "representative seller" in html


@pytest.mark.parametrize("metric", ["Late rate", "Order volume", "Mean delivery time"])
def test_route_chart_and_heatmap_options_render(metric):
    at = _run()
    at.radio(key="reg_route_metric").set_value(metric)
    at.radio(key="reg_heat_metric").set_value(metric).run()
    assert not at.exception, [e.value for e in at.exception]
    at.selectbox(key="reg_heat_top").set_value(27).run()
    assert not at.exception, [e.value for e in at.exception]


def test_no_deprecation_warnings():
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        at = _run()
    bad = [str(w.message) for w in caught if issubclass(w.category, (DeprecationWarning, FutureWarning))
           and ("streamlit" in str(w.filename) or "pages" in str(w.filename) or "src" in str(w.filename))]
    assert not at.exception and bad == []


# =========================================================================== #
# Strict value tests (each assertion targets one on-screen number or conclusion and is checked
# against an independent recomputation, so a wrong value cannot hide behind the same number
# appearing elsewhere on the page)
# =========================================================================== #
import base64  # noqa: E402
import html as _htmllib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402

import plotly.graph_objects as go  # noqa: E402
from scipy import stats as _st  # noqa: E402

_Z95 = _st.norm.ppf(0.975)


def _wilson(k, n):
    p = k / n
    d = 1 + _Z95 ** 2 / n
    c = (p + _Z95 ** 2 / (2 * n)) / d
    h = _Z95 * np.sqrt(p * (1 - p) / n + _Z95 ** 2 / (4 * n * n)) / d
    return c - h, c + h


def _holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    out = np.empty_like(p)
    run, m = 0.0, len(p)
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        out[i] = min(1.0, run)
    return out


def _txt(fragment: str) -> str:
    return _htmllib.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


def _kpis(at: AppTest) -> list[dict]:
    """Every KPI card on the page in document order: label, value, sub, delta."""
    cards = []
    for h in at.get("html"):
        for c in str(h.value).split('<div class="diq-kpi" ')[1:]:
            g = lambda cls: (lambda r: _txt(r.group(1)) if r else None)(
                re.search(rf'class="{cls}[^"]*">(.*?)</(?:div|span)>', c, re.S))
            cards.append({"label": g("diq-kpi-label"), "value": g("diq-kpi-value"), "sub": g("diq-kpi-sub"),
                          "delta": g("diq-kpi-delta")})
    return cards


def _kpi(at: AppTest, label_start: str) -> dict:
    hits = [k for k in _kpis(at) if (k["label"] or "").startswith(label_start)]
    assert len(hits) == 1, (label_start, [k["label"] for k in _kpis(at)])
    return hits[0]


def _callouts(at: AppTest) -> list[tuple[str, str]]:
    """(title, text) of every callout box in document order."""
    out = []
    for h in at.get("html"):
        for m in re.finditer(r'<div class="diq-callout [a-z]+".*?diq-callout-title">(.*?)</div>'
                             r'(?:<div class="diq-callout-text">(.*?)</div>)?', str(h.value), re.S):
            out.append((_txt(m.group(1)), _txt(m.group(2) or "")))
    return out


def _callout(at: AppTest, title: str, contains: str | None = None) -> str:
    hits = [t for ti, t in _callouts(at) if ti == title and (contains is None or contains in t)]
    assert hits, (title, contains, [ti for ti, _ in _callouts(at)])
    return hits[0]


def _spec(at: AppTest, key: str) -> dict:
    (c,) = [c for c in at.get("plotly_chart") if c.key == key]
    return json.loads(c.proto.spec)


def _arr(v) -> np.ndarray:
    """Plotly 6+ ships numeric arrays as {dtype, bdata, shape}; decode them."""
    if isinstance(v, dict) and "bdata" in v:
        a = np.frombuffer(base64.b64decode(v["bdata"]), dtype=np.dtype(v["dtype"]))
        if "shape" in v:
            a = a.reshape([int(x) for x in str(v["shape"]).split(",")])
        return a.astype(float)
    return np.asarray(v, dtype=float)


def _bar_labels(at: AppTest, key: str) -> list[str]:
    """Route labels ('SP→RJ') of every bar in the chart (bars are split into several traces by colour)."""
    spec = _spec(at, key)
    return [str(v).split()[0] for d in spec["data"] if d["type"] == "bar" for v in d["y"]]


def _route_table_pairs(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["seller_state", "customer_state"], observed=True)["is_late"].agg(["size", "sum"])
    g.columns = ["n", "late"]
    g["rate"] = g["late"] / g["n"]
    lo_hi = [_wilson(k, n) for k, n in zip(g["late"], g["n"])]
    g["lo"], g["hi"] = [a for a, _ in lo_hi], [b for _, b in lo_hi]
    return g


def test_kpi_cards_by_label_match_pandas(orders):
    at = _run()
    perf = _state_perf(orders)
    overall = orders["is_late"].mean()
    rel = perf[perf["n"] >= config.DEFAULT_MIN_ORDERS]
    k = _kpi(at, "Overall late rate")
    assert k["value"] == ui.fmt_pct(overall)
    assert k["value"] != ui.fmt_pct(1 - overall)
    assert k["sub"] == f"{int(orders['is_late'].sum()):,} late of {len(orders):,} orders"
    top = rel["rate"].idxmax()
    k = _kpi(at, "Highest observed late rate")
    assert k["value"] == ui.fmt_pct(rel.loc[top, "rate"])
    assert k["sub"].startswith(ui.state_label(top)) and f"n = {int(rel.loc[top, 'n']):,}" in k["sub"]
    assert k["delta"] == ui.fmt_pp(rel.loc[top, "rate"] - overall) + " vs overall"
    assert k["delta"].startswith("+")
    k = _kpi(at, "Customer states in view")
    assert k["value"] == str(len(perf)) and k["sub"].startswith(f"{len(rel)} with at least")
    # independent Wilson screening of the high-delay segments
    n_hi = sum(1 for c in rel.index
               if rel.loc[c, "rate"] > overall and _wilson(rel.loc[c, "late"], rel.loc[c, "n"])[0] > overall)
    k = _kpi(at, "High-delay segments")
    assert k["value"] == str(n_hi) and f"of {len(rel)} states screened" in k["sub"]


def _expected_holm(orders: pd.DataFrame, min_orders: int) -> dict:
    """Independent Holm-adjusted p of every state with at least ``min_orders`` orders (state vs all other orders)."""
    perf = _state_perf(orders)
    rel = perf[perf["n"] >= min_orders]
    n_all, k_all = len(orders), int(orders["is_late"].sum())
    pv = {}
    for c, r in rel.iterrows():
        k1, n1, k2, n2 = int(r["late"]), int(r["n"]), k_all - int(r["late"]), n_all - int(r["n"])
        pool = (k1 + k2) / (n1 + n2)
        z = (k1 / n1 - k2 / n2) / np.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
        pv[c] = 2 * _st.norm.sf(abs(z))
    return dict(zip(pv, _holm(list(pv.values()))))


def _check_holm_table(at: AppTest, orders: pd.DataFrame, min_orders: int) -> pd.DataFrame:
    adj = _expected_holm(orders, min_orders)
    tbl = _table(at, "reg_high_table").set_index("customer_state")
    for c, row in tbl.iterrows():
        exp = adj[c]
        exp_txt = "< 0.001" if exp < 0.001 else f"{exp:.3g}"
        assert row["p_holm_txt"] == exp_txt, (c, row["p_holm_txt"], exp)
        assert row["holds"] == ("Yes" if exp < config.DEFAULT_ALPHA else "No"), c
    return tbl


def test_high_delay_tab_holm_values_and_survivor_count(orders):
    """Holm-adjusted p and the 'Survives Holm?' column against an independent Holm over the screened states."""
    at = _run()
    tbl = _check_holm_table(at, orders, config.DEFAULT_MIN_ORDERS)
    assert len(tbl) > 3 and tbl["p_holm_txt"].nunique() > 1
    n_yes = int((tbl["holds"] == "Yes").sum())
    text = _callout(at, "Statistical association", "after Holm")
    assert text.startswith(f"{n_yes} of {len(tbl)} flagged states remain significant")


def test_high_delay_text_numbers(orders):
    at = _run()
    perf = _state_perf(orders)
    overall = orders["is_late"].mean()
    rel = perf[perf["n"] >= config.DEFAULT_MIN_ORDERS].copy()
    rel["lo"] = [_wilson(r.late, r.n)[0] for r in rel.itertuples()]
    flagged = rel[(rel["rate"] > overall) & (rel["lo"] > overall)].sort_values("rate", ascending=False)
    top = flagged.index[0]
    text = _callout(at, "Observed", "flagged as high-delay")
    assert text.startswith(f"{len(flagged)} states are flagged")
    assert f"{ui.state_label(top)} at {ui.fmt_pct(flagged.loc[top, 'rate'])}" in text
    assert f"{(flagged.loc[top, 'rate'] - overall) * 100:+.1f} pp above overall ({flagged.loc[top, 'rate'] / overall:.1f}x)" in text
    share = flagged["n"].sum() / len(orders)
    assert f"{int(flagged['n'].sum()):,} orders ({ui.fmt_pct(share)} of the view)" in text
    # the map's observed callout quotes the same ratio, not its inverse
    map_text = _callout(at, "Observed", "highest observed late rate")
    assert f"({flagged.loc[top, 'rate'] / overall:.1f}x the overall rate)" in map_text
    assert f"{ui.fmt_pp(flagged.loc[top, 'rate'] - overall)} versus the overall" in map_text
    lowest = rel["rate"].idxmin()
    assert f"The lowest is {ui.state_label(lowest)} at {ui.fmt_pct(rel.loc[lowest, 'rate'])}" in map_text
    assoc = _callout(at, "Statistical association", "Wilson interval entirely above")
    assert assoc.startswith(f"{len(flagged)} of {len(rel)} states have a 95% Wilson interval entirely above")


def test_same_and_cross_state_kpis_are_not_swapped(orders):
    at = _run()
    same = orders[orders["same_state"] == 1]
    cross = orders[orders["same_state"] == 0]
    ks, kc = _kpi(at, "Same-state orders"), _kpi(at, "Cross-state orders")
    assert ks["value"] == ui.fmt_pct(same["is_late"].mean()) and f"n = {len(same):,}" in ks["sub"]
    assert kc["value"] == ui.fmt_pct(cross["is_late"].mean()) and f"n = {len(cross):,}" in kc["sub"]
    assert ui.fmt_days(same["delivery_duration_days"].mean()) in ks["sub"]
    assert ui.fmt_days(cross["delivery_duration_days"].mean()) in kc["sub"]
    gap = _kpi(at, "Gap (cross minus same)")
    assert gap["value"] == ui.fmt_pp(cross["is_late"].mean() - same["is_late"].mean())
    assert gap["value"].startswith("+") and "longer on average" in gap["sub"]
    text = _callout(at, "Observed", "shipped across state lines")
    assert text.startswith(f"Orders shipped across state lines are late {ui.fmt_pct(cross['is_late'].mean())}")
    assert f"versus {ui.fmt_pct(same['is_late'].mean())} for same-state orders" in text


def test_compare_table_vs_overall_sign_and_assessment(orders):
    at = _run(states=["SP", "RJ", "AL", "AC", "AM", "GO"])
    at.multiselect(key="reg_compare_states").set_value(["SP", "RJ", "AL", "AC", "GO", "AM"]).run()
    assert not at.exception
    overall = orders["is_late"].mean()
    perf = _state_perf(orders[orders["customer_state"].isin(["SP", "RJ", "AL", "AC", "AM", "GO"])])
    # "overall" in this view is the overall rate of the FILTERED orders
    view_overall = orders[orders["customer_state"].isin(["SP", "RJ", "AL", "AC", "AM", "GO"])]["is_late"].mean()
    cmp_tbl = _table(at, "reg_cmp_table").set_index("customer_state")
    assert list(cmp_tbl.index) == ["SP", "RJ", "AL", "AC", "GO", "AM"]            # selection order kept
    for c, row in cmp_tbl.iterrows():
        assert row["vs_overall_pp"] == pytest.approx((perf.loc[c, "rate"] - view_overall) * 100)
        n, k = int(perf.loc[c, "n"]), int(perf.loc[c, "late"])
        lo, hi = _wilson(k, n)
        if n < config.DEFAULT_MIN_ORDERS:
            exp = f"Insufficient data (n < {config.DEFAULT_MIN_ORDERS})"
        else:
            exp = "Above overall" if lo > view_overall else "Below overall" if hi < view_overall else "Overlaps overall"
        assert row["assessment"] == exp, c
    assert overall != view_overall


def test_compare_default_excludes_small_states_from_the_worst_picks(orders):
    """RR (n=41) has a higher late rate than RJ but is below the minimum: it must not be a default pick."""
    perf = _state_perf(orders)
    assert perf.loc["RR", "n"] < config.DEFAULT_MIN_ORDERS and perf.loc["RR", "rate"] > perf.loc["RJ", "rate"]
    at = _run(states=["SP", "MG", "RJ", "PR", "RR"])
    assert at.multiselect(key="reg_compare_states").value == ["SP", "RJ", "MG"]


def test_compare_callout_extremes(orders):
    at = _run(states=["SP", "RJ", "AL"])
    at.multiselect(key="reg_compare_states").set_value(["SP", "RJ", "AL"]).run()
    perf = _state_perf(orders)
    text = _callout(at, "Observed", "ranges from")
    assert f"ranges from {ui.fmt_pct(perf.loc['SP', 'rate'])} in {ui.state_label('SP')}" in text
    assert f"to {ui.fmt_pct(perf.loc['AL', 'rate'])} in {ui.state_label('AL')}" in text
    assert f"a gap of {ui.fmt_pp(perf.loc['AL', 'rate'] - perf.loc['SP', 'rate'])}" in text
    assoc = _callout(at, "Statistical association", "Highest vs lowest")
    assert f"difference {ui.fmt_pp(perf.loc['AL', 'rate'] - perf.loc['SP', 'rate'])}" in assoc
    assert "do not overlap" in assoc


def test_lookup_verdict_above_below_overlapping(orders):
    pairs = _route_table_pairs(orders)
    overall = orders["is_late"].mean()
    big = pairs[pairs["n"] >= config.DEFAULT_MIN_ORDERS]
    cases = {"entirely above": big[big["lo"] > overall], "entirely below": big[big["hi"] < overall],
             "overlapping": big[(big["lo"] <= overall) & (big["hi"] >= overall)]}
    at = _run()
    for verdict, frame in cases.items():
        assert len(frame), verdict
        s, c = frame["n"].idxmax()
        at.selectbox(key="reg_lookup_seller").set_value(s)
        at.selectbox(key="reg_lookup_customer").set_value(c).run()
        assert not at.exception
        text = _callout(at, "Observed", f"On {s} → {c}")
        assert f"the interval is {verdict} the overall rate of {ui.fmt_pct(overall)}" in text, (s, c, text)
        assert f"{int(frame.loc[(s, c), 'late']):,} of {int(frame.loc[(s, c), 'n']):,} orders were late" in text
        k = _kpi(at, "Late rate")
        assert k["delta"] == ui.fmt_pp(frame.loc[(s, c), "rate"] - overall) + " vs overall"


def test_sample_size_callout_reflects_the_minimum(orders):
    perf = _state_perf(orders)
    at = _run()
    small = perf[perf["n"] < config.DEFAULT_MIN_ORDERS]
    text = _callout(at, "Sample size", "below the")
    assert text.startswith(f"{len(small)} states below the {config.DEFAULT_MIN_ORDERS}-order minimum")
    assert all(f"{c} (n = {int(small.loc[c, 'n']):,})" in text for c in small.index)
    at.slider(key="reg_min_orders").set_value(10).run()          # every state has at least 10 orders
    assert int(perf["n"].min()) >= 10
    assert _callout(at, "Sample size", "Every state in view has at least 10 orders")
    assert not [t for ti, t in _callouts(at) if ti == "Sample size" and "below the" in t]


def test_state_with_exactly_the_minimum_is_ranked_not_small(orders):
    perf = _state_perf(orders)
    n_ac = int(perf.loc["AC", "n"])
    assert n_ac % 10 == 0 and 10 <= n_ac <= 1000, "pick another boundary state"
    at = _run()
    at.slider(key="reg_min_orders").set_value(n_ac).run()
    reg = _table(at, "reg_table").set_index("customer_state")
    assert not reg.loc["AC", "flag"].startswith("Insufficient")             # n == minimum counts as enough
    assert reg.loc["AP", "flag"].startswith("Insufficient")                 # n < minimum does not
    assert "AC" not in set(_table(at, "reg_small_table")["State"])
    k = _kpi(at, "Customer states in view")
    assert k["sub"].startswith(f"{int((perf['n'] >= n_ac).sum())} with at least {n_ac}")
    _check_holm_table(at, orders, n_ac)           # the Holm family includes the state with n == minimum
    at.slider(key="reg_min_orders").set_value(n_ac + 10).run()
    assert _table(at, "reg_table").set_index("customer_state").loc["AC", "flag"].startswith("Insufficient")
    assert "AC" in set(_table(at, "reg_small_table")["State"])


def test_heatmap_masks_cells_below_the_minimum(orders):
    pairs = _route_table_pairs(orders)
    at = _run()
    at.selectbox(key="reg_heat_top").set_value(27).run()
    at.slider(key="reg_min_orders").set_value(500).run()
    z = _arr(_spec(at, "reg_heat_chart")["data"][0]["z"])
    assert np.isfinite(z).sum() == int((pairs["n"] >= 500).sum())
    at.slider(key="reg_min_orders").set_value(100).run()
    f = _spec(at, "reg_heat_chart")["data"][0]
    z = _arr(f["z"])
    assert np.isfinite(z).sum() == int((pairs["n"] >= 100).sum())
    # the late-rate values themselves are the pandas rates
    for yi, s in enumerate(f["y"]):
        for xi, c in enumerate(f["x"]):
            if np.isfinite(z[yi, xi]):
                assert z[yi, xi] == pytest.approx(pairs.loc[(s, c), "rate"])


def test_route_bar_follows_slider_and_route_filters(orders):
    pairs = _route_table_pairs(orders)
    at = _run()
    ranked = pairs[pairs["n"] >= config.DEFAULT_MIN_ORDERS].sort_values("rate", ascending=False)
    assert sorted(_bar_labels(at, "reg_route_bar")) == sorted(f"{s}→{c}" for s, c in ranked.index[:12])
    at.slider(key="reg_route_topn").set_value(5).run()
    assert sorted(_bar_labels(at, "reg_route_bar")) == sorted(f"{s}→{c}" for s, c in ranked.index[:5])
    at.slider(key="reg_route_topn").set_value(20).run()
    assert len(_bar_labels(at, "reg_route_bar")) == 20
    # the route filters narrow the chart as well as the table
    at.multiselect(key="reg_pair_seller").set_value(["MG"]).run()
    labels = _bar_labels(at, "reg_route_bar")
    assert labels and all(lb.startswith("MG→") for lb in labels)
    at.multiselect(key="reg_pair_seller").set_value([]).run()
    at.multiselect(key="reg_pair_customer").set_value(["RJ"]).run()
    labels = _bar_labels(at, "reg_route_bar")
    assert labels and all(lb.endswith("→RJ") for lb in labels)


def test_filter_change_within_one_session_updates_all_numbers(orders):
    """The cached bundles are keyed on the filter state: changing a sidebar filter must change the numbers."""
    at = _run()
    assert _kpi(at, "Overall late rate")["value"] == ui.fmt_pct(orders["is_late"].mean())
    at.session_state[ui.K_STATES] = ["SP"]
    at.run()
    assert not at.exception
    sp = orders[orders["customer_state"] == "SP"]
    assert _kpi(at, "Overall late rate")["value"] == ui.fmt_pct(sp["is_late"].mean())
    assert list(_table(at, "reg_table")["customer_state"]) == ["SP"]
    assert _kpi(at, "Customer states in view")["value"] == "1"
    cat = orders["product_category"].value_counts().index[0]
    at.session_state[ui.K_STATES] = []
    at.session_state[ui.K_CATEGORIES] = [cat]
    at.run()
    sub = orders[orders["product_category"] == cat]
    assert _kpi(at, "Overall late rate")["value"] == ui.fmt_pct(sub["is_late"].mean())
    assert int(_table(at, "reg_table")["n_orders"].sum()) == len(sub)
    # routes are cached too
    assert int(_table(at, "reg_route_table")["n_orders"].sum()) <= len(sub)
    # and so is the heat map
    at.session_state[ui.K_CATEGORIES] = []
    at.run()
    assert _kpi(at, "Overall late rate")["value"] == ui.fmt_pct(orders["is_late"].mean())


def test_single_state_view_wording_is_not_degenerate(orders):
    at = _run(states=["SP"], categories=[orders["product_category"].value_counts().index[0]])
    assert not at.exception, [e.value for e in at.exception]
    html = _html(at)
    texts = " ".join(t for _, t in _callouts(at))
    assert "1 states" not in html and "1 categories" not in html and "of 1 state screened" in html
    assert "has the highest observed late rate" not in texts and "The lowest is" not in texts
    assert "Only **" not in texts and "Only SP" in texts
    k = _kpi(at, "Only ranked state")
    assert k["delta"] is None                                      # no '+0.0 pp vs overall' alert
    assert "Highest observed late rate" not in [x["label"] for x in _kpis(at)]
    assert "the highest-rate state" not in texts


def test_historical_caveat_is_on_the_page(orders):
    at = _run()
    text = " ".join(str(c.value) for c in at.caption)
    p = pd.to_datetime(orders["purchase_date"])
    assert "Historical Olist orders" in text and f"{p.min():%b %Y} to {p.max():%b %Y}" in text
    assert "not a measure of current" in text
    assert "not current logistics performance" in _callout(at, "Suggested action")


def test_insufficient_data_header_does_not_claim_rates_are_omitted():
    at = _run()
    html = _html(at)
    assert "omitted" not in html
    assert "never ranked or flagged" in html


def test_regional_table_assessment_is_short_and_csv_keeps_full_wording(orders, downloads):
    at = _run()
    tbl = _table(at, "reg_table")
    assert set(tbl["flag"]) <= {"Above overall", "Below overall", "Overlaps overall", "Insufficient data"}
    csv = pd.read_csv(io.BytesIO([d for d in downloads if d["key"] == "reg_download"][-1]["data"]))
    short = csv["flag"].str.replace(r" \(.*\)$", "", regex=True).replace({"Interval includes overall": "Overlaps overall"})
    assert list(short) == list(tbl["flag"])
    assert len(tbl) == 27


def test_widget_values_survive_a_visit_to_another_page():
    """Page-local widget values are restored when the page is shown again after other pages ran (Streamlit
    drops the state of widgets that were not rendered; the page keeps a shadow copy)."""
    at = AppTest.from_file(PAGE, default_timeout=TIMEOUT)
    at.session_state["_keep_reg_min_orders"] = 500
    at.session_state["_keep_reg_route_topn"] = 7
    at.session_state["_keep_reg_map_metric"] = "Order volume"
    at.session_state["_keep_reg_lookup_seller"] = "ZZ"            # no longer a valid option: ignored, no crash
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.slider(key="reg_min_orders").value == 500
    assert at.slider(key="reg_route_topn").value == 7
    assert at.radio(key="reg_map_metric").value == "Order volume"
    assert at.selectbox(key="reg_lookup_seller").value == "SP"
    # changing a value updates the shadow copy that a later visit restores
    at.slider(key="reg_min_orders").set_value(300).run()
    at.run()
    assert at.session_state["_keep_reg_min_orders"] == 300
