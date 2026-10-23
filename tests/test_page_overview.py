"""Tests for the Overview page (views/1_Overview.py) and its text builders (src/overview_text.py)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.io as pio
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import analytics as an
from src import charts, config, insights as ins_mod, ui
from src import overview_text as ot

ROOT = Path(__file__).resolve().parents[1]
PAGE = str(ROOT / "views" / "1_Overview.py")
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


def _text(at: AppTest) -> str:
    return "\n".join([str(e.value) for e in at.get("html")] + [str(m.value) for m in at.markdown]
                     + [str(c.value) for c in at.caption])


def _kpi_values(at: AppTest) -> dict[str, str]:
    """label -> displayed value, parsed from the KPI card HTML."""
    out = {}
    for m in re.finditer(r'diq-kpi-label">([^<]*)</span>.*?diq-kpi-value">([^<]*)</div>', _text(at), flags=re.S):
        out[m.group(1)] = m.group(2)
    return out


def _run(states=None, categories=None, date_range=None) -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=TIMEOUT)
    if states is not None:
        at.session_state["flt_states"] = states
    if categories is not None:
        at.session_state["flt_categories"] = categories
    if date_range is not None:
        at.session_state["flt_date_range"] = date_range
    return at.run()


# --------------------------------------------------------------------------- #
# Page (AppTest)
# --------------------------------------------------------------------------- #
@needs_data
def test_default_page_renders_with_all_sections(orders):
    at = _run()
    assert not at.exception
    txt = _text(at)
    for heading in ("Overview", "Headline findings", "Delivery performance", "Monthly delivery performance",
                    "Regional delay overview"):
        assert heading in txt
    assert len(at.get("plotly_chart")) == 4
    assert "Observed finding" in txt and "Suggested next step" in txt
    # widgets have the documented keys
    assert [s.key for s in at.select_slider] == ["ov_state_min_orders"]
    assert any(b.key == "ov_monthly_view" for b in at.get("button_group"))


@needs_data
def test_kpi_cards_match_pandas_recomputation(orders):
    at = _run()
    v = _kpi_values(at)
    n = len(orders)
    n_late = int(orders["is_late"].sum())
    late = orders[orders["is_late"] == 1]
    assert v["Delivered orders"] == f"{n:,}"
    assert v["Late-delivery rate"] == f"{100 * n_late / n:.1f}%"
    assert v["On-time rate"] == f"{100 * (n - n_late) / n:.1f}%"
    assert v["Avg delivery time"] == f"{orders['delivery_duration_days'].mean():.1f} days"
    assert v["Median delivery time"] == f"{orders['delivery_duration_days'].median():.1f} days"
    assert v["Avg lateness (late orders)"] == f"{late['delivery_delay_days'].astype(float).mean():.1f} days"
    assert v["Total freight value"] == f"R$ {orders['total_freight'].sum():,.0f}"
    assert v["Customer states covered"] == str(orders["customer_state"].nunique())
    # N of M is shown prominently
    assert f"{n:,} of {n:,} orders" in _text(at)


@needs_data
def test_filtered_view_kpis_and_deltas(orders):
    at = _run(states=["SP"], categories=["health_beauty"])
    assert not at.exception
    sub = orders[(orders["customer_state"] == "SP") & (orders["product_category"] == "health_beauty")]
    v = _kpi_values(at)
    assert v["Delivered orders"] == f"{len(sub):,}"
    assert v["Late-delivery rate"] == f"{100 * sub['is_late'].mean():.1f}%"
    assert v["Customer states covered"] == "1"
    txt = _text(at)
    assert f"{len(sub):,} of {len(orders):,} orders" in txt
    assert "vs all orders" in txt            # deltas against the unfiltered data appear only when filtering
    assert "SP" in txt


@needs_data
def test_default_view_has_no_delta_chips():
    at = _run()
    assert "vs all orders" not in _text(at)


@needs_data
def test_date_filter_drives_kpis(orders):
    lo, hi = pd.Timestamp("2018-01-01").date(), pd.Timestamp("2018-03-31").date()
    at = _run(date_range=(lo, hi))
    assert not at.exception
    sub = orders[(orders["purchase_date"] >= pd.Timestamp(lo)) & (orders["purchase_date"] <= pd.Timestamp(hi))]
    assert sub.shape[0] > 0
    assert _kpi_values(at)["Delivered orders"] == f"{len(sub):,}"


@needs_data
def test_empty_filter_shows_friendly_state_without_charts():
    at = _run(states=["RR"], categories=["fixed_telephony"])
    assert not at.exception
    txt = _text(at)
    assert "No orders match the current filters" in txt
    assert len(at.get("plotly_chart")) == 0
    assert "Delivered orders" not in _kpi_values(at)
    assert "nan" not in txt.lower().split("<style")[0].replace("financ", "")  # no raw NaN text in the content


@needs_data
def test_small_sample_warns_and_skips_monthly_chart(orders):
    at = _run(states=["RR"])                      # 41 orders: below the 100-order reliability threshold
    assert not at.exception
    txt = _text(at)
    sub = orders[orders["customer_state"] == "RR"]
    assert "Small sample" in txt
    assert f"Only {len(sub):,} orders match" in txt
    assert "No month has 100 or more orders" in txt     # friendly empty state instead of empty rate panels
    assert len(at.get("plotly_chart")) == 3           # donut, histogram, state bars (no monthly chart)
    assert "Insufficient data to draw conclusions" in txt


@needs_data
def test_min_orders_slider_and_monthly_view_rerun_cleanly(orders):
    at = _run()
    at.select_slider(key="ov_state_min_orders").set_value(500).run()
    assert not at.exception
    sp = an.state_performance(orders, min_orders=500)
    n_small = int((~sp["reliable"]).sum())
    assert f"{ot._pl(n_small, 'state')} with fewer than 500 orders" in _text(at)
    bg = next(b for b in at.get("button_group") if b.key == "ov_monthly_view")
    bg.set_value("Late rate").run()
    assert not at.exception
    assert len(at.get("plotly_chart")) == 4


@needs_data
def test_headline_cards_come_from_insights_and_are_sorted_by_severity(orders):
    at = _run()
    expected = ot.headline_insights(ins_mod.generate_insights(orders), 3)
    txt = _text(at)
    assert len(expected) == 3
    for i in expected:
        assert i.title in txt
    assert txt.count('class="diq-find ') == 3
    assert "Delay Analysis" in txt and "Regional Intelligence" in txt and "Reports & Insights" in txt


@needs_data
def test_interpretation_numbers_match_data(orders):
    at = _run()
    txt = _text(at)
    sp = an.state_performance(orders, min_orders=config.DEFAULT_MIN_ORDERS)
    ok = sp[sp["reliable"]]
    top = ok.sort_values("late_rate", ascending=False).iloc[0]
    assert f"{100 * top['late_rate']:.1f}%" in txt and f"n = {int(top['n_orders']):,}" in txt
    d = an.duration_summary(orders)
    assert f"{d['p90']:.1f} days" in txt
    overall = orders["is_late"].mean()
    n_above = int((ok["late_rate"] > overall).sum())
    assert f"{n_above} of these {len(ok)} states" in txt and f"above the overall {100 * overall:.1f}%" in txt


# --------------------------------------------------------------------------- #
# Review round 2: regression tests for chart data, KPI deltas, caches, caveats and boundaries
# --------------------------------------------------------------------------- #
def _fig(at: AppTest, key: str):
    el = next(e for e in at.get("plotly_chart") if e.key == key)
    return pio.from_json(el.proto.spec)


def _arr(v) -> np.ndarray:
    """Plotly 7 ships numeric arrays as {'dtype', 'bdata'}; decode them (plain lists pass through)."""
    if isinstance(v, dict) and "bdata" in v:
        import base64
        return np.frombuffer(base64.b64decode(v["bdata"]), dtype=np.dtype(v["dtype"]))
    return np.asarray(v)


def _kpi_cards(at: AppTest) -> dict[str, dict]:
    """label -> {value, sub, delta, tone} parsed from the KPI card HTML."""
    out = {}
    for card in _text(at).split('<div class="diq-kpi" ')[1:]:
        label = re.search(r'diq-kpi-label">([^<]*)<', card).group(1)
        delta = re.search(r'<span class="diq-kpi-delta ?(good|bad)?">([^<]*)</span>', card)
        sub = re.search(r'diq-kpi-sub">([^<]*)<', card)
        out[label] = dict(value=re.search(r'diq-kpi-value">([^<]*)<', card).group(1),
                          sub=sub.group(1) if sub else "", delta=delta.group(2) if delta else None,
                          tone=(delta.group(1) or "neutral") if delta else None)
    return out


def _sel(orders, state="SP", cat="health_beauty"):
    return orders[(orders["customer_state"] == state) & (orders["product_category"] == cat)]


@needs_data
def test_kpi_delta_sign_tone_and_reference_are_the_unfiltered_data(orders):
    at = _run(states=["SP"], categories=["health_beauty"])
    cards = _kpi_cards(at)
    sub = _sel(orders)
    late_diff = sub["is_late"].mean() - orders["is_late"].mean()
    assert abs(late_diff) > 0.0005               # fixture sanity: the deltas are not trivially zero
    assert cards["Late-delivery rate"]["delta"] == f"{100 * late_diff:+.1f} pp vs all orders"
    assert cards["Late-delivery rate"]["tone"] == ("bad" if late_diff > 0 else "good")      # lower is better
    assert cards["On-time rate"]["delta"] == f"{-100 * late_diff:+.1f} pp vs all orders"
    assert cards["On-time rate"]["tone"] == ("good" if late_diff < 0 else "bad")            # higher is better
    dur = sub["delivery_duration_days"].mean() - orders["delivery_duration_days"].mean()
    assert cards["Avg delivery time"]["delta"] == f"{dur:+.1f} days vs all orders"
    assert cards["Avg delivery time"]["tone"] == ("bad" if dur > 0 else "good")


def test_delta_helper_tones_and_guards():
    src = Path(PAGE).read_text()
    # the helper is pure: exercise it from the page source without running the page
    ns: dict = {"ui": ui}
    start = src.index("def _delta(")
    end = src.index("def _kpis(")
    exec(src[start:end], ns)
    delta = ns["_delta"]
    assert delta(0.10, 0.05, kind="pp", lower_is_better=True, active=True) == {"delta": "+5.0 pp vs all orders", "delta_tone": "bad"}
    assert delta(0.10, 0.05, kind="pp", lower_is_better=False, active=True)["delta_tone"] == "good"
    assert delta(5.0, 7.0, kind="days", lower_is_better=True, active=True) == {"delta": "-2.0 days vs all orders", "delta_tone": "good"}
    assert delta(0.0500, 0.0501, kind="pp", lower_is_better=True, active=True)["delta_tone"] == "neutral"
    assert delta(0.1, 0.05, kind="pp", lower_is_better=True, active=False) == {}
    assert delta(None, 0.05, kind="pp", lower_is_better=True, active=True) == {}
    assert delta(float("nan"), 0.05, kind="pp", lower_is_better=True, active=True) == {}


@needs_data
def test_kpi_sublabels_ci_and_on_time_count(orders):
    at = _run(states=["SP"], categories=["health_beauty"])
    cards = _kpi_cards(at)
    sub = _sel(orders)
    n, n_late = len(sub), int(sub["is_late"].sum())
    lo, hi = an.wilson_ci(n_late, n)
    assert lo <= n_late / n <= hi
    assert cards["Late-delivery rate"]["sub"] == f"{n_late:,} late · 95% CI {100 * lo:.1f}% to {100 * hi:.1f}%"
    assert cards["On-time rate"]["sub"].startswith(f"{n - n_late:,} on time or early")
    assert f"n = {n:,}" in cards["On-time rate"]["sub"]
    assert cards["Avg lateness (late orders)"]["sub"] == f"Among {n_late:,} late orders"


@needs_data
def test_chart_data_matches_pandas(orders):
    at = _run()
    n, n_late = len(orders), int(orders["is_late"].sum())
    donut = _fig(at, "ov_donut").data[0]
    assert dict(zip(donut.labels, donut.values)) == {"On time (or early)": n - n_late, "Late": n_late}
    hist = _fig(at, "ov_hist")
    bars = hist.data[0]
    v = orders["delivery_duration_days"].astype(float)
    top = np.ceil(v.quantile(0.99))
    assert int(_arr(bars.y).sum()) == int((v <= top).sum()) and _arr(bars.x)[0] == 0.5
    names = [t.name for t in hist.data]
    assert f"Mean ({v.mean():.1f} d)" in names and f"Median ({v.median():.1f} d)" in names
    mon = an.monthly_performance(orders, min_orders=100).sort_values("purchase_year_month")
    mfig = _fig(at, "ov_monthly")
    late_tr = next(t for t in mfig.data if t.name == "Late rate")
    exp = mon[mon["n_orders"] >= 100].set_index("purchase_year_month")["late_rate"]
    got = pd.Series(_arr(late_tr.y).astype(float), index=pd.to_datetime(list(_arr(late_tr.x))).strftime("%Y-%m"))
    got = got.dropna()
    assert len(got) == len(exp) and np.allclose(got.sort_index().to_numpy(), exp.sort_index().to_numpy())
    vol = next(t for t in mfig.data if t.name == "Orders")
    assert int(_arr(vol.y).sum()) == int(mon.loc[mon["n_orders"] >= 100, "n_orders"].sum())
    on_tr = next(t for t in mfig.data if t.name == "On-time rate")
    on_y = _arr(on_tr.y).astype(float)
    assert np.allclose(on_y[~np.isnan(on_y)] + got.sort_index().to_numpy(), 1.0)


def _state_bars(fig) -> dict[str, tuple[float, str]]:
    out = {}
    for t in fig.data:
        if t.type == "bar":
            for lab, x in zip(_arr(t.y), _arr(t.x)):
                out[lab.split(" ")[0]] = (float(x), t.name)
    return out


@needs_data
def test_state_bars_classes_follow_ci_and_min_orders_slider(orders):
    at = _run()
    sp = an.state_performance(orders, min_orders=100).set_index("customer_state")
    overall = orders["is_late"].mean()
    bars = _state_bars(_fig(at, "ov_states"))
    assert set(bars) == set(sp.index)
    for st_, (rate, name) in bars.items():
        r = sp.loc[st_]
        assert rate == pytest.approx(r["late_rate"])
        if r["n_orders"] < 100:
            assert name.startswith("Small sample")
        elif r["late_ci_low"] > overall:
            assert name.startswith("Above overall")
        elif r["late_ci_high"] < overall:
            assert name.startswith("Below overall")
        else:
            assert name.startswith("CI includes")
    at.select_slider(key="ov_state_min_orders").set_value(1000).run()
    bars2 = _state_bars(_fig(at, "ov_states"))
    n_small_1000 = int((sp["n_orders"] < 1000).sum())
    assert sum(1 for _, nm in bars2.values() if nm.startswith("Small sample")) == n_small_1000
    assert "(n < 1,000)" in " ".join(t.name or "" for t in _fig(at, "ov_states").data)
    assert f"{ot._pl(n_small_1000, 'state')} with fewer than 1,000 orders" in _text(at)   # table and text use the slider too


@needs_data
def test_monthly_view_control_changes_the_chart():
    at = _run()
    names = lambda: {t.name for t in _fig(at, "ov_monthly").data}
    assert {"Late rate", "On-time rate"} <= names()
    bg = next(b for b in at.get("button_group") if b.key == "ov_monthly_view")
    bg.set_value("Late rate").run()
    assert "Late rate" in names() and "On-time rate" not in names()
    bg = next(b for b in at.get("button_group") if b.key == "ov_monthly_view")
    bg.set_value("On-time rate").run()
    assert "On-time rate" in names() and "Late rate" not in names()


@needs_data
def test_cached_tables_follow_the_filters(orders):
    """Changing the filters twice in one process must not serve stale tables (cache key includes the filters)."""
    at = _run()
    assert _kpi_values(at)["Delivered orders"] == f"{len(orders):,}"
    at.session_state["flt_states"] = ["SP"]
    at.run()
    assert _kpi_values(at)["Delivered orders"] == f"{int((orders['customer_state'] == 'SP').sum()):,}"
    at.session_state["flt_states"] = ["RJ"]
    at.run()
    assert _kpi_values(at)["Delivered orders"] == f"{int((orders['customer_state'] == 'RJ').sum()):,}"
    assert _kpi_values(at)["Late-delivery rate"] == f"{100 * orders[orders['customer_state'] == 'RJ']['is_late'].mean():.1f}%"
    donut = _fig(at, "ov_donut").data[0]
    assert int(_arr(donut.values).sum()) == int((orders["customer_state"] == "RJ").sum())
    bars = _state_bars(_fig(at, "ov_states"))
    assert set(bars) == {"RJ"}


@needs_data
def test_filter_banner_and_model_pointer_use_real_numbers(orders):
    at = _run(states=["MG"])
    n = int((orders["customer_state"] == "MG").sum())
    assert f"<b>{n:,}</b> of {len(orders):,} orders" in _text(at)
    metrics = json.loads(config.MODEL_METRICS_PATH.read_text())
    sel = metrics["selected_model"]["name"]
    test = metrics["models"][sel]["test"]
    val = metrics["models"][sel].get("validation") or {}
    cap = " ".join(str(c.value) for c in at.caption)
    assert f"ROC-AUC is {test['roc_auc']:.2f}" in cap and f"PR-AUC {test['pr_auc']:.2f}" in cap
    if val.get("roc_auc") is not None and f"{val['roc_auc']:.2f}" != f"{test['roc_auc']:.2f}":
        assert f"ROC-AUC is {val['roc_auc']:.2f}" not in cap


@needs_data
def test_footer_states_limitations_and_makes_no_causal_claim():
    at = _run()
    cap = " ".join(str(c.value) for c in at.caption)
    assert "does not describe current logistics performance" in cap and "associations, not proof of a cause" in cap
    assert "far from precise" in cap
    page = Path(PAGE).read_text() + Path(ROOT / "src" / "overview_text.py").read_text()
    for bad in ("because of", " causes ", "caused by", "is responsible for", "proves"):
        assert bad not in page.lower()


def test_caption_css_is_not_dimmed():
    """Streamlit dims captions with opacity 0.6; the limitations caption must stay at contrast >= 4.5:1."""
    assert re.search(r'stCaptionContainer"\]\{[^}]*opacity:\s*1', ui._CSS)


@needs_data
def test_headline_cards_carry_a_fixed_caveat_line():
    at = _run()
    txt = _text(at)
    assert txt.count('class="diq-find-caveat"') == txt.count('class="diq-find ') == 3
    assert "not proof of a cause" in txt or "not a confirmed effect" in txt
    assert "-webkit-line-clamp" not in txt    # no CSS clipping in the middle of a statistic: the text is a teaser
    texts = re.findall(r'<div class="diq-find-text">(.*?)</div>', txt, flags=re.S)
    assert len(texts) == 3 and all(len(t) <= 260 for t in texts)
    block = re.findall(r'<div class="diq-find-text">.*?</div><div class="diq-find-caveat">([^<]*)</div>', txt, flags=re.S)
    assert len(block) == 3 and all(b for b in block)


def test_teaser_cuts_at_a_sentence_or_list_item_never_inside_a_statistic():
    t = ot.teaser
    assert t("Short text.") == "Short text."
    two = "First sentence is here. " + "Second sentence is a lot longer than the limit allows for sure " * 3
    assert t(two, 60) == "First sentence is here."
    item = ("Compared with the overall 6.8%, these states are higher: AL (Alagoas): 21.4% late (n = 397; 95% CI 17.7% to 25.7%); "
            "MA (Maranhao): 17.4% late (n = 717; 95% CI 14.8% to 20.4%); SE (Sergipe): 15.2% late (n = 335)")
    out = t(item, 160)
    assert out.endswith(" ...") and out.count("(") == out.count(")")          # no dangling bracket
    assert "MA (" not in out and "n = 397; 95% CI 17.7% to 25.7%)" in out       # whole list items only
    longer = t(item, 200)
    assert longer.count("(") == longer.count(")")
    # no sentence boundary and no list item boundary: cut at a word, bracket closed
    out2 = t("word " * 10 + "(unfinished statistic n = 5 and more words follow here", 70)
    assert out2.endswith("...") and "(" not in out2


def test_pluralisation_helper():
    assert ot._pl(1, "state") == "1 state" and ot._pl(15, "month") == "15 months" and ot._pl(0, "state") == "0 states"


@needs_data
def test_single_state_or_category_view_does_not_show_a_self_comparison_card():
    df = pd.read_parquet(config.ORDERS_DELIVERED_PATH)
    sub = df[(df["customer_state"] == "RJ") & (df["product_category"] == "health_beauty")]
    from src import exports
    items = exports.comparable_insights(ins_mod.generate_insights(sub, 100), 100)
    kinds = {i.kind for i in items}
    assert not kinds & {"region_high", "category_duration", "category_delay", "category_freight"}
    head = ot.headline_insights(items, 3)
    assert all("1 segments" not in i.text for i in head)


def test_insight_caveat_by_kind():
    mk = lambda kind: ins_mod.Insight(kind, kind, "t", {}, 10, "info")
    assert "not proof of a cause" in ot.insight_caveat(mk("region_test"))
    assert "not a confirmed effect" in ot.insight_caveat(mk("region_high"))
    assert "not a confirmed effect" in ot.insight_caveat(object())


@needs_data
def test_chart_subtitles_fit_a_phone_width_column(orders):
    mon = an.monthly_performance(orders, min_orders=100)
    sp = an.state_performance(orders, min_orders=100)
    for fig in (charts.monthly_performance_chart(mon, min_orders=100), charts.state_late_rate_bar(sp, min_orders=100)):
        lines = fig.layout.title.subtitle.text.split("<br>")
        assert max(len(x) for x in lines) <= charts._NARROW_WRAP + 12      # one long word may overshoot a line
        n_legend = sum(1 for t in fig.data if t.showlegend is not False and t.name)
        assert fig.layout.margin.t >= 64 + 22 * (n_legend - 1)             # room for a legend that stacks one entry per row


@needs_data
@pytest.mark.parametrize("n, small", [(99, True), (100, False)])
def test_small_sample_boundary_is_100_orders(orders, monkeypatch, n, small):
    flt = dict(date_range=(orders["purchase_date"].min().date(), orders["purchase_date"].max().date()), states=[],
               categories=[], active=True, description="x", n_total=len(orders), n_filtered=n)
    monkeypatch.setattr(ui, "get_filtered_orders", lambda: (orders, orders.head(n), flt))
    at = AppTest.from_file(PAGE, default_timeout=TIMEOUT).run()
    assert not at.exception
    txt = _text(at)
    assert ("Small sample" in txt) is small
    assert (f"Only {n:,} orders match" in txt) is small


# --------------------------------------------------------------------------- #
# overview_text (pure)
# --------------------------------------------------------------------------- #
def _frame(n_on=90, n_late=10, state="SP", month="2018-01") -> pd.DataFrame:
    n = n_on + n_late
    return pd.DataFrame({
        "is_late": pd.array([0] * n_on + [1] * n_late, dtype="Int8"),
        "delivery_duration_days": np.linspace(3, 30, n),
        "delivery_delay_days": pd.array([-2] * n_on + [5] * n_late, dtype="Int32"),
        "total_freight": 10.0, "total_weight_g": 500.0,
        "customer_state": state, "purchase_year_month": month,
    })


def test_status_text_basic_and_filters():
    k = an.kpi_summary(_frame(950, 50))
    t = ot.status_text(k, an.kpi_summary(_frame(900, 100)), True)
    assert "**950**" in t.observed and "5.0%" in t.observed and "n" in t.observed
    assert "below the overall level" in t.observed and "-5.0 pp" in t.observed
    assert t.action and "50 late orders" in t.action


def test_status_text_edge_cases():
    assert "No delivered orders" in ot.status_text(an.kpi_summary(_frame(0, 0))).observed
    t = ot.status_text(an.kpi_summary(_frame(5, 0)))
    assert "No order in this selection was late" in t.observed and "Small sample" in t.observed
    assert t.action is None
    same = ot.status_text(an.kpi_summary(_frame(90, 10)), an.kpi_summary(_frame(90, 10)), True)
    assert "about the same as" in same.observed


def test_duration_text():
    d = an.duration_summary(_frame(900, 100))
    t = ot.duration_text(d)
    assert f"{d['median']:.1f} days" in t.observed and f"{d['p90']:.1f} days" in t.observed
    assert t.action
    empty = ot.duration_text(an.duration_summary(_frame(0, 0)))
    assert "No delivery times" in empty.observed and empty.action is None
    small = ot.duration_text(an.duration_summary(_frame(5, 1)), min_orders=100)
    assert "Small sample" in small.observed and small.action is None


def _monthly(rows):
    return pd.DataFrame(rows, columns=["purchase_year_month", "n_orders", "n_late"]).assign(
        late_rate=lambda d: d.n_late / d.n_orders, on_time_rate=lambda d: 1 - d.n_late / d.n_orders)


def test_monthly_text_cases():
    assert "No months" in ot.monthly_text(None).observed
    assert "No months" in ot.monthly_text(_monthly([]).iloc[0:0]).observed
    none_ok = ot.monthly_text(_monthly([("2018-01", 50, 5), ("2018-02", 60, 6)]), 100)
    assert "No month in this selection has 100" in none_ok.observed and none_ok.action is None
    one = ot.monthly_text(_monthly([("2018-01", 500, 50), ("2018-02", 60, 6)]), 100)
    assert "Only one month" in one.observed and "Jan 2018" in one.observed and "1 month with fewer" in one.observed and "1 month with fewer than" in one.observed and "is not charted as a rate" in one.observed
    m = _monthly([("2018-01", 1000, 50), ("2018-02", 1000, 150), ("2018-03", 2000, 100), ("2016-10", 10, 1)])
    t = ot.monthly_text(m, 100)
    assert "5.0%" in t.observed and "15.0%" in t.observed and "Feb 2018" in t.observed
    assert "change was +0.0 pp" in t.observed          # Jan 5.0% -> Mar 5.0%
    assert t.action and "Feb 2018" in t.action and "ordinary" in t.action
    busy = ot.monthly_text(_monthly([("2018-01", 1000, 50), ("2018-02", 3000, 450), ("2018-03", 1000, 50)]), 100)
    assert "3.0x the median of its 2 neighbouring months" in busy.action
    # the peak is compared with its neighbours, not with the whole (growing) series
    growth = _monthly([("2017-01", 500, 25), ("2017-02", 1000, 50), ("2017-03", 6000, 300), ("2017-04", 6500, 325),
                       ("2017-05", 6700, 670), ("2017-06", 6800, 340), ("2017-07", 7200, 360)])
    g = ot.monthly_text(growth, 100)       # all-months median is 6,500 / ordinary vs neighbours (median 6,650)
    assert "May 2017" in g.action and "ordinary (1.0x the median of its 4 neighbouring months)" in g.action
    assert "x the median month" not in g.action
    # ramp-up months (tiny volume) are not used as the first end of the first-to-last comparison
    assert "Between the first (Mar 2017, 5.0%, n = 6,000)" in g.observed and "(two end points, not a trend)" in g.observed
    assert "and the last (Jul 2017, 5.0%, n = 7,200)" in g.observed and "change was +0.0 pp" in g.observed
    two = ot.monthly_text(_monthly([("2018-01", 5000, 250), ("2018-02", 120, 12), ("2018-03", 5000, 500)]), 100)
    assert "Between the first (Jan 2018" in two.observed and "Mar 2018" in two.observed
    assert "change was +5.0 pp" in two.observed
    lonely_peak = ot.monthly_text(_monthly([("2018-01", 3000, 150), ("2018-02", 400, 120)]), 100)
    assert "Between the first" not in lonely_peak.observed            # 400 < 25% of the median: not well populated
    flat = ot.monthly_text(_monthly([("2018-01", 1000, 50), ("2018-02", 1000, 51)]), 100)
    assert "fairly stable" in flat.observed and flat.action is None


def _states(rows):
    return pd.DataFrame(rows, columns=["customer_state", "n_orders", "n_late"]).assign(
        late_rate=lambda d: d.n_late / d.n_orders)


def test_state_text_cases():
    assert "No states" in ot.state_text(None).observed
    assert "No states" in ot.state_text(_states([]).iloc[0:0]).observed
    tiny = ot.state_text(_states([("AC", 40, 4), ("AP", 30, 3)]), 100)
    assert "No state has 100 or more orders" in tiny.observed and "2 states" in tiny.observed and tiny.action is None
    one = ot.state_text(_states([("SP", 1000, 50), ("AC", 40, 4)]), 100)
    assert "Only SP (São Paulo)" in one.observed and "n = 1,000" in one.observed and "1 state with fewer" in one.observed and "is shown greyed out because its rate is unreliable" in one.observed
    many = ot.state_text(_states([("SP", 1000, 50), ("AL", 400, 80), ("MG", 500, 40), ("AC", 40, 4)]), 100, 0.1)
    assert "from **5.0%** (SP (São Paulo), n = 1,000)" in many.observed and "**20.0%** (AL (Alagoas), n = 400)" in many.observed
    assert "4.0x the lowest" in many.observed and "1 of these 3 states has a point estimate" in many.observed
    assert many.action and "AL" in many.action
    zero = ot.state_text(_states([("SP", 1000, 0), ("MG", 500, 10)]), 100)
    assert "x the lowest" not in zero.observed          # no division by a zero rate


def test_headline_insights_order_and_labels():
    mk = lambda kind, sev: ins_mod.Insight(kind, kind, "t", {}, 10, sev)
    items = [mk("overview", "info"), mk("region_test", "watch"), mk("region_high", "alert"), mk("category_duration", "info"),
             mk("month_high", "alert"), mk("same_state", "watch")]
    top = ot.headline_insights(items, 3)
    assert [i.kind for i in top] == ["region_high", "month_high", "region_test"]
    assert ot.insight_label(top[0]) == "Observed finding" and ot.insight_label(top[1]) == "Statistical association"
    only_insufficient = [mk("overview", "info"), mk("insufficient_data", "info")]
    assert [i.kind for i in ot.headline_insights(only_insufficient)] == ["insufficient_data"]
    assert ot.headline_insights([mk("no_data", "info")]) == []
    assert ot.headline_insights([], 3) == []


def test_text_builders_never_emit_nan_or_raise_on_empty():
    empty = _frame(0, 0)
    texts = [ot.status_text(an.kpi_summary(empty)), ot.duration_text(an.duration_summary(empty)),
             ot.monthly_text(an.monthly_performance(empty)), ot.state_text(an.state_performance(empty))]
    for t in texts:
        assert "nan" not in t.observed.lower()


# --------------------------------------------------------------------------- #
# Review round 3: short windows, sparse late events, notice cards, callout / colours / share
# --------------------------------------------------------------------------- #
@needs_data
def test_page_small_sample_callout_is_the_pages_own_and_sits_above_the_kpis(orders):
    """Pinned by its own title/class so the status text or the 'Insufficient data' card cannot satisfy it."""
    txt = _text(_run(states=["RR"]))
    m = re.search(r'<div class="diq-callout watch"[^>]*>.*?diq-callout-title">Small sample<.*?</div></div></div>', txt, flags=re.S)
    assert m and txt.index(m.group(0)) < txt.index('<div class="diq-kpi" ')
    n = int((orders["customer_state"] == "RR").sum())
    assert f"Only {n:,} orders match the filters (fewer than 100)" in m.group(0)
    # a big view has no such callout
    assert 'diq-callout-title">Small sample<' not in _text(_run())


@needs_data
def test_donut_uses_green_for_on_time_and_red_for_late():
    donut = _fig(_run(), "ov_donut").data[0]
    assert list(donut.labels) == ["On time (or early)", "Late"]
    assert list(donut.marker.colors) == [config.PALETTE["green"], config.PALETTE["red"]]


@needs_data
def test_share_of_all_orders_is_n_over_n_total(orders):
    at = _run(states=["SP"], categories=["health_beauty"])
    n = len(_sel(orders))
    assert f"{100 * n / len(orders):.1f}% of all delivered orders" in _kpi_cards(at)["Delivered orders"]["sub"]
    assert _kpi_cards(_run())["Delivered orders"]["sub"].endswith("100.0% of all delivered orders")


@needs_data
def test_short_window_monthly_chart_has_one_tick_per_month(orders):
    rng = (pd.Timestamp("2017-01-01").date(), pd.Timestamp("2017-04-30").date())
    at = _run(date_range=rng)
    assert not at.exception
    fig = _fig(at, "ov_monthly")
    assert fig.layout.xaxis3.dtick == "M1" and fig.layout.xaxis3.tickformat == "%b %Y"
    mon = an.monthly_performance(orders[(orders["purchase_date"] >= pd.Timestamp(rng[0]))
                                        & (orders["purchase_date"] <= pd.Timestamp(rng[1]))], min_orders=100)
    assert int((mon["n_orders"] >= 100).sum()) >= 2


def test_pct_tickformat_never_repeats_labels():
    for lo, hi in [(0.9890, 0.9990), (0.40, 0.99), (0.0, 0.02), (0.9990, 0.9995), (0.02, 0.09), (0.0, 0.30)]:
        fmt = charts._pct_tickformat([lo, hi])
        decimals = {".0%": 0, ".1%": 1, ".2%": 2}[fmt]
        ticks = np.linspace(lo, hi, 5)                           # plotly draws roughly 4-6 ticks across the range
        labels = [f"{100 * t:.{decimals}f}" for t in ticks]
        assert len(set(labels)) == len(labels), (lo, hi, fmt)
    assert charts._pct_tickformat(None) == ".0%"


@needs_data
def test_monthly_chart_y_axis_is_precise_for_a_narrow_on_time_range():
    mon = pd.DataFrame({"purchase_year_month": ["2017-01", "2017-02", "2017-03"], "n_orders": [500, 600, 700],
                        "n_late": [2, 3, 2], "late_rate": [0.004, 0.005, 0.0029],
                        "on_time_rate": [0.996, 0.995, 0.9971]})
    fig = charts.monthly_performance_chart(mon, metric="on_time", min_orders=100)
    assert fig.layout.yaxis.tickformat != ".0%"


@needs_data
def test_one_chartable_month_shows_empty_state_not_a_near_empty_chart(orders):
    at = _run(date_range=(pd.Timestamp("2016-09-15").date(), pd.Timestamp("2016-12-31").date()))
    assert not at.exception
    txt = _text(at)
    sub = orders[orders["purchase_date"] <= pd.Timestamp("2016-12-31")]
    sub = sub[sub["purchase_date"] >= pd.Timestamp("2016-09-15")]
    mon = an.monthly_performance(sub, min_orders=100)
    assert int((mon["n_orders"] >= 100).sum()) == 1                  # fixture sanity
    assert "Only one month has 100 or more orders" in txt
    assert all(e.key != "ov_monthly" for e in at.get("plotly_chart"))


@needs_data
def test_headline_drops_late_rate_findings_resting_on_few_late_orders(orders):
    rng = (pd.Timestamp("2016-09-15").date(), pd.Timestamp("2016-12-31").date())
    sub = orders[(orders["purchase_date"] >= pd.Timestamp(rng[0])) & (orders["purchase_date"] <= pd.Timestamp(rng[1]))]
    assert int(sub["is_late"].sum()) < 20                            # fixture sanity: 3 late orders
    raw = ins_mod.generate_insights(sub)
    assert any(i.kind == "processing_time" for i in raw)             # the insight exists, the page must not headline it
    txt = _text(_run(date_range=rng))
    head = txt.split("Headline findings", 1)[1].split("Delivery performance", 1)[0]
    assert "Seller processing time and lateness" not in head
    assert 'class="diq-find ' in head                                # something (e.g. weight / freight) is still shown


def test_enough_late_events_filter_is_kind_aware():
    ns: dict = {}
    src = Path(PAGE).read_text()
    exec(src[src.index("#: Insight kinds whose claim"):src.index("# -------", src.index("#: Insight kinds whose claim"))], ns)
    mk = lambda kind: ins_mod.Insight(kind, kind, "t", {}, 500, "watch")
    items = [mk("processing_time"), mk("region_high"), mk("weight_freight"), mk("insufficient_data")]
    assert [i.kind for i in ns["_enough_late_events"](items, 19)] == ["weight_freight", "insufficient_data"]
    assert len(ns["_enough_late_events"](items, 20)) == 4


@needs_data
def test_insufficient_data_card_is_a_notice_without_the_finding_caveat():
    txt = _text(_run(states=["RR"]))
    head = txt.split("Headline findings", 1)[1].split("Delivery performance", 1)[0]
    assert 'class="diq-find ' in head and "Insufficient data to draw conclusions" in head
    assert 'diq-find-kind">Notice<' in head
    assert 'diq-find-kind">Observed finding<' not in head
    assert 'class="diq-find-caveat"' not in head
    assert "A lead to check" not in head.split("<details")[0]
