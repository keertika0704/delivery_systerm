"""AppTest tests for pages/5_Reports_Insights.py (run standalone, as the shell contract requires)."""
from __future__ import annotations

import html as _html
import json
import re
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import analytics as an
from src import config, exports, ui

PAGE = str(Path(__file__).resolve().parents[1] / "pages" / "5_Reports_Insights.py")
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


def run(state: dict | None = None) -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=TIMEOUT)
    for k, v in (state or {}).items():
        at.session_state[k] = v
    return at.run()


def html(at: AppTest) -> str:
    return "\n".join([str(e.value) for e in at.get("html")] + [str(m.value) for m in at.markdown])


def plain(at: AppTest) -> str:
    """Visible text: tags stripped from st.html payloads only (markdown may legitimately contain '<' and '>')."""
    parts = [re.sub(r"<[^>]+>", "", str(e.value)) for e in at.get("html")] + [str(m.value) for m in at.markdown]
    parts += [str(c.value) for c in at.caption]
    return re.sub(r"\s+", " ", _html.unescape(" ".join(parts)).replace("**", ""))


def kpi(at: AppTest, label: str) -> tuple[str, str]:
    m = re.search(rf'diq-kpi-label">{re.escape(label)}</span>.*?diq-kpi-value">([^<]*)</div>(?:<div class="diq-kpi-sub">([^<]*)</div>)?',
                  html(at), re.S)
    assert m, f"KPI {label!r} not found"
    return m.group(1), m.group(2) or ""


def no_exception(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]


def dl_buttons(at: AppTest):
    return at.get("download_button")


# --------------------------------------------------------------------------- #
def test_default_view_renders_all_tabs_without_exception(orders):
    at = run()
    no_exception(at)
    assert [t.label for t in at.tabs] == ["Key findings", "Downloads", "Data explorer", "Model evaluation"]
    text = plain(at)
    for needle in ("How to read this page", "Observed", "Statistical association", "Model prediction", "Suggested action",
                   "Methodology note", "Downloads", "Data explorer", "Model evaluation"):
        assert needle in text, needle
    k = an.kpi_summary(orders)
    assert kpi(at, "Orders in view")[0] == f"{k['n_orders']:,}"
    val, sub = kpi(at, "Late rate")
    assert val == f"{100 * k['late_rate']:.1f}%" and f"{k['n_late']:,} late orders" in sub and f"n = {k['n_orders']:,}" in sub
    assert kpi(at, "Mean delivery time")[0] == f"{k['mean_duration']:,.1f} days"


def test_findings_update_with_filters(orders):
    base = run()
    filt = run({ui.K_STATES: ["SP"], ui.K_CATEGORIES: ["health_beauty"]})
    no_exception(filt)
    sub = orders[(orders.customer_state == "SP") & (orders.product_category == "health_beauty")]
    k = an.kpi_summary(sub)
    assert kpi(filt, "Orders in view")[0] == f"{len(sub):,}"
    assert kpi(filt, "Late rate")[0] == f"{100 * k['late_rate']:.1f}%"
    assert html(filt) != html(base)
    # the overall insight text of the strongest cards must quote the filtered numbers
    assert f"{100 * k['late_rate']:.1f}%" in plain(filt)
    # single state: regional ranking cannot flag a state against itself
    assert "region(s) with a reliably higher" not in plain(filt)
    assert "region(s) with a reliably higher" in plain(base)


def test_min_orders_slider_changes_findings(orders):
    at = run()
    assert "n >= 100" in plain(at) and "at least 100 delivered orders" in plain(at)
    at.slider(key="rep_min_orders").set_value(1000).run()
    no_exception(at)
    text = plain(at)
    assert "n >= 100," not in text and "n >= 1,000" in text and "at least 1,000 delivered orders" in text
    n_rel = int((orders.groupby("customer_state", observed=True).size() >= 1000).sum())
    assert f"Screened {n_rel} regions with n >= 1,000" in text or f"{n_rel} regions with n >= 1,000" in text
    # fewer orders in view than the minimum: comparisons are skipped, not ranked on a tiny sample
    at.session_state[ui.K_STATES] = ["RR"]
    at.run()
    no_exception(at)
    text = plain(at)
    assert "Small sample" in text and "Insufficient data" in text and "region(s) with a reliably" not in text


def test_confidence_selector_changes_interval_level():
    at = run()
    assert "95% interval" in plain(at)
    at.selectbox(key="rep_confidence").set_value(0.99).run()
    no_exception(at)
    assert "99% interval" in plain(at) and "99% Wilson" in plain(at)


def test_only_flagged_toggle_hides_info_cards():
    at = run()
    assert "rp-sev" in html(at) and ">Info<" in html(at)
    at.toggle(key="rep_only_flagged").set_value(True).run()
    no_exception(at)
    assert ">Info<" not in html(at) and ">Watch<" in html(at)


def test_empty_filter_result_is_friendly_everywhere(orders):
    assert ((orders.customer_state == "AP") & (orders.product_category == "cds_dvds_musicals")).sum() == 0
    at = run({ui.K_STATES: ["AP"], ui.K_CATEGORIES: ["cds_dvds_musicals"]})
    no_exception(at)
    text = plain(at)
    assert "No orders match the current filters" in text
    assert text.count("No orders match the current filters") >= 3       # findings, downloads, explorer
    btns = dl_buttons(at)
    assert btns and all(b.proto.disabled for b in btns)
    assert "Model evaluation" in text and "ROC-AUC" in text                # model tab does not depend on the filters


def test_downloads_tab_buttons_and_sizes(orders):
    at = run({ui.K_STATES: ["SP"], ui.K_CATEGORIES: ["health_beauty"]})
    no_exception(at)
    labels = [b.proto.label for b in dl_buttons(at)]
    for lab in ("Download shipment CSV", "Download regional CSV", "Download report (.md)", "Download report (.html)",
                "Download chart (.html)"):
        assert lab in labels, labels
    sub = orders[(orders.customer_state == "SP") & (orders.product_category == "health_beauty")]
    text = plain(at)
    assert f"{len(sub):,} rows" in re.sub(r"\s+", " ", text)
    assert re.search(r"\d+(\.\d)? (KB|MB|B)", text)
    # the regional table shown is the regional report (one state here)
    reg_tables = [d.value for d in at.dataframe if "state_name" in d.value.columns]
    assert len(reg_tables) == 1 and reg_tables[0]["customer_state"].tolist() == ["SP"]
    assert int(reg_tables[0]["n_orders"].iloc[0]) == len(sub)


def test_regional_table_matches_report(orders):
    at = run()
    reg = [d.value for d in at.dataframe if "state_name" in d.value.columns][0]
    exp = pd.read_csv(__import__("io").BytesIO(exports.regional_csv_bytes(
        __import__("src.insights", fromlist=["x"]).regional_report(orders, config.DEFAULT_MIN_ORDERS))), encoding="utf-8-sig")
    assert reg["customer_state"].tolist() == exp["customer_state"].tolist()
    assert reg["n_orders"].tolist() == exp["n_orders"].tolist()


# --------------------------------------------------------------------------- #
# Data explorer
# --------------------------------------------------------------------------- #
def explorer_table(at: AppTest) -> pd.DataFrame:
    return [d.value for d in at.dataframe if "order_id" in d.value.columns][0]


def test_explorer_paginates_and_never_renders_everything(orders):
    at = run()
    no_exception(at)
    t = explorer_table(at)
    assert len(t) == 50
    text = plain(at)
    assert f"{len(orders):,} matching orders" in text and "Page 1 of 1,930" in text
    first = t.iloc[:, 0].tolist()
    at.button(key="rep_ex_next").click().run()
    no_exception(at)
    assert "Page 2 of 1,930" in plain(at)
    assert set(explorer_table(at).iloc[:, 0]).isdisjoint(first)
    at.selectbox(key="rep_ex_size").set_value(250).run()
    assert len(explorer_table(at)) == 250 and "Page 1 of 386" in plain(at)       # back to page 1, ceil(96470/250)


def test_explorer_filters_compose_with_sidebar_and_match_pandas(orders):
    at = run({ui.K_STATES: ["SP", "RJ"]})
    at.radio(key="rep_ex_status").set_value("Late").run()
    at.multiselect(key="rep_ex_states").set_value(["SP"]).run()
    at.multiselect(key="rep_ex_cats").set_value(["health_beauty", "toys"]).run()
    no_exception(at)
    exp = orders[(orders.customer_state == "SP") & orders.product_category.isin(["health_beauty", "toys"]) & (orders.is_late == 1)]
    text = plain(at)
    assert f"{len(exp):,} matching orders" in text
    assert f"{100 * 1.0:.1f}% late (n = {len(exp):,})" in text
    t = explorer_table(at)
    assert set(t["customer_state"]) == {"SP"} and set(t["delivery_status"]) == {"late"}
    assert len(t) == min(50, len(exp))


def test_explorer_state_outside_sidebar_filter_gives_empty_state(orders):
    at = run({ui.K_STATES: ["SP"]})
    at.session_state["rep_ex_states"] = []
    at.run()
    at.text_input(key="rep_ex_search").set_value("zzzz-no-such-order").run()
    no_exception(at)
    assert "No orders match the explorer filters" in plain(at)
    assert at.button(key="rep_ex_reset") is not None
    at.button(key="rep_ex_reset").click().run()
    no_exception(at)
    assert "No orders match the explorer filters" not in plain(at)
    assert f"{(orders.customer_state == 'SP').sum():,} matching orders" in plain(at)


def test_explorer_search_and_sort(orders):
    at = run()
    some = orders["order_id"].iloc[100]
    at.text_input(key="rep_ex_search").set_value(some).run()
    no_exception(at)
    assert explorer_table(at).iloc[:, 0].tolist() == [some]
    at = run()
    at.selectbox(key="rep_ex_sort").set_value("total_price").run()
    at.radio(key="rep_ex_dir").set_value("Descending").run()
    t = explorer_table(at)
    assert t["total_price"].iloc[0] == pytest.approx(orders["total_price"].max())
    assert t["total_price"].is_monotonic_decreasing


def test_explorer_columns_are_whitelisted_only(orders):
    at = run()
    assert set(at.multiselect(key="rep_ex_cols").options) == {exports.column_label(c) for c in exports.EXPORT_COLUMNS}
    t = explorer_table(at)
    assert not {"customer_id", "seller_id", "order_status"} & set(t.columns)


# --------------------------------------------------------------------------- #
# Model evaluation
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not config.MODEL_METRICS_PATH.is_file(), reason="model metrics missing")
def test_model_tab_numbers_come_from_metrics_json():
    m = json.loads(config.MODEL_METRICS_PATH.read_text(encoding="utf-8"))
    sel = m["selected_model"]["name"]
    t = m["models"][sel]["test"]
    at = run()
    no_exception(at)
    val, sub = kpi(at, "Test ROC-AUC")
    assert val == f"{t['roc_auc']:.2f}" and f"n = {t['n']:,}" in sub
    assert kpi(at, "Test PR-AUC")[0] == f"{t['pr_auc']:.3f}"
    assert kpi(at, "Validation ROC-AUC")[0] == f"{m['models'][sel]['validation']['roc_auc']:.2f}"
    text = re.sub(r"\s+", " ", plain(at))
    for needle in ("Train / validation / test split", "Model comparison", "Risk bands", "Leakage audit", "Which metrics matter",
                   "Honest summary", "Accuracy is misleading", "not an accurate predictor"):
        assert needle in text, needle
    for lab in ("train", "validation", "test"):
        assert f"{100 * m['split'][lab]['late_rate']:.1f}%" in text
    plots = [e for e in at.get("plotly_chart")]
    assert len(plots) >= 7                                                    # compare, confusion, roc, pr, calibration, bands, importance
    tables = {tuple(d.value.columns)[:2]: d.value for d in at.dataframe}
    split = next(v for k_, v in tables.items() if k_ == ("Window", "Role"))
    assert split["Orders (n)"].tolist() == [m["split"][w]["n"] for w in ("train", "validation", "test")]
    # model tab is independent of the sidebar filters
    at2 = run({ui.K_STATES: ["SP"]})
    assert kpi(at2, "Test ROC-AUC") == (val, sub)


def test_model_tab_without_metrics_file_shows_training_command(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MODEL_METRICS_PATH", tmp_path / "missing.json")
    at = run()
    no_exception(at)
    assert "Model evaluation is not available" in plain(at)
    assert any(ui.TRAIN_COMMAND in c.value for c in at.code)
    assert "Test ROC-AUC" not in html(at).split("Model evaluation is not available")[-1]


# --------------------------------------------------------------------------- #
# Regressions: settings must reach the findings, p-values, file names, warnings
# --------------------------------------------------------------------------- #
def squash(s: str) -> str:
    return re.sub(r"\s+", " ", s)


def cards_text(at: AppTest) -> str:
    """Text of the finding cards only (not the methodology or the report preview)."""
    blocks = [str(e.value) for e in at.get("html") if "rp-card" in str(e.value)]
    return squash(_html.unescape(re.sub(r"<[^>]+>", "", " ".join(blocks))))


def test_confidence_selector_reaches_the_finding_cards(orders):
    """The selector must change the interval level (alpha) used by the findings, not just a label."""
    from src import insights as ins
    at = run()
    at.selectbox(key="rep_confidence").set_value(0.99).run()
    no_exception(at)
    items99 = [i for i in ins.generate_insights(orders, config.DEFAULT_MIN_ORDERS, 0.01) if i.kind != "overview"]
    items95 = [i for i in ins.generate_insights(orders, config.DEFAULT_MIN_ORDERS, 0.05) if i.kind != "overview"]
    text = cards_text(at)
    assert "99% CI" in text and "95% CI" not in text
    for i in items99:
        assert squash(i.text) in text, i.kind
    assert any(squash(i.text) not in text for i in items95)          # the 95% wording is gone
    assert "99% interval" in plain(at)


def test_min_orders_slider_reaches_the_finding_cards(orders):
    from src import insights as ins
    at = run()
    at.slider(key="rep_min_orders").set_value(1000).run()
    no_exception(at)
    text = cards_text(at)
    assert "n >= 1,000" in text and "n >= 100," not in text and "n >= 100 " not in text
    for i in ins.generate_insights(orders, 1000, 0.05):
        if i.kind != "overview":
            assert squash(i.text) in text, i.kind


def test_card_shows_small_sample_warning(monkeypatch):
    from src import insights as ins
    real = ins.generate_insights

    def with_small(df, min_orders=config.DEFAULT_MIN_ORDERS, alpha=config.DEFAULT_ALPHA):
        return list(real(df, min_orders, alpha)) + [ins.Insight("region_high", "Tiny segment finding", "Based on 40 orders.",
                                                                  {}, 40, "info")]

    monkeypatch.setattr(ins, "generate_insights", with_small)
    at = run()
    no_exception(at)
    page = html(at)
    assert page.count('class="rp-warn"') == 1
    assert "Small sample (n = 40; below 100): treat rates as indicative only." in plain(at)
    # a non-small card never carries the warning
    cards = [c for c in re.findall(r'<div class="rp-card".*?</div></div>', page, re.S)]
    assert sum("rp-warn" in c for c in cards) == 1


def test_no_underflowed_p_values_anywhere_on_the_page():
    at = run()
    no_exception(at)
    text = plain(at)
    assert "e-30" not in text and "2.2e-308" not in text
    assert "p < 0.001" in cards_text(at)


def test_download_file_names_record_sidebar_filters(monkeypatch):
    names: list[str] = []
    real = exports.export_filename

    def spy(*a, **k):
        out = real(*a, **k)
        names.append(out)
        return out

    monkeypatch.setattr(exports, "export_filename", spy)
    at = run({ui.K_STATES: ["SP"], ui.K_CATEGORIES: ["health_beauty"]})
    no_exception(at)
    dl = [n for n in names if "explorer" not in n]
    assert len(dl) == 5 and all("_SP_health-beauty_" in n for n in dl), dl
    ex = [n for n in names if "explorer" in n]
    assert len(ex) == 2 and all("_SP_health-beauty_" in n for n in ex), ex
    names.clear()
    run()
    assert names and all("_SP" not in n and "health-beauty" not in n for n in names)
    names.clear()
    at = run({ui.K_STATES: ["SP", "RJ", "MG"]})
    assert names and all("_3-states_" in n for n in names), names


def test_explorer_download_names_record_explorer_filters(monkeypatch):
    names: list[str] = []
    real = exports.export_filename
    monkeypatch.setattr(exports, "export_filename", lambda *a, **k: names.append(real(*a, **k)) or names[-1])
    at = run({ui.K_STATES: ["SP", "RJ"]})
    at.radio(key="rep_ex_status").set_value("Late").run()
    at.multiselect(key="rep_ex_states").set_value(["SP"]).run()
    no_exception(at)
    last = [n for n in names if "explorer_all" in n][-1]
    assert last.startswith("deliveryiq_explorer_all_SP_late_") and last.endswith(".csv")


def test_narrowing_the_sidebar_date_range_raises_no_widget_warning(monkeypatch):
    import datetime as dt
    import logging
    from streamlit.elements.lib import policies
    records: list[logging.LogRecord] = []

    class Grab(logging.Handler):
        def emit(self, record):
            records.append(record)

    logger = logging.getLogger(policies.__name__)       # Streamlit's own logger does not propagate to the root (caplog)
    handler = Grab(level=logging.WARNING)
    logger.addHandler(handler)
    try:
        # Streamlit logs this warning once per process: re-arm it so the test is independent of test order
        monkeypatch.setattr(policies, "_shown_default_value_warning", False)
        at = run()
        at.session_state[ui.K_DATE] = (dt.date(2018, 1, 1), dt.date(2018, 6, 30))
        at.run()
        no_exception(at)
        at.session_state[ui.K_DATE] = (dt.date(2017, 3, 1), dt.date(2017, 5, 31))
        at.run()
        no_exception(at)
    finally:
        logger.removeHandler(handler)
    assert not [r.getMessage() for r in records if "rep_ex_dates" in r.getMessage()], [r.getMessage() for r in records]
    assert not at.warning
    assert at.session_state["rep_ex_dates"] == (dt.date(2017, 3, 1), dt.date(2017, 5, 31))


def test_explorer_dates_follow_the_sidebar_until_the_user_narrows_them(orders):
    import datetime as dt
    at = run()
    full = (orders["purchase_date"].min().date(), orders["purchase_date"].max().date())
    assert tuple(at.session_state["rep_ex_dates"]) == full
    at.session_state[ui.K_DATE] = (dt.date(2018, 1, 1), dt.date(2018, 3, 31))
    at.run()
    assert tuple(at.session_state["rep_ex_dates"]) == (dt.date(2018, 1, 1), dt.date(2018, 3, 31))     # untouched: follows
    at.session_state[ui.K_DATE] = full
    at.run()
    assert tuple(at.session_state["rep_ex_dates"]) == full                                            # ... and widens again


def test_category_chart_excludes_the_unknown_bucket(orders):
    """'unknown' (no recorded category) is a data gap: the findings exclude it and so must the chart."""
    at = run({ui.K_STATES: ["SP"]})
    at.selectbox(key="rep_dl_chart_choice").set_value("Late rate by product category").run()
    no_exception(at)
    specs = [str(e.proto.spec) for e in at.get("plotly_chart")]
    spec = next(x for x in specs if "Late-delivery rate by product category" in x)
    assert "Unknown" not in spec and "n=550" not in spec
    assert "n=" in spec                                                                          # real categories are drawn
    assert (orders[orders.customer_state == "SP"].product_category == "unknown").sum() > 0       # the bucket does exist in SP


def test_model_evaluation_tab_callout_carries_the_honesty_wording():
    at = run()
    no_exception(at)
    page = plain(at)
    assert "Read this first: a modest signal, not an accurate predictor" in page          # Model evaluation tab callout title
    assert "never as a reliable forecast of an individual order" in page
