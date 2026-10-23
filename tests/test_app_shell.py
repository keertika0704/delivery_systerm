"""Tests for the Streamlit app shell (app.py, src/ui.py, temporary page stubs)."""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import config, ui
from src.analytics import apply_filters

ROOT = Path(__file__).resolve().parents[1]
PAGE_FILES = {
    "1_Overview.py": "overview",
    "2_Delay_Analysis.py": "delay-analysis",
    "3_Regional_Intelligence.py": "regional-intelligence",
    "4_Delay_Prediction.py": "delay-prediction",
    "5_Reports_Insights.py": "reports-insights",
}
TIMEOUT = 90

needs_data = pytest.mark.skipif(not config.ORDERS_DELIVERED_PATH.is_file(),
                                reason="processed data missing: run .venv/bin/python -m src.preprocessing")


@pytest.fixture(autouse=True)
def _clear_caches():
    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


def _html(at: AppTest) -> str:
    """All st.html / st.markdown payloads on the page (sidebar included) as one string."""
    parts = [str(e.value) for e in at.get("html")]
    parts += [str(m.value) for m in at.markdown]
    return "\n".join(parts)


def _probe_script():
    """Script body for AppTest.from_function: stores get_filtered_orders() results in session state."""
    import streamlit as st
    from src import ui

    df_all, df, flt = ui.get_filtered_orders()
    st.session_state["probe"] = {
        "n_all": len(df_all), "n": len(df), "flt": flt,
        "states": sorted(df["customer_state"].unique().tolist()),
        "min": str(df["purchase_date"].min()), "max": str(df["purchase_date"].max()),
    }


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [None, np.nan, float("inf"), -np.inf, pd.NA, "abc", pd.NaT])
def test_formatters_render_dash_for_missing(bad):
    for fn in (ui.fmt_int, ui.fmt_pct, ui.fmt_days, ui.fmt_money_brl, ui.fmt_num, ui.fmt_pp):
        assert fn(bad) == "—"
    assert ui.fmt_n(bad) == "n = —"


def test_formatters_zero_and_large_numbers():
    assert ui.fmt_int(0) == "0"
    assert ui.fmt_int(96470) == "96,470"
    assert ui.fmt_int(1234567890) == "1,234,567,890"
    assert ui.fmt_int(np.int64(12)) == "12"
    assert ui.fmt_pct(0) == "0.0%"
    assert ui.fmt_pct(0.0677, 1) == "6.8%"
    assert ui.fmt_pct(1) == "100.0%"
    assert ui.fmt_days(0) == "0.0 days"
    assert ui.fmt_days(12.345) == "12.3 days"
    assert ui.fmt_money_brl(0) == "R$ 0.00"
    assert ui.fmt_money_brl(1234567.891) == "R$ 1,234,567.89"
    assert ui.fmt_money_brl(-5) == "-R$ 5.00"
    assert ui.fmt_pp(0.012) == "+1.2 pp" and ui.fmt_pp(-0.03) == "-3.0 pp"
    assert ui.fmt_n(1234) == "n = 1,234"


def test_labels():
    assert ui.state_label("SP") == "SP — São Paulo"
    assert ui.state_label("ZZ") == "ZZ"
    assert ui.category_label("health_beauty") == "Health beauty"


def test_html_components_escape_user_text():
    html = ui._kpi_html("<b>x</b>", "<script>alert(1)</script>", sublabel="<i>", icon="<img>")
    assert "<script>" not in html and "&lt;script&gt;" in html and "<b>x</b>" not in html
    assert "<img>" not in html


def test_icons_are_defined_for_every_name_and_css_is_valid():
    css = ui._css()
    for name in ui._ICONS:
        assert f".diq-i-{name}{{" in css
    for key in config.PALETTE:
        assert f"__{key}__" not in css


# --------------------------------------------------------------------------- #
# Date normalisation guards
# --------------------------------------------------------------------------- #
LO, HI = date(2017, 1, 1), date(2018, 8, 29)


def test_normalise_dates_edge_cases():
    n = ui._normalise_dates
    assert n(None, LO, HI) == (LO, HI, True)
    assert n((), LO, HI) == (LO, HI, True)
    assert n((date(2017, 5, 1), date(2017, 6, 1)), LO, HI) == (date(2017, 5, 1), date(2017, 6, 1), True)
    # reversed range is swapped
    assert n((date(2017, 6, 1), date(2017, 5, 1)), LO, HI)[:2] == (date(2017, 5, 1), date(2017, 6, 1))
    # a single date (range being picked) runs to the end of the data and is flagged incomplete
    assert n((date(2017, 5, 1),), LO, HI) == (date(2017, 5, 1), HI, False)
    assert n(date(2017, 5, 1), LO, HI) == (date(2017, 5, 1), HI, False)
    # one-day range is valid
    assert n((date(2017, 5, 1), date(2017, 5, 1)), LO, HI)[:2] == (date(2017, 5, 1), date(2017, 5, 1))
    # out-of-span values are clipped, entirely-outside ranges fall back to the full span
    assert n((date(2010, 1, 1), date(2030, 1, 1)), LO, HI)[:2] == (LO, HI)
    assert n((date(2030, 1, 1), date(2031, 1, 1)), LO, HI)[:2] == (LO, HI)
    # timestamps / strings are accepted
    assert n((pd.Timestamp("2017-05-01"), "2017-06-01"), LO, HI)[:2] == (date(2017, 5, 1), date(2017, 6, 1))


# --------------------------------------------------------------------------- #
# Shell structure
# --------------------------------------------------------------------------- #
def test_five_navigation_targets_exist_and_are_declared():
    src = (ROOT / "app.py").read_text()
    for fn, url_path in PAGE_FILES.items():
        assert (ROOT / "pages" / fn).is_file(), fn
        assert f'"pages/{fn}"' in src and f'url_path="{url_path}"' in src
    assert 'position="hidden"' in src and "default=True" in src
    assert (ROOT / ".streamlit" / "config.toml").is_file()


@needs_data
def test_app_runs_without_exception_and_renders_shell():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT).run()
    assert not at.exception, [e.value for e in at.exception]
    html = _html(at)
    assert ui.esc(config.APP_NAME) in html and ui.esc(config.APP_TAGLINE) in html
    assert re.search(r"of [\d,]+ orders match", html)
    assert "Dataset &amp; model" in html
    assert at.sidebar.date_input(key=ui.K_DATE) is not None
    assert at.sidebar.multiselect(key=ui.K_STATES) is not None
    assert at.sidebar.multiselect(key=ui.K_CATEGORIES) is not None


@needs_data
@pytest.mark.parametrize("fn", list(PAGE_FILES))
def test_each_page_runs_standalone(fn):
    at = AppTest.from_file(str(ROOT / "pages" / fn), default_timeout=TIMEOUT).run()
    assert not at.exception, [e.value for e in at.exception]


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #
@needs_data
def test_standalone_defaults_apply_no_filters():
    at = AppTest.from_function(_probe_script, default_timeout=TIMEOUT).run()
    assert not at.exception
    p = at.session_state["probe"]
    assert p["n"] == p["n_all"] > 90_000
    assert p["flt"]["active"] is False and p["flt"]["states"] == []


@needs_data
def test_state_filter_matches_pandas():
    df = ui._read_orders(str(config.ORDERS_DELIVERED_PATH), 0)
    at = AppTest.from_function(_probe_script, default_timeout=TIMEOUT)
    at.session_state[ui.K_STATES] = ["SP"]
    at.run()
    assert not at.exception
    p = at.session_state["probe"]
    assert p["n"] == int((df["customer_state"] == "SP").sum())
    assert p["states"] == ["SP"] and p["flt"]["active"] is True
    assert p["flt"]["n_filtered"] == p["n"] and p["flt"]["n_total"] == len(df)


@needs_data
def test_combined_filters_match_apply_filters():
    df = ui._read_orders(str(config.ORDERS_DELIVERED_PATH), 0)
    cat = df["product_category"].value_counts().index[0]
    rng = (date(2017, 3, 1), date(2017, 9, 30))
    at = AppTest.from_function(_probe_script, default_timeout=TIMEOUT)
    at.session_state[ui.K_STATES] = ["RJ", "MG"]
    at.session_state[ui.K_CATEGORIES] = [cat]
    at.session_state[ui.K_DATE] = rng
    at.run()
    expected = apply_filters(df, date_range=rng, states=["RJ", "MG"], categories=[cat])
    assert at.session_state["probe"]["n"] == len(expected) > 0


@needs_data
def test_invalid_stored_filter_values_are_sanitised():
    """Unknown states / reversed or one-element dates must not crash and fall back sensibly."""
    at = AppTest.from_function(_probe_script, default_timeout=TIMEOUT)
    at.session_state[ui.K_STATES] = ["XX", "SP"]
    at.session_state[ui.K_DATE] = (date(2018, 1, 1), date(2017, 1, 1))  # reversed
    at.run()
    assert not at.exception
    p = at.session_state["probe"]
    assert p["flt"]["states"] == ["SP"]
    assert p["flt"]["date_range"] == (date(2017, 1, 1), date(2018, 1, 1))


@needs_data
def test_empty_filter_result_is_handled_by_page_stub():
    at = AppTest.from_file(str(ROOT / "pages" / "1_Overview.py"), default_timeout=TIMEOUT)
    at.session_state[ui.K_DATE] = (date(2016, 9, 15), date(2016, 9, 15))
    at.session_state[ui.K_STATES] = ["AC"]
    at.session_state[ui.K_CATEGORIES] = ["unknown"]
    at.run()
    assert not at.exception
    html = _html(at)
    assert "No orders match" in html


@needs_data
def test_sidebar_filters_in_app_and_reset():
    df = ui._read_orders(str(config.ORDERS_DELIVERED_PATH), 0)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT).run()
    at.sidebar.multiselect(key=ui.K_STATES).select("SP").run()
    assert not at.exception
    n_sp = int((df["customer_state"] == "SP").sum())
    assert f"{n_sp:,}" in _html(at)                       # counter + KPI stub show the pandas count
    assert at.session_state[ui.K_STATES] == ["SP"]
    at.sidebar.button(key="flt_reset").click().run()
    assert not at.exception
    assert at.session_state[ui.K_STATES] == [] and at.session_state[ui.K_CATEGORIES] == []
    lo, hi = ui._bounds(df)
    assert tuple(at.session_state[ui.K_DATE]) == (lo, hi)
    assert f"{len(df):,}" in _html(at)


@needs_data
def test_zero_match_counter_shows_warning_in_sidebar():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    at.session_state[ui.K_DATE] = (date(2016, 9, 15), date(2016, 9, 15))
    at.session_state[ui.K_STATES] = ["AC"]
    at.run()
    assert not at.exception
    sidebar_html = "\n".join(str(e.value) for e in at.sidebar.get("html"))
    assert "0 of" in sidebar_html and "No orders match these filters" in sidebar_html


# --------------------------------------------------------------------------- #
# Missing data / model
# --------------------------------------------------------------------------- #
def test_missing_data_shows_friendly_setup_page(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ORDERS_DELIVERED_PATH", tmp_path / "nope.parquet")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT).run()
    assert not at.exception, [e.value for e in at.exception]
    html = _html(at)
    assert "Processed data not found" in html
    assert any("src.preprocessing" in c.value for c in at.code)
    assert config.KAGGLE_URL in html


def test_missing_data_on_a_page_stops_with_friendly_error(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ORDERS_DELIVERED_PATH", tmp_path / "nope.parquet")
    at = AppTest.from_file(str(ROOT / "pages" / "2_Delay_Analysis.py"), default_timeout=TIMEOUT).run()
    assert not at.exception
    assert "Processed data not found" in _html(at)
    assert any("src.preprocessing" in c.value for c in at.code)


@needs_data
def test_missing_model_does_not_break_the_app(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "nope.joblib")
    monkeypatch.setattr(config, "MODEL_METRICS_PATH", tmp_path / "nope.json")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT).run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Not trained" in _html(at)
    assert at.session_state["model_missing"] is True
    assert "src.train_model" in (at.session_state["model_error"] or "")


def test_require_model_stops_with_training_command(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "nope.joblib")

    def script():
        from src import ui
        ui.require_model()

    at = AppTest.from_function(script, default_timeout=TIMEOUT).run()
    assert not at.exception
    assert any("src.train_model" in c.value for c in at.code)


@needs_data
def test_model_and_metrics_load_when_present():
    if not (config.MODEL_PATH.is_file() and config.MODEL_METRICS_PATH.is_file()):
        pytest.skip("model not trained")
    def script():
        import streamlit as st
        from src import ui
        st.session_state["out"] = (ui.get_model_bundle() is not None, "models" in (ui.get_model_metrics() or {}),
                                   ui.get_quality_report().get("n_eligible", 0))

    at = AppTest.from_function(script, default_timeout=TIMEOUT).run()
    assert not at.exception
    ok_model, ok_metrics, n_eligible = at.session_state["out"]
    assert ok_model and ok_metrics and n_eligible > 0
