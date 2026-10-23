"""Tests for src/charts.py (Plotly figure builders).

Real data is used wherever the processed parquet / model metrics exist (skipped otherwise); empty,
``None`` and all-NaN inputs must always produce the friendly "no data" figure and never raise.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src import analytics as A
from src import charts as C
from src.config import GEO_DIR, MODEL_METRICS_PATH, ORDERS_DELIVERED_PATH, STATE_NAMES

needs_data = pytest.mark.skipif(not ORDERS_DELIVERED_PATH.exists(), reason="processed data not built")
needs_metrics = pytest.mark.skipif(not MODEL_METRICS_PATH.exists(), reason="model metrics not trained")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    if not ORDERS_DELIVERED_PATH.exists():
        pytest.skip("processed data not built")
    return pd.read_parquet(ORDERS_DELIVERED_PATH)


@pytest.fixture(scope="module")
def metrics() -> dict:
    if not MODEL_METRICS_PATH.exists():
        pytest.skip("model metrics not trained")
    return json.loads(MODEL_METRICS_PATH.read_text(encoding="utf-8"))


def _builders(d: pd.DataFrame) -> dict[str, callable]:
    """name -> zero-arg callable building each analytics-driven figure from ``d``."""
    return {
        "status_donut": lambda: C.status_donut(A.kpi_summary(d)),
        "status_donut_df": lambda: C.status_donut(d),
        "monthly_performance_chart": lambda: C.monthly_performance_chart(A.monthly_performance(d)),
        "monthly_late": lambda: C.monthly_performance_chart(A.monthly_performance(d), metric="late"),
        "monthly_on_time": lambda: C.monthly_performance_chart(A.monthly_performance(d), metric="on_time"),
        "duration_histogram": lambda: C.duration_histogram(d),
        "state_late_rate_bar": lambda: C.state_late_rate_bar(A.state_performance(d)),
        "category_late_rate_bar": lambda: C.category_late_rate_bar(A.category_performance(d)),
        "category_duration_bar": lambda: C.category_duration_bar(A.category_performance(d)),
        "category_freight_bar": lambda: C.category_freight_bar(A.category_cost_comparison(d)),
        "category_freight_ratio": lambda: C.category_freight_bar(A.category_cost_comparison(d), metric="mean_freight_ratio"),
        "weight_vs_duration": lambda: C.weight_vs_duration(
            A.weight_bins(d), A.scatter_sample(d, "total_weight_g", "delivery_duration_days")),
        "freight_vs_duration": lambda: C.freight_vs_duration(
            A.freight_bins(d), A.scatter_sample(d, "total_freight", "delivery_duration_days")),
        "weight_late_rate_bars": lambda: C.weight_late_rate_bars(A.weight_bins(d)),
        "freight_late_rate_bars": lambda: C.freight_late_rate_bars(A.freight_bins(d)),
        "processing_histogram": lambda: C.processing_histogram(d),
        "late_rate_by_processing_bucket": lambda: C.late_rate_by_processing_bucket(A.late_by_processing_bucket(d)),
        "state_pair_heatmap": lambda: C.state_pair_heatmap(A.state_matrix(d)),
        "state_pair_heatmap_dur": lambda: C.state_pair_heatmap(A.state_matrix(d), metric="mean_duration"),
        "state_pair_heatmap_n": lambda: C.state_pair_heatmap(A.state_matrix(d), metric="n_orders"),
        "state_pair_bar": lambda: C.state_pair_bar(A.state_pair_performance(d)),
        "state_pair_bar_n": lambda: C.state_pair_bar(A.state_pair_performance(d), metric="n_orders"),
        "state_pair_bar_dur": lambda: C.state_pair_bar(A.state_pair_performance(d), metric="mean_duration"),
        "state_comparison_chart": lambda: C.state_comparison_chart(A.state_performance(d), ["SP", "RJ", "BA"]),
        "delay_trend_chart": lambda: C.delay_trend_chart(A.delay_trend(d, "M")),
        "delay_trend_chart_roll": lambda: C.delay_trend_chart(A.delay_trend(d, "W"), window=4, freq_label="week"),
        "brazil_choropleth": lambda: C.brazil_choropleth(A.state_performance(d)),
        "brazil_choropleth_dur": lambda: C.brazil_choropleth(A.state_performance(d), "mean_duration"),
    }


# builders without cartesian axes (title only)
_NO_AXES = {"status_donut", "status_donut_df", "brazil_choropleth", "brazil_choropleth_dur"}


def _axis_titles(fig: go.Figure) -> list[str]:
    out = []
    for key, ax in fig.layout.to_plotly_json().items():
        if key.startswith(("xaxis", "yaxis")):
            t = (ax.get("title") or {}).get("text")
            if t:
                out.append(t)
    return out


def _bar_labels(fig: go.Figure) -> list:
    """y labels of all real (non legend-proxy) horizontal bars."""
    return [y for t in fig.data if isinstance(t, go.Bar) and t.y is not None for y in list(t.y) if y is not None]


def _title(fig: go.Figure) -> str:
    return (fig.layout.title.text or "").strip()


def _is_empty_fig(fig: go.Figure) -> bool:
    return any("No data" in (a.text or "") or "No " in (a.text or "") for a in fig.layout.annotations) and len(fig.data) == 0


# --------------------------------------------------------------------------- #
# Template / utilities
# --------------------------------------------------------------------------- #
def test_template_registered_and_palette_based():
    import plotly.io as pio
    assert "deliveryiq" in pio.templates
    assert C.CATEGORICAL[0].lower() == "#2563eb"
    # categorical colours never reuse the semantic green / amber / red
    sem = {"#16a34a", "#f59e0b", "#dc2626"}
    assert not (set(c.lower() for c in C.CATEGORICAL) & sem)


def test_empty_figure_basics():
    fig = C.empty_figure(title="T")
    assert isinstance(fig, go.Figure)
    assert _title(fig) == "T"
    assert "No data for the current filters" in fig.layout.annotations[0].text
    fig.to_json()


def test_pretty_helpers():
    assert C.pretty_feature_name("estimated_delivery_days").startswith("Promised delivery")
    assert C.pretty_feature_name("some_new_feature") == "Some new feature"
    assert C.pretty_feature_name("num__total_price") == "Order value (R$)"
    assert C.pretty_category("health_beauty") == "Health beauty"
    assert C.pretty_category("a_very_long_category_name_here", 10).endswith("…")


def test_figure_to_html_bytes():
    html = C.figure_to_html_bytes(C.empty_figure(title="x"))
    assert isinstance(html, bytes) and b"<html" in html.lower()


# --------------------------------------------------------------------------- #
# Analytics-driven builders: real data
# --------------------------------------------------------------------------- #
@needs_data
def test_all_builders_real_data(df):
    for name, build in _builders(df).items():
        fig = build()
        assert isinstance(fig, go.Figure), name
        assert len(fig.data) > 0, f"{name}: no traces for real data"
        assert _title(fig), f"{name}: missing title"
        if name not in _NO_AXES:
            assert _axis_titles(fig), f"{name}: missing axis titles"
        json.loads(fig.to_json())


@needs_data
def test_builders_filtered_slice_with_small_samples(df):
    sub = df[df["customer_state"].isin(["RR", "AC", "AP", "SP"])]
    for name, build in _builders(sub).items():
        fig = build()
        assert isinstance(fig, go.Figure), name
        fig.to_json()


@needs_data
def test_tiny_frame_never_raises(df):
    tiny = df.head(3)
    for name, build in _builders(tiny).items():
        fig = build()
        assert isinstance(fig, go.Figure), name
        fig.to_json()


@needs_data
def test_state_bar_flags_small_samples_and_shows_n(df):
    sp = A.state_performance(df)
    fig = C.state_late_rate_bar(sp, min_orders=100)
    names = [t.name for t in fig.data if t.name]
    assert any(n.startswith("Small sample") for n in names)
    labels = _bar_labels(fig)
    assert labels and all("n=" in lbl for lbl in labels)
    # every state is shown, sorted by rate within the reliable group
    assert len(labels) == sp["customer_state"].nunique()


@needs_data
def test_category_bars_respect_top_n_and_hide_small(df):
    cp = A.category_performance(df)
    fig = C.category_late_rate_bar(cp, top_n=10, min_orders=100)
    bars = _bar_labels(fig)
    assert len(bars) == 10
    fig2 = C.category_duration_bar(cp, top_n=5)
    assert len(_bar_labels(fig2)) == 5


@needs_data
def test_heatmap_masks_tiny_cells(df):
    mats = A.state_matrix(df)
    fig = C.state_pair_heatmap(mats, min_orders=100)
    z = np.asarray(fig.data[0].z, dtype=float)
    n = mats["n_orders"].reindex_like(mats["late_rate"])
    expected = mats["late_rate"].where(n >= 100)
    expected = expected.loc[expected.notna().any(axis=1), expected.notna().any(axis=0)]
    assert z.shape == expected.shape                      # empty rows / columns are trimmed
    assert np.array_equal(np.isfinite(z), expected.notna().to_numpy())
    # a higher threshold masks strictly more cells
    z2 = np.asarray(C.state_pair_heatmap(mats, min_orders=1000).data[0].z, dtype=float)
    assert np.isfinite(z2).sum() < np.isfinite(z).sum()


@needs_data
def test_heatmap_all_masked_gives_empty(df):
    fig = C.state_pair_heatmap(A.state_matrix(df), min_orders=10**9)
    assert _is_empty_fig(fig)


@needs_data
def test_delay_trend_window_adds_rolling_trace(df):
    base = C.delay_trend_chart(A.delay_trend(df, "M"), window=1)
    roll = C.delay_trend_chart(A.delay_trend(df, "M"), window=3)
    assert len(roll.data) > len(base.data) - 2
    assert any("rolling" in (t.name or "") for t in roll.data)


@needs_data
def test_gauge_variants(df):
    thr = {"moderate": 0.064, "high": 0.165}
    for p in (0.0, 0.03, 0.064, 0.11, 0.2, 0.9, 1.0):
        fig = C.probability_gauge(p, thr)
        assert isinstance(fig, go.Figure) and len(fig.data) >= 3
        fig.to_json()
    assert isinstance(C.probability_gauge(0.1), go.Figure)   # no thresholds
    for bad in (None, float("nan"), "x", -1):
        assert _is_empty_fig(C.probability_gauge(bad, thr))


# --------------------------------------------------------------------------- #
# Empty / None / NaN inputs
# --------------------------------------------------------------------------- #
@needs_data
def test_all_builders_empty_frames(df):
    for name, build in _builders(df.iloc[0:0]).items():
        fig = build()
        assert isinstance(fig, go.Figure), name
        assert _is_empty_fig(fig), f"{name}: expected the empty-state figure"
        assert _title(fig), f"{name}: empty figure needs a title"
        fig.to_json()


def test_none_and_garbage_inputs():
    calls = [
        lambda: C.status_donut(None), lambda: C.status_donut({}), lambda: C.status_donut(pd.DataFrame()),
        lambda: C.monthly_performance_chart(None), lambda: C.duration_histogram(None),
        lambda: C.duration_histogram(pd.DataFrame({"x": [1]})),
        lambda: C.state_late_rate_bar(None), lambda: C.state_late_rate_bar(pd.DataFrame()),
        lambda: C.category_late_rate_bar(None), lambda: C.category_duration_bar(None),
        lambda: C.category_freight_bar(None), lambda: C.weight_vs_duration(None),
        lambda: C.freight_vs_duration(None), lambda: C.weight_late_rate_bars(None),
        lambda: C.freight_late_rate_bars(None), lambda: C.processing_histogram(None),
        lambda: C.late_rate_by_processing_bucket(None), lambda: C.state_pair_heatmap(None),
        lambda: C.state_pair_heatmap({}), lambda: C.state_pair_bar(None),
        lambda: C.state_comparison_chart(None), lambda: C.delay_trend_chart(None),
        lambda: C.brazil_choropleth(None), lambda: C.model_comparison_chart(None),
        lambda: C.confusion_matrix_chart(None), lambda: C.roc_curve_chart(None),
        lambda: C.pr_curve_chart(None), lambda: C.calibration_chart(None),
        lambda: C.feature_importance_chart(None), lambda: C.risk_band_chart(None),
        lambda: C.model_comparison_chart({}), lambda: C.confusion_matrix_chart({}),
        lambda: C.roc_curve_chart({}), lambda: C.feature_importance_chart({}),
        lambda: C.risk_band_chart({}),
    ]
    for build in calls:
        fig = build()
        assert isinstance(fig, go.Figure)
        assert _is_empty_fig(fig)
        assert _title(fig)
        fig.to_json()


def test_all_nan_metric_is_empty():
    sp = pd.DataFrame({"customer_state": ["SP", "RJ"], "n_orders": [10, 20], "late_rate": [np.nan, np.nan],
                       "late_ci_low": [np.nan, np.nan], "late_ci_high": [np.nan, np.nan]})
    assert _is_empty_fig(C.state_late_rate_bar(sp))
    assert _is_empty_fig(C.brazil_choropleth(sp))
    assert _is_empty_fig(C.state_comparison_chart(sp.assign(n_orders=np.nan)))


def test_state_bar_without_ci_columns_and_zero_n():
    sp = pd.DataFrame({"customer_state": ["SP", "RJ", "AC"], "n_orders": [500, 300, 0], "late_rate": [0.05, 0.1, 0.0]})
    fig = C.state_late_rate_bar(sp)
    assert isinstance(fig, go.Figure) and len(fig.data) > 0
    fig.to_json()


def test_state_comparison_unknown_states_is_empty():
    sp = pd.DataFrame({"customer_state": ["SP"], "n_orders": [500], "late_rate": [0.05]})
    assert _is_empty_fig(C.state_comparison_chart(sp, ["ZZ"]))


# --------------------------------------------------------------------------- #
# Choropleth + geojson
# --------------------------------------------------------------------------- #
def test_vendored_geojson_is_valid():
    path = GEO_DIR / "brazil_states.geojson"
    assert path.exists(), "data/geo/brazil_states.geojson is vendored"
    assert path.stat().st_size < 1_500_000
    gj = json.loads(path.read_text(encoding="utf-8"))
    codes = {f["properties"]["sigla"] for f in gj["features"]}
    assert len(gj["features"]) == 27 and codes == set(STATE_NAMES)


@needs_data
def test_choropleth_uses_map_when_geojson_present(df):
    fig = C.brazil_choropleth(A.state_performance(df))
    assert any(isinstance(t, go.Choropleth) for t in fig.data)


@needs_data
@pytest.mark.parametrize("metric", ["late_rate", "mean_duration", "n_orders"])
def test_choropleth_fallback_when_geojson_missing(df, monkeypatch, tmp_path, metric):
    monkeypatch.setattr(C, "GEOJSON_PATH", tmp_path / "does_not_exist.geojson")
    fig = C.brazil_choropleth(A.state_performance(df), metric)
    assert isinstance(fig, go.Figure) and len(fig.data) > 0
    assert not any(isinstance(t, go.Choropleth) for t in fig.data)
    assert "unavailable" in (fig.layout.title.subtitle.text or "").lower()
    fig.to_json()


def test_choropleth_corrupt_geojson_falls_back(monkeypatch, tmp_path):
    bad = tmp_path / "bad.geojson"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(C, "GEOJSON_PATH", bad)
    sp = pd.DataFrame({"customer_state": ["SP", "RJ"], "n_orders": [500, 300], "late_rate": [0.05, 0.1]})
    fig = C.brazil_choropleth(sp)
    assert isinstance(fig, go.Figure) and not any(isinstance(t, go.Choropleth) for t in fig.data)


# --------------------------------------------------------------------------- #
# Model charts
# --------------------------------------------------------------------------- #
@needs_metrics
def test_model_charts_real_metrics(metrics):
    builders = {
        "model_comparison_chart": C.model_comparison_chart, "confusion_matrix_chart": C.confusion_matrix_chart,
        "roc_curve_chart": C.roc_curve_chart, "pr_curve_chart": C.pr_curve_chart,
        "calibration_chart": C.calibration_chart, "feature_importance_chart": C.feature_importance_chart,
        "risk_band_chart": C.risk_band_chart,
    }
    for name, build in builders.items():
        fig = build(metrics)
        assert isinstance(fig, go.Figure), name
        assert len(fig.data) > 0 or len(fig.layout.annotations) > 0
        assert not _is_empty_fig(fig), f"{name}: real metrics must draw something"
        assert _title(fig), name
        assert _axis_titles(fig), f"{name}: missing axis titles"
        json.loads(fig.to_json())


@needs_metrics
def test_model_chart_variants(metrics):
    for model in metrics["models"]:
        for window in ("validation", "test"):
            assert isinstance(C.confusion_matrix_chart(metrics, model, window), go.Figure)
        for build in (C.roc_curve_chart, C.pr_curve_chart, C.calibration_chart):
            build(metrics, model).to_json()
    C.feature_importance_chart(metrics, window="test", top_n=5).to_json()
    # confusion matrix counts and row percentages come straight from the metrics file
    sel = metrics["selected_model"]["name"]
    cm = np.asarray(metrics["models"][sel]["confusion_matrix_test"], dtype=float)
    fig = C.confusion_matrix_chart(metrics)
    z = np.asarray(fig.data[0].z, dtype=float)
    assert np.allclose(z, cm / cm.sum(axis=1, keepdims=True))
    assert np.allclose(np.asarray(fig.data[0].customdata, dtype=float), cm)
    texts = " ".join(a.text for a in fig.layout.annotations)
    assert f"{int(cm[1, 1]):,}" in texts


@needs_metrics
def test_model_comparison_values_match_metrics(metrics):
    fig = C.model_comparison_chart(metrics)
    test_bars = [t for t in fig.data if isinstance(t, go.Bar) and (t.name or "").startswith("Test") and t.y is not None
                 and any(v is not None for v in t.y)]
    pr_auc_trace = test_bars[0]
    expected = [m["test"]["pr_auc"] for m in metrics["models"].values()]
    assert np.allclose(np.asarray(pr_auc_trace.y, dtype=float), expected)


@needs_metrics
def test_risk_band_chart_n_and_ci_present(metrics):
    fig = C.risk_band_chart(metrics)
    bars = [t for t in fig.data if isinstance(t, go.Bar)]
    assert len(bars) == 2 and all(b.error_y is not None and b.error_y.array is not None for b in bars)
    assert any("n=" in str(x) for b in bars for x in b.x)


@needs_metrics
def test_model_charts_with_missing_optional_sections(metrics):
    stripped = copy.deepcopy(metrics)
    for k in ("risk_bands", "permutation_importance", "split", "feature_importance"):
        stripped.pop(k, None)
    for entry in stripped["models"].values():
        for k in list(entry):
            if k.startswith(("calibration_", "pr_curve_", "roc_curve_", "test_ci95", "confusion_matrix_")):
                entry.pop(k)
        entry.pop("threshold", None)
    stripped.pop("selected_model", None)
    for build in (C.model_comparison_chart, C.confusion_matrix_chart, C.roc_curve_chart, C.pr_curve_chart,
                  C.calibration_chart, C.feature_importance_chart, C.risk_band_chart):
        fig = build(stripped)
        assert isinstance(fig, go.Figure)
        fig.to_json()
    # comparison still works without CI / reference data
    assert not _is_empty_fig(C.model_comparison_chart(stripped))
    # only the validation window available
    only_val = copy.deepcopy(metrics)
    only_val["risk_bands"].pop("test")
    assert not _is_empty_fig(C.risk_band_chart(only_val))
    for entry in only_val["models"].values():
        entry.pop("roc_curve_test", None)
    assert not _is_empty_fig(C.roc_curve_chart(only_val))


@needs_metrics
def test_model_charts_unknown_model_name_is_empty(metrics):
    assert _is_empty_fig(C.confusion_matrix_chart(metrics, "no_such_model"))
    assert _is_empty_fig(C.roc_curve_chart(metrics, "no_such_model"))


@needs_metrics
def test_model_charts_malformed_values(metrics):
    bad = copy.deepcopy(metrics)
    sel = bad["selected_model"]["name"]
    bad["models"][sel]["confusion_matrix_test"] = [[1, 2, 3]]
    bad["models"][sel]["roc_curve_test"] = {"fpr": [0, 1], "tpr": [0]}
    bad["permutation_importance"]["items"] = [{"feature": "x"}]
    assert _is_empty_fig(C.confusion_matrix_chart(bad))
    assert isinstance(C.roc_curve_chart(bad), go.Figure)
    assert _is_empty_fig(C.feature_importance_chart(bad))


@needs_data
def test_heatmap_masked_cells_are_visible_and_cells_carry_their_values(df):
    mats = A.state_matrix(df)
    fig = C.state_pair_heatmap(mats, min_orders=100)
    assert fig.layout.plot_bgcolor == C.MASKED_CELL != C.MASKED       # masked cells are a visible grey, not 'missing' white
    tr = fig.data[0]
    assert tr.texttemplate == "%{z:.1%}"                              # every drawn cell shows its value
    dur = C.state_pair_heatmap(mats, metric="mean_duration", min_orders=100).data[0]
    assert dur.texttemplate == "%{z:.1f}"


def test_legends_sit_above_the_top_axis_tick():
    d = pd.DataFrame({"purchase_year_month": ["2018-01", "2018-02", "2018-03"], "n_orders": [500, 600, 700],
                      "n_late": [30, 40, 50], "late_rate": [0.06, 0.067, 0.071], "on_time_rate": [0.94, 0.933, 0.929],
                      "late_ci_low": [0.04, 0.05, 0.05], "late_ci_high": [0.08, 0.09, 0.09], "mean_duration": [11.0, 12.0, 13.0],
                      "reliable": [True, True, True]})
    fig = C.monthly_performance_chart(d, min_orders=100)
    assert fig.layout.legend.y > 1.0 and fig.layout.legend.yanchor == "bottom"
