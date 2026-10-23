"""Unit tests for src/exports.py (byte-builders, explorer helpers, model-evaluation helpers)."""
from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest

from src import analytics as an
from src import config, exports, insights as ins

needs_data = pytest.mark.skipif(not config.ORDERS_DELIVERED_PATH.is_file(), reason="processed data missing")
needs_metrics = pytest.mark.skipif(not config.MODEL_METRICS_PATH.is_file(), reason="model metrics missing")


@pytest.fixture(scope="module")
def orders() -> pd.DataFrame:
    return pd.read_parquet(config.ORDERS_DELIVERED_PATH)


@pytest.fixture(scope="module")
def metrics() -> dict:
    return json.loads(config.MODEL_METRICS_PATH.read_text(encoding="utf-8"))


def _toy() -> pd.DataFrame:
    ts = pd.to_datetime(["2018-01-02 10:00:00", "2018-01-03 11:30:00", "2018-02-01 09:15:00"])
    return pd.DataFrame({
        "order_id": ["a1", "b2", "c3"],
        "order_purchase_timestamp": ts,
        "order_estimated_delivery_date": ts + pd.Timedelta(days=10),
        "order_delivered_customer_date": ts + pd.Timedelta(days=8),
        "purchase_date": ts.normalize(),
        "is_late": pd.array([0, 1, pd.NA], dtype="Int8"),
        "customer_state": pd.Categorical(["SP", "RJ", "SP"]),
        "customer_city": pd.Categorical(["são paulo", "niterói", "são paulo"]),
        "product_category": pd.Categorical(["toys", "auto", "toys"]),
        "total_price": [10.5, 20.0, np.nan],
        # columns that must never be exported:
        "order_status": "delivered", "approval_missing": 0, "customer_zip": "123",
    })


# --------------------------------------------------------------------------- #
# whitelist / CSV
# --------------------------------------------------------------------------- #
def test_whitelist_has_no_identifiers_other_than_order_id():
    banned = ("customer_id", "customer_unique_id", "seller_id", "zip", "review_comment", "order_status",
              "approval_missing", "carrier_date_invalid", "exclusion_reason", "analysis_eligible")
    for col in exports.EXPORT_COLUMNS:
        assert col == "order_id" or not any(b in col for b in banned), col
    assert not set(exports.EXPORT_COLUMNS) & set(config.LEAKAGE_COLUMNS) - {
        "delivery_delay_days", "delivery_duration_days", "processing_days", "is_late", "review_score",
        "order_delivered_customer_date"}
    assert len(exports.data_dictionary()) == len(exports.EXPORT_COLUMNS)


def test_export_frame_rejects_non_whitelisted_columns():
    with pytest.raises(ValueError):
        exports.export_frame(_toy(), ["order_id", "customer_zip"])
    out = exports.export_frame(_toy())
    assert "customer_zip" not in out.columns and "order_status" not in out.columns
    assert list(out.columns) == [c for c in exports.EXPORT_COLUMNS if c in out.columns]


def test_delivery_status_is_derived_from_is_late():
    out = exports.export_frame(_toy(), ["order_id", "delivery_status"])
    assert out["delivery_status"].tolist()[:2] == ["on_time", "late"]
    assert pd.isna(out["delivery_status"].iloc[2])


def test_csv_round_trip_toy_with_accents():
    df = _toy()
    data = exports.shipments_csv_bytes(df)
    assert data.startswith(b"\xef\xbb\xbf")              # BOM so Excel shows accents
    text = data.decode("utf-8-sig")
    assert "são paulo" in text and "niterói" in text
    back = exports.read_csv_bytes(data, parse_dates=["order_purchase_timestamp"])
    exp = exports.export_frame(df)
    assert list(back.columns) == list(exp.columns)
    assert not any(c.startswith("Unnamed") for c in back.columns)    # no index column
    assert len(back) == len(exp)
    assert back["order_id"].tolist() == ["a1", "b2", "c3"]
    assert back["customer_city"].tolist() == ["são paulo", "niterói", "são paulo"]
    assert back["order_purchase_timestamp"].tolist() == exp["order_purchase_timestamp"].tolist()
    assert back["is_late"].tolist()[:2] == [0, 1] and pd.isna(back["is_late"].iloc[2])
    assert back["total_price"].iloc[:2].tolist() == [10.5, 20.0] and pd.isna(back["total_price"].iloc[2])


def test_empty_frame_gives_header_only_csv():
    empty = _toy().iloc[0:0]
    back = exports.read_csv_bytes(exports.shipments_csv_bytes(empty))
    assert len(back) == 0
    assert list(back.columns) == list(exports.export_frame(empty).columns) and back.shape[1] > 5
    reg = ins.regional_report(empty)
    back = exports.read_csv_bytes(exports.regional_csv_bytes(reg))
    assert len(back) == 0 and list(back.columns) == list(reg.columns)


@needs_data
def test_csv_round_trip_real_filtered_frame(orders):
    df = an.apply_filters(orders, states=["SP", "RJ"], categories=["health_beauty", "toys"])
    assert len(df) > 100
    back = exports.read_csv_bytes(exports.shipments_csv_bytes(df), parse_dates=[
        "order_purchase_timestamp", "order_estimated_delivery_date", "order_delivered_customer_date"])
    exp = exports.export_frame(df)
    assert len(back) == len(df) and list(back.columns) == list(exp.columns)
    assert back["order_id"].tolist() == df["order_id"].tolist()
    assert back["customer_state"].tolist() == df["customer_state"].astype(str).tolist()
    assert back["order_purchase_timestamp"].tolist() == df["order_purchase_timestamp"].tolist()
    assert back["is_late"].sum() == int(df["is_late"].sum())
    np.testing.assert_allclose(back["total_price"].to_numpy(), df["total_price"].to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(back["delivery_duration_days"].to_numpy(), df["delivery_duration_days"].to_numpy(), rtol=1e-12)
    assert set(back["delivery_status"]) <= {"late", "on_time"}
    assert (back["delivery_status"] == "late").sum() == back["is_late"].sum()


@needs_data
def test_regional_csv_equals_regional_report(orders):
    df = an.apply_filters(orders, states=["SP", "MG", "BA"])
    rep = ins.regional_report(df, 100, 0.05)
    back = exports.read_csv_bytes(exports.regional_csv_bytes(rep))
    assert list(back.columns) == list(rep.columns) and len(back) == len(rep)
    assert back["customer_state"].tolist() == rep["customer_state"].tolist()
    assert "São Paulo" in back["state_name"].tolist()
    np.testing.assert_allclose(back["late_rate"], rep["late_rate"], rtol=1e-12)
    assert back["n_orders"].tolist() == rep["n_orders"].tolist()
    assert back["flag"].tolist() == rep["flag"].tolist()


@needs_data
def test_summary_contains_kpis_and_html_is_safe(orders):
    df = an.apply_filters(orders, states=["SP"])
    k = an.kpi_summary(df)
    md = ins.summary_report(df, "customer state: SP", ins.generate_insights(df))
    assert f"{k['n_orders']:,}" in md and f"{100 * k['late_rate']:.1f}%" in md
    html = exports.summary_html_bytes(md + "\n<script>alert(1)</script>").decode("utf-8")
    assert "<h1>" in html and "<table>" in html and "<script>" not in html and "&lt;script&gt;" in html
    assert html.startswith("<!doctype html>") and 'charset="utf-8"' in html


def test_markdown_to_html_basics():
    html = exports.markdown_to_html("# T\n\ntext **b** and `c`\n\n- one\n- two\n\n1. x\n2. y\n\n| a | b |\n|---|---|\n| 1 | 2 |\n")
    assert "<h1>T</h1>" in html and "<strong>b</strong>" in html and "<code>c</code>" in html
    assert html.count("<li>") == 4 and "<ul>" in html and "<ol>" in html
    assert "<th>a</th>" in html and "<td>2</td>" in html


def test_human_size_and_filename():
    assert exports.human_size(500) == "500 B" and exports.human_size(1536) == "1.5 KB"
    assert exports.human_size(5 * 1024 ** 2) == "5.0 MB" and exports.human_size(float("nan")) == "—"
    assert exports.export_filename("Late rate by state", ("2017-01-01", "2017-02-01"), "html") == \
        "deliveryiq_late_rate_by_state_2017-01-01_to_2017-02-01.html"
    assert exports.export_filename("x") == "deliveryiq_x.csv"


# --------------------------------------------------------------------------- #
# explorer helpers
# --------------------------------------------------------------------------- #
def test_explorer_view_paging_and_sorting():
    df = pd.DataFrame({"a": [3, 1, 2, np.nan, 5, 4, 7], "b": list("gfedcba")})
    v, info = exports.explorer_view(df, "a", True, 1, 3)
    assert v["a"].tolist() == [1, 2, 3] and info["n_pages"] == 3 and info["start"] == 1 and info["end"] == 3
    v, info = exports.explorer_view(df, "a", True, 3, 3)
    assert len(v) == 1 and pd.isna(v["a"].iloc[0])               # NaN sorts last
    v, info = exports.explorer_view(df, "a", False, 99, 3)           # page clamped
    assert info["page"] == 3 and info["total"] == 7
    v, info = exports.explorer_view(df.iloc[0:0], "a", True, 1, 3)
    assert len(v) == 0 and info == {"total": 0, "page": 1, "n_pages": 1, "page_size": 3, "start": 0, "end": 0}


@needs_data
def test_explorer_filters_compose_like_pandas(orders):
    sidebar = an.apply_filters(orders, date_range=(pd.Timestamp("2017-06-01"), pd.Timestamp("2018-03-31")), states=["SP", "RJ", "MG"])
    got = exports.explorer_filter(sidebar, date_range=(pd.Timestamp("2017-09-01"), pd.Timestamp("2017-12-31")),
                                  states=["SP"], categories=["health_beauty", "toys"], status="late", search=None)
    exp = orders[(orders.purchase_date >= "2017-09-01") & (orders.purchase_date <= "2017-12-31")
                 & (orders.customer_state == "SP") & orders.product_category.isin(["health_beauty", "toys"])
                 & (orders.is_late == 1)]
    assert len(got) == len(exp) > 0 and set(got["order_id"]) == set(exp["order_id"])
    on_time = exports.explorer_filter(sidebar, status="on_time")
    assert len(on_time) == int((sidebar.is_late == 0).sum())
    assert len(exports.explorer_filter(sidebar, status="all")) == len(sidebar)
    # search matches order id, city and category (case-insensitive), layered on the other filters
    some = sidebar["order_id"].iloc[5]
    assert exports.explorer_filter(sidebar, search=some[:12].upper())["order_id"].tolist() == [some]
    s = exports.explorer_filter(sidebar, states=["SP"], search="health")
    assert len(s) and (s["customer_state"] == "SP").all()
    hit = sum(sidebar[c].astype(str).str.contains("health", case=False, regex=False) for c in
              ("order_id", "customer_city", "seller_city", "product_category")) > 0
    assert len(s) == int(((sidebar.customer_state == "SP") & hit).sum())


# --------------------------------------------------------------------------- #
# insights presentation
# --------------------------------------------------------------------------- #
@needs_data
def test_insight_helpers_cover_every_generated_kind(orders):
    items = ins.generate_insights(orders)
    for i in items:
        assert exports.insight_type_label(i) in (exports.OBSERVED, exports.ASSOCIATION, exports.DATA_NOTE)
        pairs = exports.insight_metric_pairs(i)
        assert all(isinstance(k, str) and isinstance(v, str) and v for k, v in pairs)
        if i.kind not in ("insufficient_data",):
            assert pairs, i.kind
    order = exports.sort_insights(items)
    assert order[0].kind == "overview" and len(order) == len(items)
    sev = [{"alert": 0, "watch": 1, "info": 2}[i.severity] for i in order[1:] if i.kind != "insufficient_data"]
    assert sev == sorted(sev)
    assert exports.insight_type_label(next(i for i in items if i.kind == "region_test")) == exports.ASSOCIATION
    empty = ins.generate_insights(orders.iloc[0:0])
    assert exports.insight_type_label(empty[0]) == exports.DATA_NOTE


# --------------------------------------------------------------------------- #
# model evaluation helpers
# --------------------------------------------------------------------------- #
@needs_metrics
def test_split_and_model_tables_match_json(metrics):
    t = exports.split_table(metrics)
    assert t["Window"].tolist() == ["Train", "Validation", "Test"]
    for w, row in zip(("train", "validation", "test"), t.itertuples()):
        assert row[5] == metrics["split"][w]["n"] and row[7] == pytest.approx(metrics["split"][w]["late_rate"])
    mt = exports.model_metrics_table(metrics)
    assert len(mt) == 2 * len(metrics["models"])
    sel = metrics["selected_model"]["name"]
    r = mt[(mt["Selected"] == "yes") & (mt["Window"] == "Test")].iloc[0]
    assert r["ROC-AUC"] == pytest.approx(metrics["models"][sel]["test"]["roc_auc"])
    assert r["No-skill PR-AUC"] == pytest.approx(metrics["split"]["test"]["late_rate"])
    assert (mt[mt["Model"].str.contains("Dummy")]["ROC-AUC"] == 0.5).all()
    bt = exports.risk_band_table(metrics)
    assert bt["Band"].tolist() == ["Low", "Moderate", "High"] * 2
    assert bt[bt["Window"] == "Test"]["Orders (n)"].sum() == metrics["split"]["test"]["n"]


@needs_metrics
def test_confusion_reading_numbers(metrics):
    sel = metrics["selected_model"]["name"]
    (tn, fp), (fn, tp) = metrics["models"][sel]["confusion_matrix_test"]
    c = exports.confusion_counts(metrics)
    assert (c["tn"], c["fp"], c["fn"], c["tp"]) == (tn, fp, fn, tp)
    assert c["recall"] == pytest.approx(tp / (tp + fn)) and c["precision"] == pytest.approx(tp / (tp + fp))
    text = exports.confusion_reading(metrics)
    for needle in (f"{tp + fp:,}", f"{tp:,}", f"{tp + fn:,}", f"{fn:,}", f"{fp:,}"):
        assert needle in text
    assert exports.confusion_reading({}) is None and exports.confusion_counts(None) is None


@needs_metrics
def test_headline_and_summaries_are_honest(metrics):
    h = exports.model_headline(metrics)
    sel = metrics["selected_model"]["name"]
    assert h["test_roc_auc"] == pytest.approx(metrics["models"][sel]["test"]["roc_auc"])
    assert h["test_lift"] == pytest.approx(h["test_pr_auc"] / h["test_late_rate"])
    summary = " ".join(exports.honest_summary(metrics))
    assert f"{h['test_roc_auc']:.2f}" in summary and "validation" in summary.lower()
    assert "Moderate and High" in summary                       # test bands overlap in the stored diagnostics
    assert "overstate" not in summary.lower()
    guidance = " ".join(exports.metric_guidance_points(metrics))
    assert f"{100 * (1 - h['test_late_rate']):.1f}%" in guidance and "misleading" in guidance


def test_helpers_tolerate_missing_metrics():
    assert exports.model_headline(None) is None and exports.honest_summary({}) == []
    assert exports.metric_guidance_points(None) == []
    assert exports.split_table(None).empty and exports.model_metrics_table({}).empty and exports.risk_band_table(None).empty
    assert exports.split_notes(None) == [] and exports.selected_model_name(None) is None


def test_signal_phrase_never_overstates():
    assert exports.signal_phrase(0.62)[0] == "a modest signal"
    assert "chance" in exports.signal_phrase(0.51)[1]
    assert exports.signal_phrase(0.9)[1].endswith("far from perfect")
    assert exports.signal_phrase(None)[0] == "an unknown signal"


# --------------------------------------------------------------------------- #
# p-value formatting, file names, model section of the summary report
# --------------------------------------------------------------------------- #
def test_underflowed_p_values_are_never_printed_as_numbers():
    from src import statistical_analysis as sa
    for p in (0.0, sa.P_FLOOR, 21 * sa.P_FLOOR, 1e-301):      # 21 x floor = a Bonferroni-adjusted underflow
        assert exports._p(p) == "p < 0.001" and ins._p(p) == "p < 0.001", p
    # any p below 0.001 reads "p < 0.001": scientific notation ("p = 2.5e-07") is jargon for a non-programmer
    assert exports._p(2.5e-7) == "p < 0.001" and ins._p(2.5e-7) == "p < 0.001"
    assert "e-" not in exports._p(5.1e-98) and "e-" not in ins._p(5.1e-98)
    assert exports._p(0.0123) == "p = 0.012" and exports._p(float("nan")) == "p = n/a" and ins._p(None) == "p = n/a"


@needs_data
def test_reports_and_cards_contain_no_floating_point_floor(orders, metrics):
    items = ins.generate_insights(orders, config.DEFAULT_MIN_ORDERS)
    text = ins.summary_report(orders, "all", items, metrics) + " ".join(i.text for i in items)
    text += " ".join(v for i in items for _, v in exports.insight_metric_pairs(i))
    assert "e-30" not in text and "p < 0.001" in text


def test_filter_slug_and_export_filename_record_the_filters():
    assert exports.filter_slug() == "" and exports.filter_slug([], []) == ""
    assert exports.filter_slug(["SP"]) == "SP" and exports.filter_slug(["SP", "RJ"]) == "RJ-SP"
    assert exports.filter_slug(["SP", "RJ", "MG"]) == "3-states"
    assert exports.filter_slug(None, ["health_beauty"]) == "health-beauty"
    assert exports.filter_slug(None, ["a", "b"]) == "2-categories"
    assert exports.filter_slug(["SP"], ["toys"], ["late"]) == "SP_toys_late"
    rng = ("2018-01-01", "2018-06-30")
    full = exports.export_filename("shipments", rng)
    sp = exports.export_filename("shipments", rng, states=["SP"])
    assert sp == "deliveryiq_shipments_SP_2018-01-01_to_2018-06-30.csv" and sp != full
    assert exports.export_filename("shipments", states=["SP"], categories=["health_beauty"], ext="html") == \
        "deliveryiq_shipments_SP_health-beauty.html"
    assert exports.export_filename("x", states=["../ev il"]) == "deliveryiq_x_EV-IL.csv"          # no path characters survive


@needs_metrics
def test_report_model_section_is_honest_and_readable(orders, metrics):
    sub = orders[orders.customer_state == "SP"]
    rep = ins.summary_report(sub, "State: SP", ins.generate_insights(sub, config.DEFAULT_MIN_ORDERS), metrics)
    sec = rep[rep.index("## Prediction model"):rep.index("## Methodology and caveats")]
    h = exports.model_headline(metrics)
    noun = exports.signal_phrase(h["test_roc_auc"])[0]
    assert sec.splitlines()[2].startswith(f"**Read this first: {noun}, not an accurate predictor.**")     # headline first
    assert f"ROC-AUC {h['test_roc_auc']:.2f}" in sec and f"{h['test_pr_auc']:.3f}" in sec
    assert "does not change with the filters" in sec                                                       # model figures are not filtered
    assert "logistic_regression" not in sec and "dummy_prior" not in sec and "Selected model: **Logistic regression**" in sec
    for w in ("train", "validation", "test"):                                                              # real dates, not raw bounds
        s = metrics["split"][w]
        assert f"{w} {s['start']} to {s['end']} ({s['n']:,} orders" in sec
    assert "to open" not in sec and "2018-07-01 to open" not in sec
    assert "Drift between windows" in sec and "Probabilities are not better than the base rate" in sec
    assert sec.index("Read this first") < sec.index("Selection rationale")                                 # warning is not buried


@needs_metrics
def test_honest_summary_states_drift_and_brier_flag_from_the_metrics(metrics):
    pts = " ".join(exports.honest_summary(metrics))
    sp = {w: metrics["split"][w]["late_rate"] for w in ("train", "validation", "test")}
    assert "Drift between windows" in pts
    assert f"from {100 * sp['train']:.1f}% (train) to {100 * sp['validation']:.1f}% (validation) and {100 * sp['test']:.1f}% (test)" in pts
    cav = metrics["selected_model"]["test_window_caveat"]
    assert cav["brier_not_better_than_baseline"] is True                      # facts of the committed model
    assert "Probabilities are not better than the base rate" in pts and "no better than it" in pts
    assert "(better than it)" not in pts


@needs_metrics
def test_honest_summary_wording_flips_when_the_model_beats_the_base_rate(metrics):
    m = json.loads(json.dumps(metrics))
    m["selected_model"]["test_window_caveat"]["brier_not_better_than_baseline"] = False
    pts = " ".join(exports.honest_summary(m))
    assert "Probabilities on the test window" in pts and "(better than it)" in pts
    assert "Probabilities are not better than the base rate" not in pts
    del m["split"]["train"]["late_rate"]                                      # incomplete split info: no drift claim
    assert "Drift between windows" not in " ".join(exports.honest_summary(m))


# --------------------------------------------------------------------------- #
# comparable_insights: a single region cannot be compared with itself
# --------------------------------------------------------------------------- #
@needs_data
def test_comparable_insights_replaces_the_one_region_card_with_a_data_note(orders):
    sub = orders[orders.customer_state == "SE"]
    raw = ins.generate_insights(sub, 100, 0.05)
    vac = [i for i in raw if i.kind == "region_high"]
    assert vac and "None of the 1 region " in vac[0].text          # the raw finding is the vacuous one
    out = exports.comparable_insights(raw, 100)
    assert not [i for i in out if i.kind == "region_high"]
    notes = [i for i in out if i.kind == "insufficient_data"]
    assert len(notes) == 1 and exports.SINGLE_REGION_SECTION in notes[0].metrics["sections"]
    assert exports.SINGLE_REGION_SECTION in notes[0].text and "None of the 1 region " not in " ".join(i.text for i in out)
    # the region card is merged into the (already present) data note; every other finding is untouched
    assert len(out) == len(raw) - 1
    assert [i for i in raw if i.kind not in ("region_high", "insufficient_data")] == \
           [i for i in out if i.kind != "insufficient_data"]


@needs_data
def test_comparable_insights_leaves_a_multi_region_view_untouched(orders):
    raw = ins.generate_insights(orders, 100, 0.05)
    out = exports.comparable_insights(raw, 100)
    assert out == raw and out is not raw
    # two states are enough for a comparison
    two = orders[orders.customer_state.isin(["SP", "RJ"])]
    raw2 = ins.generate_insights(two, 100, 0.05)
    assert exports.comparable_insights(raw2, 100) == raw2


def test_comparable_insights_adds_the_note_when_none_exists_and_is_idempotent():
    over = ins.Insight("overview", "Overview", "x", {}, 500)
    vac = ins.Insight("region_high", "No region reliably above", "None of the 1 regions", {"n_screened": 1, "n_flagged": 0}, 500)
    out = exports.comparable_insights([over, vac], 100)
    assert [i.kind for i in out] == ["overview", "insufficient_data"] and out[1].n == 500
    assert out[1].metrics["sections"] == [exports.SINGLE_REGION_SECTION]
    assert exports.comparable_insights(out, 100) == out
    # a flagged region is never hidden
    flagged = ins.Insight("region_high", "Region above", "AL is higher", {"n_screened": 1, "n_flagged": 1}, 500)
    assert exports.comparable_insights([over, flagged], 100) == [over, flagged]


@needs_data
def test_comparable_insights_drops_the_category_cards_for_a_single_category(orders):
    """One product category in view: 'longest delivery' / 'highest late rate' / 'highest freight' would rank the category
    against itself, and a recommendation would call the only category 'slower'."""
    sub = orders[(orders.customer_state == "RJ") & (orders.product_category == "health_beauty")]
    assert len(sub) >= 100
    raw = ins.generate_insights(sub, 100, 0.05)
    assert {"category_duration", "category_delay", "category_freight"} <= {i.kind for i in raw}   # the raw cards exist
    out = exports.comparable_insights(raw, 100)
    assert not [i for i in out if i.kind.startswith("category_")]
    note = [i for i in out if i.kind == "insufficient_data"]
    assert len(note) == 1
    assert exports.SINGLE_CATEGORY_SECTION in note[0].metrics["sections"] and exports.SINGLE_REGION_SECTION in note[0].metrics["sections"]
    recs = " ".join(ins.recommendations(out))
    assert "slower categories" not in recs and "only one category in view" in recs
    assert "1 segments" not in " ".join(i.text for i in raw)           # singular: '1 segment'


@needs_data
def test_a_view_with_several_categories_keeps_its_category_cards(orders):
    sub = orders[orders.product_category.isin(["health_beauty", "bed_bath_table", "sports_leisure"])]
    out = exports.comparable_insights(ins.generate_insights(sub, 100, 0.05), 100)
    assert {"category_duration", "category_delay"} <= {i.kind for i in out}


@needs_data
@pytest.mark.parametrize("states,cats", [(["SP"], []), (["RJ"], ["health_beauty"]), ([], ["health_beauty"])])
def test_downloaded_report_and_page_cards_are_built_from_the_same_findings(orders, states, cats):
    """The report file must say what the page says: both go through comparable_insights, so a one-state or one-category
    view never prints a vacuous 'region vs overall' / 'category vs overall' finding in either place."""
    sub = orders
    if states:
        sub = sub[sub.customer_state.isin(states)]
    if cats:
        sub = sub[sub.product_category.isin(cats)]
    items = exports.comparable_insights(ins.generate_insights(sub, 100, 0.05), 100)
    md = ins.summary_report(sub, "test view", items, None, alpha=0.05)
    if states and len(states) == 1:
        assert "No region reliably above the overall late rate" not in md
        assert exports.SINGLE_REGION_SECTION in md
    if cats and len(cats) == 1:
        assert "Categories with the longest delivery times" not in md
        assert exports.SINGLE_CATEGORY_SECTION in md


def test_every_findings_consumer_goes_through_comparable_insights():
    """Regression guard: the Reports page cards, the downloadable report and the Overview headline cards must all use
    comparable_insights (the report once printed the vacuous one-region card that the page hid)."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "views"
    for fn in ("1_Overview.py", "5_Reports_Insights.py"):
        src = (root / fn).read_text(encoding="utf-8")
        assert src.count("comparable_insights(") >= 1, fn
    rep = (root / "5_Reports_Insights.py").read_text(encoding="utf-8")
    body = rep[rep.index("def cached_summary"):rep.index("def cached_shipments_csv")]
    assert "comparable_insights(" in body


def test_display_frame_uses_readable_labels_but_leaves_the_export_untouched():
    raw = pd.DataFrame({"order_id": ["a", "b"], "product_category": ["health_beauty", "bed_bath_table"],
                        "delivery_status": ["on_time", "late"], "customer_city": ["sao paulo", "rio de janeiro"],
                        "is_late": [0, 1], "total_price": [10.0, 20.0]})
    shown = exports.display_frame(raw)
    assert shown["product_category"].tolist() == ["Health beauty", "Bed bath table"]
    assert shown["delivery_status"].tolist() == ["On time", "Late"]
    assert shown["customer_city"].tolist() == ["Sao Paulo", "Rio De Janeiro"]
    assert raw["delivery_status"].tolist() == ["on_time", "late"] and raw["product_category"].iloc[0] == "health_beauty"
    # the CSV a user downloads keeps the raw machine-friendly values
    csv = exports.read_csv_bytes(exports.shipments_csv_bytes(raw.assign(is_late=raw["is_late"]), ["order_id", "product_category"]))
    assert csv["product_category"].tolist() == ["health_beauty", "bed_bath_table"]
    assert exports.column_label("delivery_delay_days") == "Days vs promised date"
    assert "negative" in exports.COLUMN_INFO["delivery_delay_days"][1] or "below 0 = arrived early" in exports.COLUMN_INFO["delivery_delay_days"][1]
