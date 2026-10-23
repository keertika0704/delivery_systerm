"""Tests for the Delay Prediction page (views/4_Delay_Prediction.py) and src/prediction_helpers.py."""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import charts, config, prediction, ui
from src import prediction_helpers as H

ROOT = Path(__file__).resolve().parents[1]
PAGE = str(ROOT / "views" / "4_Delay_Prediction.py")
TIMEOUT = 120

needs_data = pytest.mark.skipif(not config.ORDERS_DELIVERED_PATH.is_file(), reason="processed data missing")
needs_model = pytest.mark.skipif(not (config.MODEL_PATH.is_file() and config.MODEL_METRICS_PATH.is_file()),
                                 reason="model missing: run .venv/bin/python -m src.train_model")
pytestmark = [needs_data, needs_model]


@pytest.fixture(autouse=True)
def _clear_caches():
    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


@pytest.fixture(scope="module")
def bundle():
    return prediction.load_model()


@pytest.fixture(scope="module")
def metrics():
    return json.loads(config.MODEL_METRICS_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def orders():
    return pd.read_parquet(config.ORDERS_DELIVERED_PATH)


def _html(at: AppTest) -> str:
    return "\n".join([str(e.value) for e in at.get("html")] + [str(m.value) for m in at.markdown])


def _kpis(at: AppTest) -> list[str]:
    return re.findall(r'diq-kpi-value">([^<]*)', _html(at))


def _submit(at: AppTest) -> AppTest:
    return at.get("form_submit_button")[0].click().run(timeout=TIMEOUT)


def _fresh() -> AppTest:
    return AppTest.from_file(PAGE, default_timeout=TIMEOUT).run()


def _inputs_from_widgets(at: AppTest) -> dict:
    promise_key = next(n.key for n in at.number_input if n.key.startswith("pred_promise_"))
    return {
        "customer_state": at.selectbox(key="pred_customer_state").value, "seller_state": at.selectbox(key="pred_seller_state").value,
        "product_category": at.selectbox(key="pred_category").value, "n_items": at.number_input(key="pred_items").value,
        "total_price": at.number_input(key="pred_price").value, "total_freight": at.number_input(key="pred_freight").value,
        "purchase_date": at.date_input(key="pred_date").value, "estimated_delivery_days": at.number_input(key=promise_key).value,
    }


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def test_humanise_error_uses_labels_and_names_field():
    assert H.humanise_error("total_price is required.") == ("total_price", "Total product value is required.")
    f, t = H.humanise_error("n_sellers cannot exceed n_items.")
    assert f == "n_sellers" and t == "Number of sellers cannot exceed number of items."
    f, t = H.humanise_error("'XX' is not a known customer_state (choose one of the 27 values seen in training).")
    assert f == "customer_state" and "customer state" in t and "customer_state" not in t
    assert H.humanise_error("Unrecognised input field(s): ['foo'].")[0] is None


def test_field_errors_groups_by_field(bundle):
    _, errs = prediction.validate_input({"customer_state": "SP", "seller_state": "XX", "product_category": "nope", "n_items": 0,
                                         "total_price": None, "total_freight": 1, "purchase_date": "2018-01-02"}, bundle)
    by_field, general = H.field_errors(errs)
    assert {"seller_state", "product_category", "n_items", "total_price"} <= set(by_field)
    assert general == []
    assert all(f not in m for ms in by_field.values() for m in ms for f in ("seller_state", "n_items", "total_price"))


def test_form_defaults_match_pandas(bundle, orders):
    d = H.form_defaults(orders, bundle)
    assert d["total_price"] == round(float(orders["total_price"].median()), 2)
    assert d["total_freight"] == round(float(orders["total_freight"].median()), 2)
    assert d["customer_state"] == orders["customer_state"].astype(str).value_counts().index[0]
    assert d["product_category"] in bundle["form_options"]["product_category"]
    assert d["n_items"] >= 1
    empty = H.form_defaults(orders.iloc[0:0], bundle)
    assert empty["total_price"] is None and empty["customer_state"] == bundle["form_options"]["customer_state"][0]


def test_pair_promise_uses_pair_median_then_overall(bundle):
    opts = bundle["form_options"]
    key, med = next(iter(opts["state_pair_median_estimated_days"].items()))
    s, c = key.split(prediction.STATE_PAIR_SEP)
    assert H.pair_promise(bundle, s, c) == (int(round(med)), "pair")
    v, src = H.pair_promise(bundle, "ZZ", "ZZ")
    assert src == "overall" and v == int(round(opts["defaults"]["estimated_delivery_days"]))


def test_band_table_equals_metrics(bundle, metrics):
    tbl = H.band_table(bundle["risk_thresholds"], metrics, "Moderate")
    for _, r in tbl.iterrows():
        v = next(x for x in metrics["risk_bands"]["validation"] if x["band"] == r["Risk band"])
        t = next(x for x in metrics["risk_bands"]["test"] if x["band"] == r["Risk band"])
        assert r["Validation late rate"] == pytest.approx(v["late_rate"]) and r["Validation n"] == v["n"]
        assert r["Test late rate"] == pytest.approx(t["late_rate"]) and r["Test n"] == t["n"]
    assert list(tbl["This order"]) == ["", "◀ this order", ""]
    ranges = H.band_ranges(bundle["risk_thresholds"])
    assert ranges["Low"].startswith("below") and ranges["High"].endswith("and above")


def test_band_table_without_metrics_is_safe(bundle):
    tbl = H.band_table(bundle["risk_thresholds"], None)
    assert len(tbl) == 3 and tbl["Test late rate"].isna().all()
    assert H.band_separation_text(None) is None


def test_band_separation_text_states_overlap_from_diagnostics(metrics):
    txt = H.band_separation_text(metrics)
    ov = metrics["risk_bands"]["test_diagnostics"]["adjacent_intervals_overlap"]
    assert ("barely separated" in txt) == bool(ov["Moderate-High"])
    low = next(x for x in metrics["risk_bands"]["test"] if x["band"] == "Low")
    assert f"{low['late_rate']:.1%}" in txt and f"{low['n']:,}" in txt


def test_limitations_quote_real_numbers(bundle, metrics):
    text = " ".join(H.limitations(bundle, metrics))
    sel = metrics["selected_model"]["name"]
    test = metrics["models"][sel]["test"]
    assert f"{test['roc_auc']:.2f}" in text and f"{test['pr_auc']:.3f}" in text and f"{test['positive_rate']:.3f}" in text
    sp = metrics["split"]
    for w in ("train", "validation", "test"):
        assert f"{sp[w]['late_rate']:.1%}" in text
    assert "associations" in text and "guarantee" in text
    short = " ".join(H.limitations(bundle, metrics, include_calibration=False, include_bands=False))
    assert "Probabilities are rough" not in short and "Bands overlap" not in short
    assert f"{test['roc_auc']:.2f}" in H.read_first(bundle, metrics)


def test_limitations_degrade_gracefully_without_metrics(bundle):
    lines = H.limitations(bundle, None)
    assert lines and "Historical data only" in lines[-1]
    assert "missing" in H.read_first(bundle, None)


def test_historical_context_matches_pandas(orders):
    ctx = H.historical_context(orders, "SP", "RJ", "bed_bath_table")
    sub = orders[(orders["seller_state"].astype(str) == "SP") & (orders["customer_state"].astype(str) == "RJ")]
    assert ctx["route"]["n"] == len(sub) and ctx["route"]["late_rate"] == pytest.approx(sub["is_late"].astype(float).mean())
    assert ctx["route"]["ci_low"] < ctx["route"]["late_rate"] < ctx["route"]["ci_high"]
    assert ctx["overall"]["n"] == len(orders)
    none = H.historical_context(orders, "ZZ", "ZZ", "no_such_category")
    assert none["route"]["n"] == 0 and np.isnan(none["route"]["late_rate"]) and none["route"]["small"]
    empty = H.historical_context(orders.iloc[0:0], "SP", "RJ", "x")
    assert empty["overall"]["n"] == 0


def _example_inputs(bundle, orders):
    d = H.form_defaults(orders, bundle)
    return {"customer_state": "RJ", "seller_state": "SP", "product_category": d["product_category"], "n_items": 1,
            "total_price": d["total_price"], "total_freight": d["total_freight"], "purchase_date": d["purchase_date"],
            "estimated_delivery_days": H.pair_promise(bundle, "SP", "RJ")[0]}


def test_what_if_rows_match_predict_risk_and_skip_out_of_range(bundle, orders):
    inp = _example_inputs(bundle, orders)
    wi = H.what_if_promise(inp, bundle, inp["estimated_delivery_days"])
    assert list(wi["extra_days"])[0] == 0 and len(wi) >= 2
    for _, r in wi.iterrows():
        res = prediction.predict_risk({**inp, "estimated_delivery_days": int(r["promise_days"])}, bundle)
        assert r["probability"] == pytest.approx(res["probability"]) and r["risk_label"] == res["risk_label"]
    hi = bundle["form_options"]["ranges"]["estimated_delivery_days"][1]
    far = H.what_if_promise(inp, bundle, hi)       # +3 etc. fall outside the training range -> only the entered value stays
    assert list(far["extra_days"]) == [0]


def test_interpretation_labels_and_numbers(bundle, metrics, orders):
    inp = _example_inputs(bundle, orders)
    res = prediction.predict_risk(inp, bundle)
    hist = H.historical_context(orders, "SP", "RJ", inp["product_category"])
    wi = H.what_if_promise(inp, bundle, inp["estimated_delivery_days"])
    items = H.interpretation(res, bundle, metrics, hist, ("SP", "RJ"), wi)
    tags = [i["tag"] for i in items]
    assert tags == ["Model prediction", "Observed", "Statistical association", "Suggested actions"]
    assert f"{res['probability']:.1%}" in items[0]["body"] and res["risk_label"] in items[0]["body"]
    test = metrics["split"]["test"]
    assert f"{test['late_rate']:.1%}" in items[0]["body"]
    assert f"n = {hist['route']['n']:,}" in items[1]["body"]
    assert "not by itself" in items[2]["body"]          # association is never presented as a cause


@pytest.mark.parametrize("label,kind", [("Low", "success"), ("Moderate", "watch"), ("High", "watch")])
def test_suggested_action_depends_on_band(bundle, metrics, label, kind):
    res = {"probability": 0.1, "risk_label": label}
    items = H.interpretation(res, bundle, metrics, None)
    assert items[-1]["tag"] == "Suggested actions" and items[-1]["kind"] == kind


def test_describe_order_has_no_identifiers():
    s = H.describe_order({"customer_state": "RJ", "seller_state": "SP", "product_category": "health_beauty", "n_items": 2,
                          "total_price": 120, "total_freight": 20.5, "estimated_delivery_days": 24, "purchase_date": "2018-03-31"})
    assert s == "SP to RJ · Health beauty · 2 items · R$ 120.00 + R$ 20.50 freight · 24-day promise · purchased 2018-03-31"


# ---- batch -----------------------------------------------------------------
def test_template_has_exact_headers_and_example_labels(bundle, orders):
    tpl = H.build_template(orders, bundle)
    assert list(tpl.columns) == H.BATCH_COLUMNS
    assert list(tpl["order_ref"]) == ["EXAMPLE-1", "EXAMPLE-2", "EXAMPLE-3"]
    parsed, notes = H.parse_batch_csv(H.template_csv(orders, bundle))
    assert notes == [] and len(parsed) == 3
    scored, summary = H.score_batch(parsed, bundle)
    assert summary["n_valid"] == 3 and summary["n_invalid"] == 0
    assert all(c in bundle["form_options"]["product_category"] for c in tpl["product_category"])


def _csv(rows: list[dict], sep: str = ",") -> bytes:
    return pd.DataFrame(rows).to_csv(index=False, sep=sep).encode("utf-8")


def _row(**kw):
    base = {"customer_state": "RJ", "seller_state": "SP", "product_category": "bed_bath_table", "n_items": "1",
            "total_price": "100", "total_freight": "20", "purchase_date": "2018-03-30", "estimated_delivery_days": "24"}
    base.update(kw)
    return base


def test_parse_errors_are_plain_language():
    with pytest.raises(H.BatchFileError, match="empty"):
        H.parse_batch_csv(b"")
    with pytest.raises(H.BatchFileError, match="Required column"):
        H.parse_batch_csv(b"a,b\n1,2\n")
    with pytest.raises(H.BatchFileError, match="no rows"):
        H.parse_batch_csv((",".join(H.BATCH_REQUIRED) + "\n").encode())
    with pytest.raises(H.BatchFileError, match="limit"):
        H.parse_batch_csv(b"x" * (H.MAX_BATCH_BYTES + 1))
    with pytest.raises(H.BatchFileError, match="duplicate"):
        H.parse_batch_csv((",".join(H.BATCH_REQUIRED + ["n_items"]) + "\n" + ",".join(["1"] * 8) + "\n").encode())


def test_parse_row_limit(monkeypatch):
    monkeypatch.setattr(H, "MAX_BATCH_ROWS", 2)
    with pytest.raises(H.BatchFileError, match="rows"):
        H.parse_batch_csv(_csv([_row(), _row(), _row()]))


def test_parse_handles_semicolon_bom_headers_and_ignored_columns():
    raw = _csv([_row(customer_name="Jane Doe")], sep=";")
    raw = b"\xef\xbb\xbf" + raw.replace(b"customer_state", b"Customer State")
    df, notes = H.parse_batch_csv(raw)
    assert "customer_state" in df.columns and "customer_name" not in df.columns
    assert notes and "customer_name" in notes[0]
    assert df.loc[0, "n_items"] == "1"


def test_score_batch_reports_invalid_rows_and_matches_single(bundle):
    rows = [_row(order_ref="ok"), _row(order_ref="bad-state", customer_state="XX"), _row(order_ref="bad-date", purchase_date="31/03/2018"),
            _row(order_ref="neg", total_price="-5"), _row(order_ref="blank", total_freight="")]
    parsed, _ = H.parse_batch_csv(_csv(rows))
    scored, s = H.score_batch(parsed, bundle)
    assert s["n_rows"] == 5 and s["n_valid"] == 1 and s["n_invalid"] == 4
    assert list(scored["csv_row"]) == [2, 3, 4, 5, 6]
    ok = scored.iloc[0]
    single = prediction.predict_risk({k: v for k, v in rows[0].items() if k != "order_ref"}, bundle)
    assert ok["late_probability"] == pytest.approx(single["probability"], abs=5e-5) and ok["risk_label"] == single["risk_label"]
    assert scored.loc[1:, "late_probability"].isna().all()
    rep = H.invalid_rows_report(scored)
    assert set(rep["csv_row"]) == {3, 4, 5, 6}
    probs = " ".join(rep["problem"])
    assert "customer state" in probs and "purchase date" in probs.lower() and "total_price" not in probs
    assert sum(s["bands"].values()) == 1


def test_score_batch_all_invalid_and_empty(bundle):
    parsed, _ = H.parse_batch_csv(_csv([_row(customer_state="XX")]))
    scored, s = H.score_batch(parsed, bundle)
    assert s["n_valid"] == 0 and np.isnan(s["mean_probability"])
    assert H.invalid_rows_report(scored.iloc[0:0]).empty


def test_scenario_chart_handles_empty_and_data():
    assert "No scenarios" in charts.scenario_comparison_chart([]).layout.annotations[0].text
    fig = charts.scenario_comparison_chart([{"label": "A", "probability": 0.2, "risk_label": "High"},
                                            {"label": "B", "probability": float("nan"), "risk_label": "Low"}],
                                           {"moderate": 0.064, "high": 0.165})
    assert list(fig.data[0].y) == [0.2] and fig.layout.title.text


# --------------------------------------------------------------------------- #
# page (AppTest)
# --------------------------------------------------------------------------- #
def test_page_renders_form_without_exception(bundle):
    at = _fresh()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Predict an order", "Score a CSV file", "What changes the risk?"]
    html = _html(at)
    assert "Read this first" in html and "Predict Delay Risk" in [b.label for b in at.get("form_submit_button")]
    assert "No prediction yet" in html and "Model card" in html
    assert not at.warning                                   # no streamlit deprecation / warning boxes


def test_submit_shows_probability_equal_to_predict_risk(bundle):
    at = _fresh()
    at.selectbox(key="pred_customer_state").select("RJ")
    at.run(timeout=TIMEOUT)                                  # route change refreshes the prefilled promise
    at = _submit(at)
    assert not at.exception
    inputs = _inputs_from_widgets(at)
    expected = prediction.predict_risk(inputs, bundle)
    kpis = _kpis(at)
    assert kpis[0] == f"{expected['probability']:.1%}" and kpis[1] == expected["risk_label"]
    html = _html(at)
    assert "Limitations: please read" in html and "Suggested actions" in html and "Assumptions used" in html
    assert "What each risk band has meant historically" in html
    assert at.session_state["pred_last"]["result"]["probability"] == pytest.approx(expected["probability"])


def test_changing_inputs_changes_output(bundle):
    at = _fresh()
    at = _submit(at)
    p1 = at.session_state["pred_last"]["result"]["probability"]
    k1 = _kpis(at)[0]
    pk = next(n.key for n in at.number_input if n.key.startswith("pred_promise_"))
    at.number_input(key=pk).set_value(60)
    at = _submit(at)
    p2 = at.session_state["pred_last"]["result"]["probability"]
    assert p2 != p1 and _kpis(at)[0] == f"{p2:.1%}" and _kpis(at)[0] != k1
    assert p2 < p1                                            # longer promise: lower estimate in this model


def test_invalid_inputs_show_readable_errors_and_no_result():
    at = _fresh()
    at.number_input(key="pred_items").set_value(0)
    at = _submit(at)
    assert not at.exception
    html = _html(at)
    assert "Number of items = 0 is outside the allowed range" in html and "Please fix" in html
    assert "pred_last" not in at.session_state and "No prediction yet" in html
    field_msgs = " ".join(re.findall(r'pred-field-err">([^<]*)', html))
    assert field_msgs and "n_items" not in field_msgs        # no raw field names leak into the messages


def test_required_fields_left_blank_are_reported_not_substituted(monkeypatch, orders):
    """With no history to prefill from, price / freight start blank: submitting must say so, never invent a value."""
    monkeypatch.setattr(ui, "get_orders", lambda: orders.iloc[0:0])
    at = AppTest.from_file(PAGE, default_timeout=TIMEOUT).run()
    assert not at.exception
    assert at.number_input(key="pred_price").value is None
    at = _submit(at)
    assert not at.exception
    html = _html(at)
    assert "Total product value is required." in html and "Total freight value is required." in html
    assert "pred_last" not in at.session_state


def test_date_outside_training_window_warns_but_predicts(bundle):
    at = _fresh()
    import datetime
    at.date_input(key="pred_date").set_value(datetime.date(2026, 10, 23))
    at = _submit(at)
    assert not at.exception and "pred_last" in at.session_state
    assert "Outside what the model was trained on" in _html(at)


def test_pin_and_clear_scenarios(bundle):
    at = _fresh()
    at = _submit(at)
    at.button(key="pred_pin").click().run(timeout=TIMEOUT)
    assert len(at.session_state["pred_scenarios"]) == 1
    at.button(key="pred_pin").click().run(timeout=TIMEOUT)          # same order again: not duplicated
    assert len(at.session_state["pred_scenarios"]) == 1
    pk = next(n.key for n in at.number_input if n.key.startswith("pred_promise_"))
    at.number_input(key=pk).set_value(50)
    at = _submit(at)
    at.button(key="pred_pin").click().run(timeout=TIMEOUT)
    scen = at.session_state["pred_scenarios"]
    assert [s["label"] for s in scen] == ["Scenario A", "Scenario B"]
    assert "3. Compare scenarios" in _html(at) and not at.exception
    at.button(key="pred_clear").click().run(timeout=TIMEOUT)
    assert at.session_state["pred_scenarios"] == []


@needs_data
def test_empty_filter_does_not_break_page(orders):
    combo = orders.groupby(["customer_state", "product_category"], observed=True).size()
    states, cats = sorted(orders["customer_state"].astype(str).unique()), sorted(orders["product_category"].astype(str).unique())
    present = {(str(a), str(b)) for a, b in combo.index}
    s, c = next((s, c) for s in states for c in cats if (s, c) not in present)
    at = AppTest.from_file(PAGE, default_timeout=TIMEOUT)
    at.session_state[ui.K_STATES] = [s]
    at.session_state[ui.K_CATEGORIES] = [c]
    at.run()
    assert not at.exception
    assert "Sidebar filters do not apply" in _html(at)
    at = _submit(at)
    assert not at.exception and _kpis(at)[0].endswith("%")


def test_missing_model_shows_friendly_error_and_stops(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "nope.joblib")
    at = AppTest.from_file(PAGE, default_timeout=TIMEOUT).run()
    assert not at.exception
    html = _html(at)
    assert "The prediction model is not available" in html
    assert any(".venv/bin/python -m src.train_model" in c.value for c in at.code)
    assert "Predict Delay Risk" not in [b.label for b in at.get("form_submit_button")]


def test_missing_metrics_file_degrades_gracefully(monkeypatch):
    monkeypatch.setattr(ui, "get_model_metrics", lambda: None)
    at = AppTest.from_file(PAGE, default_timeout=TIMEOUT).run()
    assert not at.exception
    at = _submit(at)
    assert not at.exception and _kpis(at)[0].endswith("%")
    assert "missing" in _html(at)


# =========================================================================== #
# Hardening: page-level KPIs, small-sample rule, pins, batch, model card, state, charts
# (each block below is aimed at a behaviour the brief treats as critical)
# =========================================================================== #
import datetime  # noqa: E402

KPI_RE = re.compile(r'<div class="diq-kpi" style="--accent:(#[0-9A-Fa-f]{6})[^"]*"[^>]*>.*?diq-kpi-label">([^<]*)</span>.*?'
                    r'diq-kpi-value">([^<]*)</div>(?:<div class="diq-kpi-sub">([^<]*)</div>)?', re.S)


def _cards(at: AppTest) -> dict[str, dict]:
    """Every KPI card on the page: ``{label: {accent, value, sub}}`` (HTML entities decoded)."""
    import html as _h
    return {_h.unescape(m[1]): {"accent": m[0].upper(), "value": _h.unescape(m[2]), "sub": _h.unescape(m[3] or "")}
            for m in KPI_RE.findall(_html(at))}


def _route(orders: pd.DataFrame, seller: str, customer: str) -> pd.DataFrame:
    return orders[(orders["seller_state"].astype(str) == seller) & (orders["customer_state"].astype(str) == customer)]


def _predict_route(seller: str, customer: str, filters: dict | None = None) -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=TIMEOUT)
    for k, v in (filters or {}).items():
        at.session_state[k] = v
    at.run()
    at.selectbox(key="pred_seller_state").select(seller)
    at.selectbox(key="pred_customer_state").select(customer)
    at.run(timeout=TIMEOUT)
    return _submit(at)


def _pin(at: AppTest) -> AppTest:
    return at.button(key="pred_pin").click().run(timeout=TIMEOUT)


def _promise_key(at: AppTest) -> str:
    return next(n.key for n in at.number_input if n.key.startswith("pred_promise_"))


def _chart_spec(at: AppTest, key: str) -> dict:
    el = next(e for e in at.get("plotly_chart") if e.proto.id.endswith(f"-{key}"))
    return json.loads(el.proto.spec)


def _arr(v) -> list:
    """Plotly >= 6 serialises numeric arrays as ``{"dtype", "bdata"}``; decode to a plain list."""
    if isinstance(v, dict) and "bdata" in v:
        import base64
        return np.frombuffer(base64.b64decode(v["bdata"]), dtype=np.dtype(v["dtype"])).tolist()
    return list(v)


def _dataframes(at: AppTest) -> list:
    return list(at.dataframe)


def _frame_with(at: AppTest, column: str) -> pd.DataFrame:
    return next(d.value for d in at.dataframe if column in d.value.columns)


@pytest.fixture
def downloads(monkeypatch):
    """Record the bytes handed to every ``st.download_button`` (AppTest cannot read them back)."""
    got: dict[str, bytes] = {}
    orig = st.download_button

    def wrap(label, data=None, *a, **k):
        got[k.get("key", label)] = data
        return orig(label, data, *a, **k)
    monkeypatch.setattr(st, "download_button", wrap)
    return got


# ---- result KPIs vs pandas ----------------------------------------------------
@pytest.mark.parametrize("seller,customer", [("SP", "RJ"), ("RJ", "SP")])
def test_result_kpis_match_pandas_for_route(bundle, metrics, orders, seller, customer):
    """Probability, band, test-window rate and the route's observed rate: all four KPIs, both route directions."""
    at = _predict_route(seller, customer)
    assert not at.exception
    expected = prediction.predict_risk(_inputs_from_widgets(at), bundle)
    cards = _cards(at)
    sub = _route(orders, seller, customer)
    assert len(sub) > 100 and sub["is_late"].astype(float).mean() != pytest.approx(orders["is_late"].astype(float).mean(), abs=1e-3)
    assert cards["Late-delivery probability"]["value"] == f"{expected['probability']:.1%}"
    band = cards["Predicted risk band"]
    assert band["value"] == expected["risk_label"] and band["sub"] == f"Probability {H.band_ranges(expected['thresholds'])[expected['risk_label']]}"
    tw, vw = metrics["split"]["test"], metrics["split"]["validation"]
    tcard = cards["Test-window late rate"]
    assert tcard["value"] == ui.fmt_pct(tw["late_rate"]) and tcard["value"] != ui.fmt_pct(vw["late_rate"])
    sub_txt = tcard["sub"].replace("\u2011", "-")          # dates use non-breaking hyphens so they never wrap mid-date
    assert ui.fmt_n(tw["n"]) in sub_txt and tw["start"] in sub_txt and tw["end"] in sub_txt
    assert "\u2011" in tcard["sub"] and not re.search(r"\d-\d", tcard["sub"])
    rcard = cards["Observed on this route"]
    assert rcard["value"] == ui.fmt_pct(sub["is_late"].astype(float).mean())
    assert rcard["sub"] == f"{seller} to {customer}, {ui.fmt_n(len(sub))}"
    assert f"{seller} to {customer}" in _html(at)


def test_route_history_uses_all_orders_not_the_sidebar_filter(orders):
    """The banner says the history ignores the sidebar filters: an active filter must not change the route KPI."""
    sub = _route(orders, "SP", "RJ")
    at = _predict_route("SP", "RJ", {ui.K_STATES: ["AM"]})
    assert not at.exception and "Sidebar filters do not apply" in _html(at)
    card = _cards(at)["Observed on this route"]
    assert card["value"] == ui.fmt_pct(sub["is_late"].astype(float).mean())
    assert ui.fmt_n(len(sub)) in card["sub"]


def test_route_without_history_says_so(orders):
    present = {(str(a), str(b)) for a, b in orders.groupby(["seller_state", "customer_state"], observed=True).size().index}
    opts = prediction.load_model()["form_options"]
    pair = next(((s, c) for s in opts["seller_state"] for c in opts["customer_state"] if (s, c) not in present), None)
    if pair is None:
        pytest.skip("every route has history")
    at = _predict_route(*pair)
    card = _cards(at)["Observed on this route"]
    assert card["value"] == ui.DASH and card["sub"] == f"no orders on {pair[0]} to {pair[1]}"
    assert not at.exception


# ---- small-sample rule on the page ------------------------------------------
def test_small_route_gets_small_sample_footnote_and_big_route_does_not(orders):
    g = orders.groupby([orders["seller_state"].astype(str), orders["customer_state"].astype(str)]).size()
    opts = prediction.load_model()["form_options"]
    small = [(s, c, int(n)) for (s, c), n in g.items() if 20 <= n < config.DEFAULT_MIN_ORDERS and s in opts["seller_state"]]
    assert small, "the data has routes with 20-99 orders"
    s, c, n = small[0]
    at = _predict_route(s, c)
    assert not at.exception
    note = ui.small_sample_note(n, config.DEFAULT_MIN_ORDERS)
    assert note and note in _html(at)
    assert _cards(at)["Observed on this route"]["sub"] == f"{s} to {c}, {ui.fmt_n(n)}"
    big = _predict_route("SP", "RJ")
    assert "Small sample (" not in _html(big).split("Do the risk bands hold up?")[0].split("2. Result")[1]


# ---- colours of the band ----------------------------------------------------
@pytest.mark.parametrize("label,accent,p", [("Low", "green", 0.02), ("Moderate", "amber", 0.10), ("High", "red", 0.30)])
def test_probability_and_band_cards_use_band_colour(monkeypatch, label, accent, p):
    real = prediction.predict_risk

    def forced(inputs, bundle):
        r = real(inputs, bundle)
        return {**r, "probability": p, "risk_label": label}
    monkeypatch.setattr(prediction, "predict_risk", forced)
    at = _submit(_fresh())
    cards = _cards(at)
    want = ui._ACCENTS[accent][0].upper()
    assert cards["Late-delivery probability"]["accent"] == want and cards["Predicted risk band"]["accent"] == want
    assert cards["Predicted risk band"]["value"] == label
    # the other two KPIs stay neutral slate whatever the band
    assert cards["Test-window late rate"]["accent"] == cards["Observed on this route"]["accent"] == ui._ACCENTS["slate"][0].upper()


# ---- scenario pins ----------------------------------------------------------
def test_pin_limit_is_three_and_pins_store_the_real_estimates(bundle):
    at = _submit(_fresh())
    seen = []
    for days in (30, 40, 50):
        at.number_input(key=_promise_key(at)).set_value(days)
        at = _submit(at)
        at = _pin(at)
        seen.append(days)
    scen = at.session_state["pred_scenarios"]
    assert [s["label"] for s in scen] == ["Scenario A", "Scenario B", "Scenario C"]
    base = _inputs_from_widgets(at)
    for s, days in zip(scen, seen):
        assert s["probability"] == pytest.approx(prediction.predict_risk({**base, "estimated_delivery_days": days}, bundle)["probability"])
    assert at.button(key="pred_pin").disabled                     # a 4th pin is not possible
    at.number_input(key=_promise_key(at)).set_value(60)
    at = _submit(at)
    assert at.button(key="pred_pin").disabled and len(at.session_state["pred_scenarios"]) == 3
    assert any("3 of 3 scenarios pinned" in c.value for c in at.caption)
    assert len(_arr(_chart_spec(at, "pred_scenario_chart")["data"][0]["y"])) == 3


# ---- batch scoring through the page -----------------------------------------
def _batch_file(rows: list[dict]) -> tuple[str, bytes, str]:
    return ("orders.csv", _csv(rows), "text/csv")


def _upload(rows: list[dict]) -> AppTest:
    at = _fresh()
    at.file_uploader(key="pred_batch_file").set_value(_batch_file(rows)).run(timeout=TIMEOUT)
    return at


def test_batch_page_counts_exclude_invalid_rows(bundle, downloads):
    rows = [_row(order_ref="A1"), _row(order_ref="A2", estimated_delivery_days="", total_weight_g="500"),
            _row(order_ref="A3", n_sellers="1", estimated_delivery_days="25", total_weight_g="300", total_volume_cm3="1000",
                 max_product_length_cm="20", max_product_height_cm="10", max_product_width_cm="10"),
            _row(order_ref="A4", customer_state="RJ", seller_state="SP", purchase_date="2026-10-23"),
            _row(order_ref="A9", customer_state="XX", total_price="abc")]
    at = _upload(rows)
    assert not at.exception
    parsed, _ = H.parse_batch_csv(_csv(rows))
    scored, s = H.score_batch(parsed, bundle)
    assert s["n_rows"] == 5 and s["n_valid"] == 4 and s["n_invalid"] == 1
    cards = _cards(at)
    assert cards["Rows in file"]["value"] == "5" and cards["Scored"]["value"] == "4" and cards["Rows with problems"]["value"] == "1"
    assert cards["Scored"]["sub"] == f"mean probability {ui.fmt_pct(s['mean_probability'])}"
    assert cards["High / Moderate / Low"]["value"] == f"{s['bands']['High']:,} / {s['bands']['Moderate']:,} / {s['bands']['Low']:,}"
    assert sum(s["bands"].values()) == 4                                    # the invalid row is in no band
    caption = " ".join(c.value for c in at.caption)
    assert f"{s['n_assumed']:,} scored row(s) used training-window medians" in caption
    assert s["n_assumed"] == 3                                              # A1, A2, A4 (A3 filled everything); not the invalid row
    assert "outside the training range" in _html(at) and f"{s['n_warnings']:,} row(s)" in _html(at) and s["n_warnings"] == 1


def test_batch_scored_csv_holds_every_row_and_blank_results_for_invalid(downloads):
    rows = [_row(order_ref=f"R{i}") for i in range(5)] + [_row(order_ref="BAD", customer_state="XX")]
    at = _upload(rows)
    assert not at.exception
    out = pd.read_csv(io.BytesIO(downloads["pred_scored_dl"]), dtype=str, keep_default_na=False)
    assert len(out) == 6 and list(out["order_ref"]) == [f"R{i}" for i in range(5)] + ["BAD"]
    assert "csv_row" not in out.columns and "input_error" in out.columns
    bad = out.iloc[-1]
    assert bad["late_probability"] == "" and bad["risk_label"] == "" and bad["assumptions_used"] == "" and bad["input_warning"] == ""
    assert "customer_state" in bad["input_error"] and all(out.iloc[:5]["late_probability"] != "")


def test_batch_preview_is_capped_at_200_and_lists_only_scored_rows_results_first(downloads):
    rows = [_row(order_ref=f"R{i}") for i in range(230)] + [_row(order_ref="BAD", customer_state="XX")]
    at = _upload(rows)
    assert not at.exception
    prev = _frame_with(at, "late_probability")
    assert len(prev) == 200 and "BAD" not in set(prev["order_ref"])
    assert list(prev.columns[:4]) == ["csv_row", "order_ref", "late_probability", "risk_label"]
    assert not prev.astype(str).apply(lambda c: c.str.contains("None|nan", regex=True)).any().any()
    assert "input_error" not in prev.columns
    assert len(downloads["pred_scored_dl"].decode().strip().splitlines()) == 232           # header + 231 rows
    inv = _frame_with(at, "problem")
    assert list(inv["csv_row"]) == [232]


def test_batch_bad_file_shows_friendly_message_not_traceback():
    at = _fresh()
    at.file_uploader(key="pred_batch_file").set_value(("x.csv", b"a,b\n1,2\n", "text/csv")).run(timeout=TIMEOUT)
    assert not at.exception and "Required column(s) missing" in _html(at)


# ---- model card values ------------------------------------------------------
def test_model_card_kpis_match_metrics_file(bundle, metrics):
    cards = _cards(_fresh())
    sel = metrics["selected_model"]["name"]
    entry = metrics["models"][sel]
    tr, te = metrics["split"]["train"], metrics["split"]["test"]
    assert tr["n"] != te["n"]
    assert cards["Training orders"]["value"] == f"{bundle['train_window']['n']:,}" == f"{tr['n']:,}"
    assert cards["Training orders"]["sub"] == f"late rate {ui.fmt_pct(tr['late_rate'])}"
    roc = cards["Test ROC-AUC"]
    assert roc["value"] == f"{entry['test']['roc_auc']:.2f}"
    lo, hi = entry["test_ci95"]["roc_auc"]
    assert f"95% CI {lo:.2f} to {hi:.2f}" in roc["sub"] and f"validation {entry['validation']['roc_auc']:.2f}" in roc["sub"]
    pr = cards["Test PR-AUC"]
    assert pr["value"] == f"{entry['test']['pr_auc']:.3f}" and f"no-skill baseline {entry['test']['positive_rate']:.3f}" in pr["sub"]
    assert cards["Model"]["value"] == entry["label"]


def test_windows_table_shows_late_rate_not_its_complement(metrics):
    at = _fresh()
    tbl = _frame_with(at, "Late rate")
    rates = list(tbl["Late rate"])
    assert rates == [ui.fmt_pct(metrics["split"][w]["late_rate"]) for w in ("train", "validation", "test")]
    assert list(tbl["Orders"]) == [f"{metrics['split'][w]['n']:,}" for w in ("train", "validation", "test")]
    assert list(tbl["Late orders"]) == [f"{metrics['split'][w]['n_late']:,}" for w in ("train", "validation", "test")]


def test_importance_radio_switches_the_window_and_chart_data(metrics):
    at = _fresh()
    pi = metrics["permutation_importance"]

    def expected(items):
        top = sorted(items, key=lambda i: -i["importance_mean"])[:12]
        return [i["feature"] for i in top], [i["importance_mean"] for i in top]
    spec = _chart_spec(at, "pred_importance")
    feats, vals = expected(pi["items"])
    assert np.allclose(_arr(spec["data"][0]["x"]), vals)
    assert [charts.pretty_feature_name(f) for f in feats] == list(spec["data"][0]["y"])
    at.radio(key="pred_imp_window").set_value("test").run(timeout=TIMEOUT)
    spec_t = _chart_spec(at, "pred_importance")
    feats_t, vals_t = expected(pi["test_items"])
    assert np.allclose(_arr(spec_t["data"][0]["x"]), vals_t) and not np.allclose(_arr(spec_t["data"][0]["x"]), vals)
    assert spec_t["layout"]["title"]["subtitle"]["text"].startswith("Test window.")
    assert spec["layout"]["title"]["subtitle"]["text"].startswith("Validation window.")
    labels = [str(t) for t in spec_t["data"][1]["text"]]
    assert not any(t.strip().startswith("-0.000") for t in labels)           # no negative zero


def test_association_callout_has_no_raw_column_names(metrics):
    html = _html(_fresh())
    body = html.split("Statistical association, not cause")[1].split("Do the risk bands hold up?")[0]
    assert "not cause" not in body[:0] and "Top-3 by validation importance" in body
    for raw in ("estimated_delivery_days", "customer_state", "seller_state", "same_state", "total_freight", "purchase_dow", "total_weight_g"):
        assert raw not in body
    assert "Promised delivery time (days)" in body


def test_calibration_caption_orders_predicted_before_observed(bundle):
    cav = bundle["test_window_caveat"]
    caps = " ".join(c.value for c in _fresh().caption)
    assert (f"Test window: mean predicted risk {ui.fmt_pct(cav['mean_predicted'])} vs {ui.fmt_pct(cav['observed_late_rate'])} observed "
            f"({ui.fmt_n(cav['n'])}).") in caps


def test_band_table_highlights_only_the_current_band(bundle):
    at = _submit(_fresh())
    cur = at.session_state["pred_last"]["result"]["risk_label"]
    idx = ["Low", "Moderate", "High"].index(cur)
    styled = [d.proto.arrow_data.styler for d in at.dataframe if d.proto.arrow_data.HasField("styler") and d.proto.arrow_data.styler.styles]
    assert len(styled) == 1
    rows = set(re.findall(r"_row(\d+)_col", styled[0].styles))
    assert rows == {str(idx)}
    assert "This order" in _frame_with(at, "This order").columns
    assert list(_frame_with(at, "This order")["This order"]) == [("◀ this order" if b == cur else "") for b in ("Low", "Moderate", "High")]


# ---- returning to the page ---------------------------------------------------
def test_form_is_restored_from_the_stored_result_when_the_page_is_revisited(bundle):
    at = _predict_route("SP", "AM")
    at.number_input(key=_promise_key(at)).set_value(60)
    at.number_input(key="pred_weight").set_value(1234)
    at = _submit(at)
    last = at.session_state["pred_last"]
    assert last["inputs"]["estimated_delivery_days"] == 60
    # a new browser run of the page that still has the stored result but no widget state (navigated away and back)
    back = AppTest.from_file(PAGE, default_timeout=TIMEOUT)
    back.session_state["pred_last"] = last
    back.run()
    assert not back.exception and not back.warning
    assert back.selectbox(key="pred_seller_state").value == "SP" and back.selectbox(key="pred_customer_state").value == "AM"
    assert back.number_input(key=_promise_key(back)).value == 60
    assert back.number_input(key="pred_weight").value == 1234
    for f, k in (("total_price", "pred_price"), ("total_freight", "pred_freight"), ("n_items", "pred_items")):
        assert back.number_input(key=k).value == last["inputs"][f]
    assert back.date_input(key="pred_date").value == last["inputs"]["purchase_date"]
    assert back.selectbox(key="pred_category").value == last["inputs"]["product_category"]
    html = _html(back)
    assert "previously submitted order" not in html and "2. Result" in html
    assert _cards(back)["Late-delivery probability"]["value"] == f"{last['result']['probability']:.1%}"


def test_changing_the_route_after_a_result_shows_the_stale_notice():
    at = _predict_route("SP", "AM")
    at.selectbox(key="pred_customer_state").select("RJ").run(timeout=TIMEOUT)
    html = _html(at)
    assert "This result is for the previously submitted order" in html and "no longer matches" in html


# ---- helper-level behaviours --------------------------------------------------
def test_band_ranges_text_in_full():
    assert H.band_ranges({"moderate": 0.064, "high": 0.165}) == {
        "Low": "below 6.4%", "Moderate": "6.4% to below 16.5%", "High": "16.5% and above"}


def test_top_features_sorted_largest_first_and_windows_differ(metrics):
    pi = metrics["permutation_importance"]
    for window, key in (("validation", "items"), ("test", "test_items")):
        want = [i["feature"] for i in sorted(pi[key], key=lambda i: -i["importance_mean"])][:3]
        assert H.top_features(metrics, window) == want
    assert H.top_features(metrics, "validation") != H.top_features(metrics, "test")
    assert H.top_features(metrics, "validation", 1) == H.top_features(metrics, "validation")[:1]
    assert H.top_features(None) == [] and H.top_features({"permutation_importance": {}}) == []


def test_association_direction_word_follows_the_sign(bundle, metrics):
    res = {"probability": 0.1, "risk_label": "Moderate"}
    down = pd.DataFrame({"extra_days": [0, 7], "promise_days": [20, 27], "probability": [0.12, 0.05], "risk_label": ["Moderate", "Low"]})
    up = down.assign(probability=[0.05, 0.12])
    flat = down.assign(probability=[0.05, 0.05])
    body = lambda w: next(i for i in H.interpretation(res, bundle, metrics, None, None, w) if i["tag"] == "Statistical association")["body"]
    assert "(lower)" in body(down) and "(higher)" in body(up) and "(unchanged)" in body(flat)
    assert "12.0% at a 20-day promise to 5.0% at 27 days" in body(down)


def test_read_first_states_the_validation_gap_with_real_numbers(bundle, metrics):
    mi = H.model_info(bundle, metrics)
    txt = H.read_first(bundle, metrics)
    assert mi["val_roc_auc"] - mi["test_roc_auc"] > 0.05
    assert f"much weaker than on the validation window ({mi['val_roc_auc']:.2f})" in txt
    close = json.loads(json.dumps(metrics))
    sel = close["selected_model"]["name"]
    close["models"][sel]["validation"]["roc_auc"] = close["models"][sel]["test"]["roc_auc"] + 0.01
    assert "validation window" not in H.read_first(bundle, close)


def test_score_batch_summary_ignores_invalid_rows(bundle):
    rows = [_row(order_ref="ok"), _row(order_ref="bad", customer_state="XX")]
    scored, s = H.score_batch(H.parse_batch_csv(_csv(rows))[0], bundle)
    assert (s["n_valid"], s["n_assumed"], s["n_warnings"]) == (1, 1, 0)
    bad = scored.iloc[1]
    assert bad["assumptions_used"] == "" and bad["input_warning"] == "" and bad["input_error"]
    assert scored.iloc[0]["assumptions_used"] != ""
    parsed = H.parse_batch_csv(_csv([_row(order_ref="x", purchase_date="2026-10-23"), _row(order_ref="y", customer_state="XX", purchase_date="2026-10-23")]))[0]
    _, s2 = H.score_batch(parsed, bundle)
    assert s2["n_warnings"] == 1 and s2["n_valid"] == 1


def test_batch_preview_helper():
    scored = pd.DataFrame({"csv_row": [2, 3, 4], "order_ref": ["a", None, "c"], "customer_state": ["RJ", "XX", "SP"],
                           "total_weight_g": [None, None, "500"], "late_probability": [0.1, np.nan, 0.2],
                           "risk_label": ["Low", None, "High"], "assumptions_used": ["x", "", ""], "input_warning": ["", "", "w"],
                           "input_error": ["", "bad", ""]})
    prev = H.batch_preview(scored)
    assert list(prev["csv_row"]) == [2, 4] and list(prev.columns[:4]) == ["csv_row", "order_ref", "late_probability", "risk_label"]
    assert list(prev["total_weight_g"]) == ["—", "500"] and "input_error" not in prev.columns
    assert list(prev["assumptions_used"]) == ["x", "—"]
    assert len(H.batch_preview(pd.concat([scored] * 200, ignore_index=True), limit=50)) == 50
    assert H.batch_preview(pd.DataFrame()).empty


def test_friendly_feature_text_replaces_identifiers_only():
    out = H.friendly_feature_text("Top-3: estimated_delivery_days, same_state; flips for total_weight_g. Plain words stay.")
    assert "_" not in out and "Promised delivery time (days)" in out and "Plain words stay." in out
    assert H.friendly_feature_text(None) == ""


# ---- chart fixes used on this page -------------------------------------------
def test_gauge_labels_sit_below_the_bar_and_tick_label_has_room():
    thr = {"moderate": 0.064, "high": 0.165}
    fig = charts.probability_gauge(0.019, thr)
    assert all(not getattr(b, "text", None) for b in fig.data)                  # no text inside the bars (the needle used to cover it)
    ann = [a for a in fig.layout.annotations if a.text in ("Low", "Moderate", "High")]
    assert [a.text for a in ann] == ["Low", "Moderate", "High"]
    needle = next(s for s in fig.layout.shapes if s.type == "line")
    assert all(a.y < min(needle.y0, needle.y1) for a in ann)                  # labels never overlap the needle
    xmax = fig.layout.xaxis.range[1]
    centres = [thr["moderate"] / 2, (thr["moderate"] + thr["high"]) / 2, (thr["high"] + xmax) / 2]
    assert [a.x for a in ann] == pytest.approx(centres)
    assert fig.layout.margin.r >= 36 and fig.layout.yaxis.range[0] < 0


def test_risk_band_chart_facet_titles_are_short_and_labels_clear_the_baseline(metrics):
    fig = charts.risk_band_chart(metrics)
    titles = [a.text for a in fig.layout.annotations]
    sp = metrics["split"]
    assert titles[:2] == [f"Validation ({100 * sp['validation']['late_rate']:.1f}% late)", f"Test ({100 * sp['test']['late_rate']:.1f}% late)"]
    assert all(len(t) <= 24 for t in titles[:2])
    base = [sp["validation"]["late_rate"], sp["test"]["late_rate"]]
    for tr, b, w in zip([t for t in fig.data if getattr(t, "mode", None) == "text"], base, ("validation", "test")):
        rates = [r["late_rate"] for r in metrics["risk_bands"][w]]
        for y, r in zip(tr.y, rates):
            assert y >= r                                                      # labels never drop below their bar top
    assert "<br>" in fig.layout.title.subtitle.text or len(fig.layout.title.subtitle.text) <= 60


def test_importance_chart_title_is_short_enough_for_a_phone(metrics):
    fig = charts.feature_importance_chart(metrics, window="test")
    assert len(fig.layout.title.text) <= 32 and "Permutation importance" in fig.layout.title.subtitle.text
    assert not any(str(t).strip().startswith("-0.000") for t in fig.data[1].text)


# =========================================================================== #
# Review follow-ups: server-log warning, stale-result notice, mutation survivors
# =========================================================================== #
@pytest.fixture
def widget_log(monkeypatch):
    """Capture the streamlit logger and re-arm its one-shot 'default value + Session State' warning."""
    import logging
    from streamlit.elements.lib import policies
    monkeypatch.setattr(policies, "_shown_default_value_warning", False)
    records: list[str] = []

    class Grab(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())
    h, lg = Grab(level=logging.WARNING), logging.getLogger("streamlit")
    lg.addHandler(h)
    yield records
    lg.removeHandler(h)


def _bad_widget_logs(records: list[str]) -> list[str]:
    return [r for r in records if "default value" in r and "Session State" in r]


def test_no_widget_default_vs_session_state_warning_on_first_visit_and_on_return(widget_log):
    at = _predict_route("SP", "AM")                          # route change + submit: all widgets keep their own state
    at.number_input(key=_promise_key(at)).set_value(60)
    at = _submit(at)
    assert not at.exception and _bad_widget_logs(widget_log) == []
    last = at.session_state["pred_last"]
    back = AppTest.from_file(PAGE, default_timeout=TIMEOUT)  # navigated away and back: widget state gone, result kept
    back.session_state["pred_last"] = last
    back.run()
    assert not back.exception and _bad_widget_logs(widget_log) == []
    assert back.selectbox(key="pred_customer_state").value == "AM" and back.number_input(key=_promise_key(back)).value == 60


def test_page_source_never_sets_widget_defaults_next_to_session_state():
    """Starting values go through session_state only: widgets must not also receive index= / a non-None value=."""
    import ast
    tree = ast.parse(Path(PAGE).read_text(encoding="utf-8"))
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr in ("selectbox", "number_input", "date_input")):
        key = next((k.value.value if isinstance(k.value, ast.Constant) else "<dynamic>" for k in call.keywords if k.arg == "key"), None)
        assert key is not None
        kw = {k.arg: k.value for k in call.keywords}
        assert "index" not in kw, key
        if "value" in kw:
            assert isinstance(kw["value"], ast.Constant) and kw["value"].value is None, key


# ---- stale-result notice ------------------------------------------------------
def test_stale_notice_for_route_change_and_standing_note_for_uncommitted_form_edits():
    """Widgets inside st.form only deliver their values on submit, so the page cannot see uncommitted edits (verified:
    AppTest, like the browser, keeps returning the submitted value).  Hence: a standing 'press Predict' note beside every
    result, and a notice naming the changed fields as soon as the outside-the-form route selectors differ."""
    at = _submit(_fresh())
    html = _html(at)
    assert "previously submitted order" not in html and "Edited the form?" in html and "Predict Delay Risk" in html
    at.number_input(key=_promise_key(at)).set_value(60).run(timeout=TIMEOUT)      # uncommitted: form keeps the submitted value
    assert _inputs_from_widgets(at)["estimated_delivery_days"] == at.session_state["pred_last"]["inputs"]["estimated_delivery_days"]
    assert "Edited the form?" in _html(at)
    other = next(c for c in at.selectbox(key="pred_customer_state").options if c != at.selectbox(key="pred_customer_state").value)
    at.selectbox(key="pred_customer_state").select(other).run(timeout=TIMEOUT)
    html = _html(at)
    assert "previously submitted order" in html and "Edited the form?" not in html
    changed = html.split("changed:")[1].split(")")[0]
    assert H.FIELD_LABELS["customer_state"] in changed and H.FIELD_LABELS["estimated_delivery_days"] in changed
    assert "previously submitted order" not in _html(_submit(at))


def test_render_result_compares_the_whole_input_dict(monkeypatch):
    """Any difference between the stored order and the current form values (not only the route) triggers the notice."""
    at = _submit(_fresh())
    last = at.session_state["pred_last"]
    stale = {**last, "inputs": {**last["inputs"], "total_price": last["inputs"]["total_price"] + 111.0}}
    at.session_state["pred_last"] = stale
    at.run(timeout=TIMEOUT)
    html = _html(at)
    assert "previously submitted order" in html and "changed: " + H.FIELD_LABELS["total_price"] in html


# ---- mutation survivors --------------------------------------------------------
def test_historical_context_category_block_matches_pandas(orders):
    cat = orders["product_category"].astype(str).value_counts().index[0]
    ctx = H.historical_context(orders, "SP", "RJ", cat)
    sub = orders[orders["product_category"].astype(str) == cat]
    assert ctx["category"]["n"] == len(sub) and ctx["category"]["n"] < len(orders)
    assert ctx["category"]["late_rate"] == pytest.approx(sub["is_late"].astype(float).mean())
    assert ctx["category"]["n_late"] == int(sub["is_late"].astype(bool).sum())


def test_historical_context_small_flag_threshold_and_wilson_fallback(orders):
    n = len(_route(orders, "SP", "RJ"))
    assert H.historical_context(orders, "SP", "RJ", "x", min_orders=n)["route"]["small"] is False
    assert H.historical_context(orders, "SP", "RJ", "x", min_orders=n + 1)["route"]["small"] is True
    zero = H.historical_context(orders, "ZZ", "ZZ", "x")["route"]
    assert zero["n"] == 0 and zero["small"] is True and np.isnan(zero["ci_low"]) and np.isnan(zero["ci_high"])


def test_interpretation_ratio_sentence_is_probability_over_test_rate(bundle, metrics):
    t = metrics["split"]["test"]["late_rate"]
    for p in (0.5 * t, 3.0 * t):
        body = H.interpretation({"probability": p, "risk_label": "Moderate"}, bundle, metrics, None)[0]["body"]
        assert f"This estimate is {p / t:.1f}x the test-window rate" in body
    body = H.interpretation({"probability": t, "risk_label": "Low"}, bundle, metrics, None)[0]["body"]
    assert "about the same as the test-window rate" in body


def test_interpretation_small_sample_sentence_only_for_small_routes(bundle, metrics, orders):
    res = {"probability": 0.1, "risk_label": "Moderate"}
    n = len(_route(orders, "SP", "RJ"))
    big = H.historical_context(orders, "SP", "RJ", "bed_bath_table")
    small = H.historical_context(orders, "SP", "RJ", "bed_bath_table", min_orders=n + 1)
    body = lambda h, r=("SP", "RJ"): next(i for i in H.interpretation(res, bundle, metrics, h, r) if i["tag"] == "Observed")["body"]
    assert "Small sample" not in body(big)
    assert f"Small sample (n = {n:,}, below {n + 1:,})" in body(small)
    nothing = H.historical_context(orders, "ZZ", "ZZ", "bed_bath_table")
    assert "no historical delivered orders on the route ZZ to ZZ" in body(nothing, ("ZZ", "ZZ"))


def test_invalid_resubmission_removes_the_previous_result():
    at = _submit(_fresh())
    assert "pred_last" in at.session_state and "2. Result" in _html(at)
    at.number_input(key="pred_items").set_value(0)
    at = _submit(at)
    html = _html(at)
    assert "pred_last" not in at.session_state and "Please fix" in html
    assert "2. Result" not in html and "No prediction yet" in html and "Late-delivery probability" not in html


def test_scenario_chart_colours_follow_the_band_and_threshold_lines_are_drawn():
    thr = {"moderate": 0.064, "high": 0.165}
    sc = [{"label": "A", "probability": 0.02, "risk_label": "Low"}, {"label": "B", "probability": 0.10, "risk_label": "Moderate"},
          {"label": "C", "probability": 0.30, "risk_label": "High"}]
    fig = charts.scenario_comparison_chart(sc, thr)
    assert list(fig.data[0].marker.color) == [charts.RISK_COLORS[b] for b in ("Low", "Moderate", "High")]
    assert len(set(fig.data[0].marker.color)) == 3
    lines = [s for s in fig.layout.shapes if s.type == "line" and s.line.dash == "dash"]
    assert sorted(s.y0 for s in lines) == pytest.approx([0.064, 0.165])
    notes = " ".join(a.text for a in fig.layout.annotations)
    assert "Moderate from 6.4%" in notes and "High from 16.5%" in notes
    assert not [s for s in charts.scenario_comparison_chart(sc, None).layout.shapes if s.type == "line"]


# =========================================================================== #
# Second review round: warnings wording, text facts, same-state Low band, KPI layout, mutation survivors
# =========================================================================== #
RAW_FIELD_RE = re.compile(r"\b(" + "|".join(sorted(H.FIELD_LABELS, key=len, reverse=True)) + r")\b")


def test_humanise_warning_replaces_every_raw_field_name(bundle, orders):
    inp = {**_example_inputs(bundle, orders), "estimated_delivery_days": 3, "total_price": 90_000.0, "total_freight": 5_000.0,
           "n_items": 40, "total_weight_g": 900_000, "purchase_date": "2026-10-23"}
    raw = prediction.predict_risk(inp, bundle)["warnings"]
    ratio_inp = {**_example_inputs(bundle, orders), "total_price": 10.0, "total_freight": 200.0}   # each in range, the ratio is not
    raw += prediction.predict_risk(ratio_inp, bundle)["warnings"]
    assert len(raw) >= 5 and any("freight_ratio" in w for w in raw) and any("purchase_date" in w for w in raw)
    for w in raw:
        nice = H.humanise_warning(w)
        assert not RAW_FIELD_RE.search(nice) and "_" not in nice.split("(")[0], nice
        assert nice[0].isupper()
    assert H.humanise_warning("freight_ratio (freight / price) = 57.8 is outside the range the model was trained on (0 to 1)") \
        .startswith("Freight-to-price ratio = 57.8 is outside")
    assert H.humanise_warning("estimated_delivery_days = 3 is outside the range (8 to 156)").startswith("Delivery promise = 3")


def test_out_of_range_warnings_on_the_page_use_labels_not_column_names():
    at = _fresh()
    at.date_input(key="pred_date").set_value(datetime.date(2026, 10, 23))
    at.number_input(key="pred_price").set_value(90_000.0)
    at.number_input(key=_promise_key(at)).set_value(3)
    at = _submit(at)
    assert not at.exception
    texts = [str(m.value) for m in at.get("html")] + [str(m.value) for m in at.markdown]
    watch = [t for t in texts if "Outside what the model was trained on" in t]
    assert len(watch) >= 3
    body = " ".join(t for t in texts if "is outside the" in t)
    for raw in ("purchase_date", "total_price", "estimated_delivery_days", "freight_ratio"):
        assert raw not in body, raw
    for label in ("Purchase date is outside the dates the model was trained on", "Total product value = 90,000",
                  "Delivery promise = 3"):
        assert label in body, label


def test_historical_limitation_names_the_training_years_not_2016(bundle, metrics):
    mi = H.model_info(bundle, metrics)
    line = next(ln for ln in H.limitations(bundle, metrics) if ln.startswith("**Historical data only.**"))
    assert f"placed in {mi['train_start'][:4]}-{mi['test_end'][:4]}" in line and "2016" not in line
    assert str(metrics["split"]["train"]["start"])[:4] == "2017"          # the 2016 orders were excluded from training
    assert "2016" not in " ".join(H.limitations(bundle, metrics))
    # without any window information the sentence still reads correctly
    bare = next(ln for ln in H.limitations({"model_name": "x"}, None) if ln.startswith("**Historical data only.**"))
    assert "Olist orders; it says nothing" in bare


def test_limitation_lines_include_the_weaker_on_later_data_line(bundle, metrics):
    mi = H.model_info(bundle, metrics)
    line = next(ln for ln in H.limitations(bundle, metrics) if ln.startswith("**Weaker on later data.**"))
    assert f"ROC-AUC was {mi['val_roc_auc']:.2f}" in line and f"fell to {mi['test_roc_auc']:.2f}" in line
    assert next(ln for ln in H.limitations(bundle, metrics) if ln.startswith("**Modest signal.**"))


def _low_note(metrics, bundle, route, p=0.02):
    items = H.interpretation({"probability": p, "risk_label": "Low"}, bundle, metrics, None, route)
    return items[-1]


def test_low_band_same_state_orders_get_their_own_rates_and_no_green_reassurance(bundle, metrics):
    tbs = metrics["risk_bands"]["test_by_same_state"]
    overall = metrics["split"]["test"]["late_rate"]
    same, other = tbs["same_state"]["bands"]["Low"], tbs["other_state"]["bands"]["Low"]
    pooled = next(r for r in metrics["risk_bands"]["test"] if r["band"] == "Low")
    assert same["late_rate"] >= 0.9 * overall > other["late_rate"]           # the stored diagnostics really differ by route type
    a = _low_note(metrics, bundle, ("SP", "SP"))
    assert a["kind"] == "watch" and "Routine monitoring is enough" not in a["body"]
    assert f"{100 * same['late_rate']:.1f}%" in a["body"] and f"n = {same['n']:,}" in a["body"]
    assert f"{100 * pooled['late_rate']:.1f}%" in a["body"] and f"{100 * overall:.1f}%" in a["body"]
    b = _low_note(metrics, bundle, ("SP", "RJ"))
    assert b["kind"] == "success" and "Routine monitoring is enough" in b["body"]
    assert f"{100 * other['late_rate']:.1f}%" in b["body"] and f"n = {other['n']:,}" in b["body"] and "across states" in b["body"]
    c = _low_note(metrics, bundle, None)                                      # no route known: pooled Low-band wording as before
    assert c["kind"] == "success" and f"n = {pooled['n']:,}" in c["body"]
    # a non-Low band never gets the Low wording
    mod = H.interpretation({"probability": 0.1, "risk_label": "Moderate"}, bundle, metrics, None, ("SP", "SP"))[-1]
    assert mod["kind"] == "watch" and "Do not read Low" not in mod["body"]


def test_same_state_low_result_on_the_page_is_not_a_green_box():
    at = _submit(_fresh())                                                    # default order is SP to SP
    inp = at.session_state["pred_last"]["inputs"]
    res = at.session_state["pred_last"]["result"]
    assert inp["seller_state"] == inp["customer_state"]
    if res["risk_label"] == "Low":
        html = _html(at)
        assert "Do not read Low as safe for this kind of order" in html and "Routine monitoring is enough" not in html


def test_kpi_label_rows_share_a_min_height_and_dates_do_not_wrap():
    src = Path(PAGE).read_text(encoding="utf-8")
    assert re.search(r"\.diq-kpi-top\{[^}]*min-height", src)
    at = _submit(_fresh())
    assert "\u2011" in _cards(at)["Test-window late rate"]["sub"]


def test_importance_chart_axis_title_is_short_and_window_is_in_the_subtitle(metrics):
    for w in ("validation", "test"):
        fig = charts.feature_importance_chart(metrics, window=w)
        assert fig.layout.xaxis.title.text == "Drop in PR-AUC" and len(fig.layout.xaxis.title.text) <= 20
        assert fig.layout.title.subtitle.text.startswith(f"{w.capitalize()} window.")


def _wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return c - h, c + h


def test_observed_text_prints_the_wilson_interval_of_the_route(bundle, metrics, orders):
    hist = H.historical_context(orders, "SP", "RJ", "bed_bath_table")
    r = _route(orders, "SP", "RJ")
    k, n = int(r["is_late"].astype(float).sum()), len(r)
    lo, hi = _wilson(k, n)
    assert (hist["route"]["ci_low"], hist["route"]["ci_high"]) == pytest.approx((lo, hi), abs=1e-9)
    body = next(i for i in H.interpretation({"probability": 0.1, "risk_label": "Moderate"}, bundle, metrics, hist, ("SP", "RJ"))
                if i["tag"] == "Observed")["body"]
    assert f"95% CI {100 * lo:.1f}% to {100 * hi:.1f}%, n = {n:,}" in body
    assert f"{100 * k / n * 0.9:.1f}% to {100 * k / n * 1.1:.1f}%" not in body


def test_model_prediction_sentence_quotes_the_band_rate_of_each_window(bundle, metrics):
    for label in ("Low", "Moderate", "High"):
        body = H.interpretation({"probability": 0.1, "risk_label": label}, bundle, metrics, None)[0]["body"]
        v = next(r for r in metrics["risk_bands"]["validation"] if r["band"] == label)
        t = next(r for r in metrics["risk_bands"]["test"] if r["band"] == label)
        assert v["late_rate"] != pytest.approx(t["late_rate"], abs=5e-4)
        assert (f"late {100 * v['late_rate']:.1f}% of the time on the validation window (n = {v['n']:,}) "
                f"and {100 * t['late_rate']:.1f}% on the later test window (n = {t['n']:,})") in body


def test_result_shows_the_trust_and_band_separation_callouts():
    at = _submit(_fresh())
    res = at.session_state["pred_last"]["result"]
    assert res["probability_note"]                                            # so the assertion below is not vacuous
    html = _html(at)
    for title in ("How much to trust this number", "Do the bands separate?", "Limitations: please read", "Read this first"):
        assert title in html, title
    assert res["probability_note"][:40] in html.replace("&#x27;", "'")


def test_batch_page_ends_with_the_use_with_care_callout(downloads):
    at = _upload([_row(order_ref="a")])
    assert not at.exception
    assert "Use the scores with care" in _html(at)
    assert "Use the scores with care" not in _html(_upload([_row(customer_state="XX")]))      # nothing scored: no scores to caution about


def test_batch_row_limit_boundary_is_inclusive(monkeypatch):
    monkeypatch.setattr(H, "MAX_BATCH_ROWS", 2)
    df, _ = H.parse_batch_csv(_csv([_row(), _row()]))                         # exactly the limit is fine
    assert len(df) == 2
    with pytest.raises(H.BatchFileError, match="3 rows"):
        H.parse_batch_csv(_csv([_row(), _row(), _row()]))


def test_probability_gauge_bar_colours_and_needle_position():
    thr = {"moderate": 0.064, "high": 0.165}
    for p in (0.019, 0.10, 0.30):
        fig = charts.probability_gauge(p, thr)
        bars = [t for t in fig.data if t.type == "bar"]
        assert [b.marker.color for b in bars] == [charts.PALETTE["green"], charts.PALETTE["amber"], charts.PALETTE["red"]]
        assert [b.base[0] for b in bars] == pytest.approx([0.0, 0.064, 0.165])
        needle = next(s for s in fig.layout.shapes if s.type == "line")
        assert needle.x0 == needle.x1 == pytest.approx(p)
        marker = next(t for t in fig.data if t.type == "scatter")
        assert marker.x[0] == pytest.approx(p)


# --------------------------------------------------------------------------- #
# Review round: result flow
# --------------------------------------------------------------------------- #
def test_fmt_value_shows_whole_numbers_without_decimals():
    assert H.fmt_value(1.0) == "1" and H.fmt_value(800.0) == "800" and H.fmt_value(1234.0) == "1,234"
    assert H.fmt_value(3.5) == "3.5" and H.fmt_value(2.04) == "2.0"          # one decimal otherwise
    assert H.fmt_value(float("nan")) == "n/a" and H.fmt_value("x") == "x"


def test_result_flow_scrolls_to_the_result_and_pin_sits_next_to_the_numbers():
    src = Path(PAGE).read_text(encoding="utf-8")
    assert 'st.container(key="pred-result")' in src and "scrollIntoView" in src and "unsafe_allow_javascript=True" in src
    assert 'st.spinner("Scoring this order and preparing the result' in src
    # the pin button is created right after the headline KPI row, before the gauge, not at the bottom of the result
    body = src[src.index("def render_result"):src.index("def render_scenarios")]
    assert body.index("render_pin()") < body.index("probability_gauge")
    assert body.count('key="pred_pin"') == 0 and src.count('key="pred_pin"') == 1


def test_limitations_are_collapsed_in_the_result_but_still_present(bundle):
    at = _fresh()
    at = _submit(at)
    assert not at.exception
    labels = [e.label for e in at.expander]
    assert any(l.startswith("Limitations: please read") for l in labels)
    assert "Limitations: please read" in _html(at)
    # the page-level notice at the top is the full warning; the result does not repeat it as an open callout
    assert _html(at).count("Read this first") == 1


def test_assumption_values_have_no_spurious_decimals(bundle):
    at = _fresh()
    at = _submit(at)
    assert not at.exception
    df = [d for d in at.dataframe if getattr(d, "key", None) == "pred_assumptions"]
    if df:                                     # blank optional inputs were filled with training medians
        vals = df[0].value["Value used"].tolist()
        assert all(not v.endswith(".0") for v in vals), vals
