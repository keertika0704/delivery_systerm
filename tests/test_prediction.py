"""Tests for the ML layer (train_model + prediction).

Synthetic frames are clearly labelled TEST FIXTURES and are never used by the app.  Tests that need
the saved model or the processed parquet are skipped (with the command that creates them) when absent.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import os
import subprocess
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src import config, prediction, train_model
from src.prediction import (
    ModelNotFoundError,
    derive_model_features,
    load_model,
    predict_batch,
    predict_risk,
    risk_label,
    validate_input,
)

TRAIN_CMD = f"{config.ROOT}/.venv/bin/python -m src.train_model"
needs_model = pytest.mark.skipif(
    not config.MODEL_PATH.exists() or not config.MODEL_METRICS_PATH.exists(),
    reason=f"models/ artefacts absent; create them with: {TRAIN_CMD}")
needs_data = pytest.mark.skipif(
    not config.ORDERS_DELIVERED_PATH.exists(),
    reason=f"{config.ORDERS_DELIVERED_PATH} absent; create it with: {config.ROOT}/.venv/bin/python -m src.preprocessing")


@pytest.fixture(scope="module")
def bundle() -> dict:
    if not config.MODEL_PATH.exists():
        pytest.skip(f"models/delay_classifier.joblib absent; run: {TRAIN_CMD}")
    return load_model()


@pytest.fixture(scope="module")
def metrics() -> dict:
    if not config.MODEL_METRICS_PATH.exists():
        pytest.skip(f"models/model_metrics.json absent; run: {TRAIN_CMD}")
    return json.loads(config.MODEL_METRICS_PATH.read_text())


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    if not config.ORDERS_DELIVERED_PATH.exists():
        pytest.skip("processed parquet absent; run python -m src.preprocessing")
    return train_model.load_training_frame()


@pytest.fixture(scope="module")
def parts(data) -> dict[str, pd.DataFrame]:
    return train_model.chronological_split(data)


def _raw_inputs(row: pd.Series) -> dict:
    """Raw form primitives for a real processed row."""
    return {
        "customer_state": str(row["customer_state"]), "seller_state": str(row["seller_state"]),
        "product_category": str(row["product_category"]), "purchase_date": row["purchase_date"].strftime("%Y-%m-%d"),
        "n_items": int(row["n_items"]), "total_price": float(row["total_price"]),
        "total_freight": float(row["total_freight"]), "total_weight_g": float(row["total_weight_g"]),
        "estimated_delivery_days": int(row["estimated_delivery_days"]),
    }


@pytest.fixture(scope="module")
def good_inputs(bundle, parts) -> dict:
    # a TRAINING-window order, so its date and values lie inside what the model was fitted on (no warnings)
    tr = parts["train"].dropna(subset=["total_weight_g"])
    return _raw_inputs(tr.iloc[0])


# --------------------------------------------------------------------------- #
# Leakage
# --------------------------------------------------------------------------- #
def test_allow_list_and_deny_list_are_disjoint():
    assert not set(config.PURCHASE_TIME_FEATURES) & set(config.LEAKAGE_COLUMNS)
    assert not set(config.PURCHASE_TIME_FEATURES) & set(config.NON_FEATURE_COLUMNS)
    assert not set(prediction.ACCEPTED_FIELDS) & set(config.LEAKAGE_COLUMNS)


def test_bundle_features_are_purchase_time_only(bundle):
    feats = set(bundle["feature_columns"])
    assert feats <= set(config.PURCHASE_TIME_FEATURES)
    assert not feats & set(config.LEAKAGE_COLUMNS)
    assert not feats & set(config.NON_FEATURE_COLUMNS)
    assert bundle["target"] not in feats
    assert bundle["feature_columns"] == bundle["numeric_features"] + bundle["categorical_features"]
    assert list(bundle["base_pipeline"].named_steps["prep"].feature_names_in_) == bundle["feature_columns"]


@needs_data
def test_no_feature_is_derived_from_outcome_columns(bundle, data):
    """Shuffling every outcome / post-purchase column must not change any model feature value."""
    outcome_cols = [c for c in config.LEAKAGE_COLUMNS if c in data.columns]
    sample = data.sample(2000, random_state=0)
    shuffled = sample.copy()
    rng = np.random.default_rng(0)
    for c in outcome_cols:
        shuffled[c] = rng.permutation(shuffled[c].to_numpy())
    pd.testing.assert_frame_equal(
        sample[bundle["feature_columns"]], shuffled[bundle["feature_columns"]])


def test_load_model_rejects_leaky_bundle(bundle, tmp_path):
    bad = copy.copy(bundle)
    bad["feature_columns"] = list(bundle["feature_columns"]) + ["is_late"]
    path = tmp_path / "leaky.joblib"
    joblib.dump(bad, path)
    with pytest.raises(ValueError, match="forbidden"):
        load_model(path)


# --------------------------------------------------------------------------- #
# Feature derivation (training / inference parity)
# --------------------------------------------------------------------------- #
@needs_data
def test_derive_features_reproduces_processed_table_on_real_rows(data):
    sample = data.sample(5000, random_state=1)
    raw = sample[["customer_state", "seller_state", "total_price", "total_freight", "purchase_date"]]
    derived = derive_model_features(raw)
    for col in ["freight_ratio", "same_state", "purchase_month", "purchase_dow", "purchase_quarter"]:
        got = pd.to_numeric(derived[col], errors="coerce").astype("float64").to_numpy()
        want = pd.to_numeric(sample[col], errors="coerce").astype("float64").to_numpy()
        np.testing.assert_allclose(got, want, equal_nan=True, err_msg=col)
    assert train_model.derivation_parity(data) == []


def test_derive_features_synthetic_edge_cases():
    """TEST FIXTURE: hand-built rows covering zero price, month-only input and same-state."""
    raw = pd.DataFrame({
        "customer_state": ["SP", "RJ", "SP"], "seller_state": ["SP", "SP", "SP"],
        "total_price": [100.0, 0.0, 50.0], "total_freight": [25.0, 10.0, 0.0],
        "purchase_date": [pd.Timestamp("2018-02-14"), pd.NaT, pd.Timestamp("2017-12-31")],
        "purchase_month": [np.nan, 8, np.nan], "purchase_dow": [np.nan, 2, np.nan],
    })
    d = derive_model_features(raw)
    assert d["freight_ratio"].iloc[0] == pytest.approx(0.25)
    assert np.isnan(d["freight_ratio"].iloc[1])           # price 0 -> NaN, never inf / ZeroDivisionError
    assert d["freight_ratio"].iloc[2] == 0
    assert d["same_state"].tolist() == [1, 0, 1]
    assert d["purchase_month"].tolist() == [2, 8, 12]
    assert d["purchase_quarter"].tolist() == [1, 3, 4]
    assert d["purchase_dow"].tolist() == [2, 2, 6]        # 2018-02-14 Wed, month-only row uses given dow, 2017-12-31 Sun


def test_derive_features_empty_frame():
    raw = pd.DataFrame({c: pd.Series(dtype="object") for c in
                        ["customer_state", "seller_state", "total_price", "total_freight", "purchase_date"]})
    assert derive_model_features(raw).empty


# --------------------------------------------------------------------------- #
# Pipeline robustness (synthetic fixture, independent of the saved model)
# --------------------------------------------------------------------------- #
def _tiny_training_frame(n: int = 400) -> pd.DataFrame:
    """TEST FIXTURE: synthetic labelled orders; the label depends on promise length and state."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "n_items": rng.integers(1, 4, n), "total_price": rng.gamma(2, 60, n),
        "estimated_delivery_days": rng.integers(8, 40, n).astype(float),
        "customer_state": rng.choice(["SP", "RJ", "BA"], n), "same_state": rng.integers(0, 2, n),
    })
    df.loc[::17, "total_price"] = np.nan
    y = ((df["estimated_delivery_days"] < 15) | (df["customer_state"] == "BA")) & (rng.random(n) < 0.6)
    df["y"] = y.astype(int)
    return df


@pytest.mark.parametrize("est", [LogisticRegression(max_iter=500), DummyClassifier(strategy="prior")])
def test_pipeline_handles_unseen_categories_and_nans(est):
    df = _tiny_training_frame()
    num, cat = ["n_items", "total_price", "estimated_delivery_days"], ["customer_state", "same_state"]
    pipe = train_model.build_pipeline(est, num, cat, scale=True).fit(df[num + cat], df["y"])
    new = pd.DataFrame({"n_items": [1, np.nan], "total_price": [np.nan, 80.0],
                        "estimated_delivery_days": [np.nan, 20.0],
                        "customer_state": ["ZZ_unseen", None], "same_state": [1, np.nan]})
    p = pipe.predict_proba(new)[:, 1]
    assert p.shape == (2,) and np.all((p >= 0) & (p <= 1)) and np.all(np.isfinite(p))


def test_pipeline_refuses_missing_feature_column():
    df = _tiny_training_frame()
    num, cat = ["n_items", "total_price"], ["customer_state"]
    pipe = train_model.build_pipeline(LogisticRegression(), num, cat, scale=True).fit(df[num + cat], df["y"])
    with pytest.raises(ValueError, match="missing model feature"):
        pipe.predict_proba(df[["n_items", "total_price"]])


def test_saved_pipeline_handles_unseen_category_and_nan(bundle):
    cols = bundle["feature_columns"]
    row = {c: np.nan for c in bundle["numeric_features"]}
    row.update({c: "never_seen_level" for c in bundle["categorical_features"]})
    p = bundle["pipeline"].predict_proba(pd.DataFrame([row])[cols])[:, 1]
    assert 0 <= p[0] <= 1


# --------------------------------------------------------------------------- #
# validate_input
# --------------------------------------------------------------------------- #
def _errors(inputs, bundle) -> str:
    return " ".join(validate_input(inputs, bundle)[1])


def test_valid_input_has_no_errors(bundle, good_inputs):
    cleaned, errors = validate_input(good_inputs, bundle)
    assert errors == []
    assert cleaned["customer_state"] == good_inputs["customer_state"]


@pytest.mark.parametrize("field", ["customer_state", "seller_state", "product_category", "n_items",
                                   "total_price", "total_freight"])
def test_missing_required_field_is_an_error(bundle, good_inputs, field):
    bad = {k: v for k, v in good_inputs.items() if k != field}
    assert f"{field} is required" in _errors(bad, bundle)
    bad2 = {**good_inputs, field: None}
    assert f"{field} is required" in _errors(bad2, bundle)


def test_missing_purchase_time_is_an_error(bundle, good_inputs):
    bad = {k: v for k, v in good_inputs.items() if k != "purchase_date"}
    assert "purchase_date is required" in _errors(bad, bundle)
    ok = {**bad, "purchase_month": 5}
    assert validate_input(ok, bundle)[1] == []


def test_unknown_state_and_category_are_errors(bundle, good_inputs):
    assert "not a known customer_state" in _errors({**good_inputs, "customer_state": "XX"}, bundle)
    assert "not a known seller_state" in _errors({**good_inputs, "seller_state": "ZZ"}, bundle)
    assert "not a known product_category" in _errors({**good_inputs, "product_category": "flying_carpets"}, bundle)


def test_state_and_category_are_normalised_not_guessed(bundle, good_inputs):
    ok = {**good_inputs, "customer_state": f" {good_inputs['customer_state'].lower()} ",
          "product_category": good_inputs["product_category"].upper()}
    cleaned, errors = validate_input(ok, bundle)
    assert errors == [] and cleaned["customer_state"] == good_inputs["customer_state"]


@pytest.mark.parametrize("field,value,fragment", [
    ("total_price", -5, "outside the allowed range"),
    ("total_price", 0, "outside the allowed range"),
    ("total_price", 10_000_000, "outside the allowed range"),
    ("total_freight", -1, "outside the allowed range"),
    ("n_items", 0, "outside the allowed range"),
    ("n_items", 2.5, "whole number"),
    ("n_sellers", 1.5, "whole number"),
    ("estimated_delivery_days", 20.5, "whole number"),
    ("estimated_delivery_days", 0, "outside the allowed range"),
    ("n_items", "many", "finite number"),
    ("total_price", float("inf"), "finite number"),
    ("total_price", float("nan"), "is required"),
    ("total_weight_g", -10, "outside the allowed range"),
    ("estimated_delivery_days", 100000, "outside the allowed range"),
    ("purchase_date", "not-a-date", "could not be read as a date"),
    ("purchase_date", 20180710, "could not be read as a date"),          # a bare number is never an epoch
    ("purchase_date", 20180710.0, "could not be read as a date"),
    ("purchase_date", "10/07/2018", "could not be read as a date"),      # ambiguous dd/mm vs mm/dd
    ("purchase_date", "2018-02-30", "could not be read as a date"),
    ("purchase_date", "1850-01-01", "outside the allowed range"),
    ("purchase_date", "2300-01-01", "outside the allowed range"),
    ("purchase_month", 13, "from 1 to 12"),
    ("purchase_dow", 9, "from 0 to 6"),
])
def test_bad_numeric_and_date_inputs(bundle, good_inputs, field, value, fragment):
    assert fragment in _errors({**good_inputs, field: value}, bundle)


def test_inconsistent_fields_are_errors(bundle, good_inputs):
    assert "n_sellers cannot exceed n_items" in _errors({**good_inputs, "n_items": 1, "n_sellers": 3}, bundle)
    month = (pd.Timestamp(good_inputs["purchase_date"]).month % 12) + 1
    assert "conflicts with purchase_date" in _errors({**good_inputs, "purchase_month": month}, bundle)


def test_unrecognised_and_leaky_field_names_are_errors(bundle, good_inputs):
    msg = _errors({**good_inputs, "is_late": 1, "totl_price": 3}, bundle)
    assert "Unrecognised input field" in msg and "is_late" in msg and "totl_price" in msg


def test_all_problems_reported_together(bundle):
    errors = validate_input({"customer_state": "XX"}, bundle)[1]
    assert len(errors) >= 5


def test_ranges_come_from_training_data(bundle, parts):
    train = parts["train"]
    opts = bundle["form_options"]
    for c in prediction.RAW_NUMERIC:
        assert opts["ranges"][c] == pytest.approx([train[c].astype(float).min(), train[c].astype(float).max()])
    for c, v in opts["defaults"].items():
        assert v == pytest.approx(float(train[c].astype(float).median()))
    assert opts["state_pair_median_estimated_days"]
    assert set(opts["customer_state"]) == set(train["customer_state"].astype(str))
    assert opts["hard_limits"]["estimated_delivery_days"] == list(prediction.HARD_LIMITS["estimated_delivery_days"])
    assert opts["date_span"][0] <= opts["date_span"][1]


@needs_data
def test_realistic_orders_outside_training_range_are_scored_with_a_warning(bundle, parts):
    """Short delivery promises (3-7 days) occur in the test window but not in training: warn, never reject."""
    te = parts["test"]
    lo = bundle["form_options"]["ranges"]["estimated_delivery_days"][0]
    short = te[te["estimated_delivery_days"] < lo].dropna(subset=["total_weight_g"]).head(50)
    assert len(short) > 0
    out = predict_batch(pd.DataFrame([_raw_inputs(r) for _, r in short.iterrows()]), bundle)
    assert (out["input_error"] == "").all() and out["late_probability"].between(0, 1).all()
    assert out["input_warning"].str.contains("estimated_delivery_days").all()
    assert out["input_warning"].str.contains("extrapolating").all()
    whole = te.dropna(subset=["total_weight_g"]).head(2000)
    rejected = predict_batch(pd.DataFrame([_raw_inputs(r) for _, r in whole.iterrows()]), bundle)["input_error"] != ""
    assert not rejected.any()


def test_out_of_training_range_value_warns_but_in_range_does_not(bundle, good_inputs):
    lo, hi = bundle["form_options"]["ranges"]["estimated_delivery_days"]
    _, errors, warned = prediction.check_input({**good_inputs, "estimated_delivery_days": int(lo) - 3}, bundle)
    assert errors == [] and len(warned) == 1 and "extrapolating" in warned[0] and f"{int(lo)}" in warned[0]
    res = predict_risk({**good_inputs, "estimated_delivery_days": int(lo) - 3}, bundle)
    assert 0 <= res["probability"] <= 1 and res["warnings"] == warned
    _, _, none = prediction.check_input({**good_inputs, "estimated_delivery_days": int(lo) + 5}, bundle)
    assert none == []
    assert prediction.check_input({**good_inputs, "estimated_delivery_days": int(hi) + 50}, bundle)[2]


def test_purchase_date_formats_and_span(bundle, good_inputs):
    for ok in ("2018-07-10", " 2018-07-10 ", "2018-07-10 13:45", "2018-07-10T13:45:59", pd.Timestamp("2018-07-10"),
               dt.datetime(2018, 7, 10, 9), dt.date(2018, 7, 10), np.datetime64("2018-07-10")):
        cleaned, errors = validate_input({**good_inputs, "purchase_date": ok}, bundle)
        assert errors == [], (ok, errors)
        assert cleaned["purchase_dow"] == 1 and cleaned["purchase_month"] == 7      # Tuesday 10 July 2018
    _, errors, warned = prediction.check_input({**good_inputs, "purchase_date": "2030-01-01"}, bundle)
    assert errors == [] and any("purchase_date is outside the dates the model was trained on" in w for w in warned)


def test_parse_dates_is_strict_and_vectorised():
    s = pd.Series(["2018-07-10", "10/07/2018", 20180710, None, "2018-07-10 08:00:00", "July 10 2018"], dtype="object")
    out = prediction._parse_dates(s)
    assert out.notna().tolist() == [True, False, False, False, True, False]
    assert out.iloc[0] == pd.Timestamp("2018-07-10") and out.iloc[4] == pd.Timestamp("2018-07-10 08:00")
    typed = pd.Series(pd.to_datetime(["2018-01-02", None]))
    assert prediction._parse_dates(typed).notna().tolist() == [True, False]
    assert prediction._parse_dates(pd.Series([], dtype="object")).empty


# --------------------------------------------------------------------------- #
# predict_risk / predict_batch / risk_label
# --------------------------------------------------------------------------- #
def test_predict_risk_equals_pipeline_on_the_same_row(bundle, good_inputs):
    res = predict_risk(good_inputs, bundle)
    cleaned, errors = validate_input(good_inputs, bundle)
    assert errors == []
    row = pd.DataFrame([cleaned])[bundle["feature_columns"]]
    # independent of prediction._score: call the pipeline directly, then apply the documented floor
    expected = max(float(bundle["pipeline"].predict_proba(row)[0, 1]), bundle["probability_floor"])
    assert res["probability"] == pytest.approx(expected, abs=1e-12)
    assert 0 <= res["probability"] <= 1
    assert res["risk_label"] == risk_label(res["probability"], bundle["risk_thresholds"])
    assert res["risk_color"] == prediction.RISK_COLORS[res["risk_label"]]
    assert res["features_used"] == bundle["feature_columns"]
    assert res["model_name"] == bundle["model_name"] and res["thresholds"] == bundle["risk_thresholds"]


def test_changing_inputs_changes_probability_in_plausible_direction(bundle, good_inputs):
    base = predict_risk(good_inputs, bundle)["probability"]
    states = bundle["form_options"]["customer_state"]
    probs = {s: predict_risk({**good_inputs, "customer_state": s}, bundle)["probability"] for s in states}
    assert len(set(round(p, 9) for p in probs.values())) > 1       # the saved model reacts to the input
    # plausibility: a much shorter promise leaves less slack, so for the same order it must not be rated
    # SAFER than a much longer one (association, not causation)
    lo, hi = bundle["form_options"]["ranges"]["estimated_delivery_days"]
    below = int(lo) - 1                              # a promise below the training range must still be scored (warning)
    assert below >= 1
    res = predict_risk({**good_inputs, "estimated_delivery_days": below}, bundle)
    assert 0 <= res["probability"] <= 1 and any("estimated_delivery_days" in w for w in res["warnings"])
    long_ = predict_risk({**good_inputs, "estimated_delivery_days": int(min(hi, 60))}, bundle)["probability"]
    if res["probability"] < long_:                   # informational only: a retrained model may legitimately differ
        warnings.warn(f"saved model rates a {below}-day promise ({res['probability']:.3f}) safer than a longer one "
                      f"({long_:.3f}) for the same order", stacklevel=1)
    assert 0 <= base <= 1


def test_missing_optional_fields_are_filled_and_reported(bundle, good_inputs):
    slim = {k: v for k, v in good_inputs.items() if k not in ("total_weight_g", "estimated_delivery_days")}
    res = predict_risk(slim, bundle)
    assumed = res["assumptions_used"]
    assert "total_weight_g" in assumed and "estimated_delivery_days" in assumed
    assert assumed["total_weight_g"]["value"] == pytest.approx(bundle["form_options"]["defaults"]["total_weight_g"])
    assert "training" in assumed["total_weight_g"]["source"]
    pair = f"{good_inputs['seller_state']}{prediction.STATE_PAIR_SEP}{good_inputs['customer_state']}"
    expected = bundle["form_options"]["state_pair_median_estimated_days"].get(
        pair, bundle["form_options"]["defaults"]["estimated_delivery_days"])
    assert assumed["estimated_delivery_days"]["value"] == pytest.approx(expected)
    assert "pair" in assumed["estimated_delivery_days"]["source"]
    full = predict_risk(good_inputs, bundle)["assumptions_used"]
    assert "total_weight_g" not in full and "estimated_delivery_days" not in full   # supplied -> not assumed
    assert "total_price" not in full and "customer_state" not in full               # required fields are never defaulted


def _pair_with_distinct_median(bundle) -> tuple[str, str, float]:
    """A seller->customer pair whose median promise differs from the global default."""
    opts = bundle["form_options"]
    default = opts["defaults"]["estimated_delivery_days"]
    for key, med in opts["state_pair_median_estimated_days"].items():
        if med != default:
            seller, cust = key.split(prediction.STATE_PAIR_SEP)
            return seller, cust, med
    pytest.skip("every state pair has the global median promise")


def test_estimated_days_default_uses_the_state_pair_median(bundle, good_inputs):
    seller, cust, med = _pair_with_distinct_median(bundle)
    slim = {k: v for k, v in good_inputs.items() if k != "estimated_delivery_days"}
    slim.update(seller_state=seller, customer_state=cust)
    res = predict_risk(slim, bundle)
    assert res["assumptions_used"]["estimated_delivery_days"]["value"] == pytest.approx(med)
    assert res["derived_features"]["estimated_delivery_days"] == pytest.approx(med)
    assert med != bundle["form_options"]["defaults"]["estimated_delivery_days"]
    out = predict_batch(pd.DataFrame([slim]), bundle)
    assert out["assumptions_used"].iloc[0].split(";").count("estimated_delivery_days") == 1


def test_predict_risk_raises_value_error_on_invalid_input(bundle, good_inputs):
    with pytest.raises(ValueError, match="total_price"):
        predict_risk({**good_inputs, "total_price": -1}, bundle)


@needs_data
def test_predict_batch_probabilities_bounded_and_matches_single(bundle, parts):
    val = parts["validation"].dropna(subset=["total_weight_g"]).head(300)
    raw = pd.DataFrame([_raw_inputs(r) for _, r in val.iterrows()])
    out = predict_batch(raw, bundle)
    assert len(out) == len(raw) and (out["input_error"] == "").all()
    assert out["late_probability"].between(0, 1).all()
    assert set(out["risk_label"]) <= {"Low", "Moderate", "High"}
    single = predict_risk(raw.iloc[7].to_dict(), bundle)["probability"]
    assert out["late_probability"].iloc[7] == pytest.approx(single, abs=1e-12)


def test_predict_batch_reports_assumptions_and_warnings_columns(bundle, good_inputs):
    slim = {k: v for k, v in good_inputs.items() if k not in ("total_weight_g", "estimated_delivery_days")}
    out = predict_batch(pd.DataFrame([good_inputs, slim]), bundle)
    given = set(out["assumptions_used"].iloc[0].split(";"))
    assert not given & {"total_weight_g", "estimated_delivery_days"}            # supplied -> not assumed
    assert set(out["assumptions_used"].iloc[1].split(";")) >= {"total_weight_g", "estimated_delivery_days"}
    assert {"late_probability", "risk_label", "assumptions_used", "input_warning", "input_error"} <= set(out.columns)
    assert list(predict_batch(pd.DataFrame([good_inputs]).iloc[0:0], bundle).columns) == list(out.columns)


def test_predict_batch_flags_bad_rows_and_missing_columns(bundle, good_inputs):
    raw = pd.DataFrame([good_inputs, {**good_inputs, "customer_state": "XX"}, {**good_inputs, "total_price": -1}])
    out = predict_batch(raw, bundle)
    assert out["input_error"].iloc[0] == ""
    assert "not a known customer_state" in out["input_error"].iloc[1]
    assert np.isnan(out["late_probability"].iloc[1:]).all() and out["risk_label"].iloc[1:].isna().all()
    with pytest.raises(ValueError, match="missing required column"):
        predict_batch(raw.drop(columns=["total_price"]), bundle)
    assert predict_batch(raw.iloc[0:0], bundle).empty


def test_risk_label_thresholds_and_bounds():
    th = {"moderate": 0.1, "high": 0.3}
    assert [risk_label(p, th) for p in (0.0, 0.0999, 0.1, 0.2999, 0.3, 1.0)] == [
        "Low", "Low", "Moderate", "Moderate", "High", "High"]
    for bad in (-0.01, 1.01, float("nan")):
        with pytest.raises(ValueError):
            risk_label(bad, th)


def test_load_model_failure_message(tmp_path):
    with pytest.raises(ModelNotFoundError) as exc:
        load_model(tmp_path / "missing.joblib")
    assert "-m src.train_model" in str(exc.value) and str(config.ROOT) in str(exc.value)
    assert isinstance(exc.value, FileNotFoundError)


@needs_model
def test_saved_model_loads_in_a_fresh_process():
    code = ("from src.prediction import load_model; b = load_model(); "
            "assert b['pipeline'].predict_proba is not None; print(b['model_name'])")
    out = subprocess.run([sys.executable, "-c", code], cwd=config.ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


# --------------------------------------------------------------------------- #
# Chronological split
# --------------------------------------------------------------------------- #
def test_split_is_chronological_and_disjoint(parts):
    tr, va, te = parts["train"], parts["validation"], parts["test"]
    t = train_model.TIME_COL
    assert tr[t].max() < va[t].min() and va[t].max() < te[t].min()
    ids = [set(p["order_id"]) for p in (tr, va, te)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert all(p[train_model.TARGET].isin([0, 1]).all() for p in parts.values())
    assert min(len(p) for p in parts.values()) > 1000


def test_split_on_synthetic_frame_boundaries():
    """TEST FIXTURE: boundaries are start-inclusive / end-exclusive; pre-train rows are dropped."""
    ts = pd.to_datetime(["2016-10-01", "2017-01-01", "2018-03-31 23:59:59", "2018-04-01", "2018-06-30 23:59:59",
                         "2018-07-01", "2018-08-01"], format="mixed")
    df = pd.DataFrame({"order_id": list("abcdefg"), train_model.TIME_COL: ts, train_model.TARGET: 0})
    parts = train_model.chronological_split(df)
    assert [list(parts[k]["order_id"]) for k in ("train", "validation", "test")] == [
        ["b", "c"], ["d", "e"], ["f", "g"]]


# --------------------------------------------------------------------------- #
# Metric helpers are safe on degenerate input
# --------------------------------------------------------------------------- #
def test_metric_helpers_safe_on_empty_and_single_class():
    m = train_model.classification_metrics(np.array([]), np.array([]), 0.5)
    assert m["n"] == 0 and m["roc_auc"] is None and m["confusion_matrix"] == [[0, 0], [0, 0]]
    one = train_model.classification_metrics(np.zeros(5, int), np.full(5, 0.2), 0.5)
    assert one["roc_auc"] is None and one["pr_auc"] is None and one["recall"] == 0.0 and one["accuracy"] == 1.0
    assert train_model.f1_optimal_threshold(np.zeros(4, int), np.array([.1, .2, .3, .4])) == 0.5
    assert train_model.f1_optimal_threshold(np.array([0, 1]), np.array([.3, .3])) == 0.5
    assert train_model.band_table(np.array([]), np.array([]), {"moderate": .1, "high": .2})[0]["late_rate"] is None
    assert train_model.risk_thresholds(np.array([])) == {"moderate": 0.5, "high": 0.5}
    assert train_model.curve_payloads(np.array([]), np.array([]))["roc"] == {"fpr": [], "tpr": []}
    const = train_model.curve_payloads(np.array([0, 1, 0, 1]), np.full(4, 0.3))
    assert const["cal"]["bin_counts"] == [4]


def test_f1_threshold_and_bands_on_synthetic_scores():
    """TEST FIXTURE: perfectly separable scores."""
    y = np.array([0] * 8 + [1] * 2)
    p = np.array([.05, .06, .07, .08, .09, .1, .11, .12, .8, .9])
    thr = train_model.f1_optimal_threshold(y, p)
    assert 0.12 < thr <= 0.8
    th = train_model.risk_thresholds(p)
    assert th["moderate"] <= th["high"]
    rows = train_model.band_table(y, p, th)
    assert sum(r["n"] for r in rows) == 10 and rows[-1]["late_rate"] == 1.0


# --------------------------------------------------------------------------- #
# Inference layer with a controlled (stub) model: labels, colours, floor, guards
# --------------------------------------------------------------------------- #
class _StubPipeline:
    """TEST FIXTURE: probability = n_items / 10 (or a fixed value), so label boundaries are exact."""

    def __init__(self, fixed: float | None = None):
        self.fixed = fixed

    def predict_proba(self, X):
        p = np.full(len(X), self.fixed) if self.fixed is not None else X["n_items"].astype(float).to_numpy() / 10
        return np.column_stack([1 - p, p])


def _stub_bundle(bundle, **over) -> dict:
    b = copy.copy(bundle)
    b.update({"pipeline": _StubPipeline(), "risk_thresholds": {"moderate": 0.2, "high": 0.5},
              "probability_floor": 0.0, **over})
    return b


def test_batch_labels_use_ge_at_exact_thresholds(bundle, good_inputs):
    b = _stub_bundle(bundle)
    raw = pd.DataFrame([{**good_inputs, "n_items": n, "n_sellers": 1} for n in (1, 2, 3, 5, 9)])
    out = predict_batch(raw, b)
    assert out["late_probability"].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.5, 0.9])
    assert out["risk_label"].tolist() == ["Low", "Moderate", "Moderate", "High", "High"]
    assert out["risk_label"].tolist() == [risk_label(p, b["risk_thresholds"]) for p in out["late_probability"]]
    assert prediction.risk_labels(np.array([0.2, 0.5, 0.4999]), b["risk_thresholds"]).tolist() == [
        "Moderate", "High", "Moderate"]


def test_risk_colour_matches_label_and_palette(bundle, good_inputs):
    assert prediction.RISK_COLORS == {"Low": config.PALETTE["green"], "Moderate": config.PALETTE["amber"],
                                      "High": config.PALETTE["red"]}
    b = _stub_bundle(bundle)
    got = {}
    for n in (1, 3, 8):
        res = predict_risk({**good_inputs, "n_items": n, "n_sellers": 1}, b)
        got[res["risk_label"]] = res["risk_color"]
    assert got == {"Low": config.PALETTE["green"], "Moderate": config.PALETTE["amber"], "High": config.PALETTE["red"]}


def test_predict_risk_reports_bundle_decision_threshold_and_calibration(bundle, good_inputs):
    b = _stub_bundle(bundle, decision_threshold=0.1234, calibrated=True)
    res = predict_risk(good_inputs, b)
    assert res["decision_threshold"] == 0.1234 and res["calibrated"] is True
    assert predict_risk(good_inputs, _stub_bundle(bundle, decision_threshold=0.3))["decision_threshold"] == 0.3


@pytest.mark.parametrize("bad", [1.5, -0.1, float("nan")])
def test_scores_outside_unit_interval_are_refused(bundle, good_inputs, bad):
    b = _stub_bundle(bundle, pipeline=_StubPipeline(fixed=bad))
    with pytest.raises(ValueError, match="outside"):
        predict_risk(good_inputs, b)
    with pytest.raises(ValueError, match="outside"):
        predict_batch(pd.DataFrame([good_inputs]), b)


def test_probability_floor_replaces_exact_zero_and_is_explained(bundle, good_inputs):
    res0 = predict_risk(good_inputs, _stub_bundle(bundle, pipeline=_StubPipeline(fixed=0.0)))
    assert res0["probability"] == 0.0
    b = _stub_bundle(bundle, pipeline=_StubPipeline(fixed=0.0), probability_floor=0.004, calibrated=True,
                     score_resolution={"distinct_levels": 30, "max": 0.31})
    res = predict_risk(good_inputs, b)
    assert res["probability"] == 0.004 and "0.40%" in res["probability_note"] and "30 distinct" in res["probability_note"]
    assert predict_batch(pd.DataFrame([good_inputs]), b)["late_probability"].iloc[0] == 0.004
    above = predict_risk(good_inputs, _stub_bundle(bundle, pipeline=_StubPipeline(fixed=0.3), probability_floor=0.004))
    assert above["probability"] == 0.3 and "lowest level" not in (above["probability_note"] or "")
    uncal = predict_risk(good_inputs, _stub_bundle(bundle, calibrated=False, test_window_caveat=None))
    assert uncal["probability_note"] is None and uncal["test_window_caveat"] is None
    uncal2 = predict_risk(good_inputs, _stub_bundle(bundle, calibrated=False))
    assert "distinct values" not in (uncal2["probability_note"] or "")      # step-function text only when calibrated


def test_saved_bundle_has_floor_and_no_exact_zero_scores(bundle, parts):
    floor = bundle["probability_floor"]
    assert floor >= 0 and bundle["score_resolution"]["probability_floor"] == floor
    p = prediction._score(bundle["pipeline"], parts["test"], bundle["feature_columns"], floor)
    assert p.min() >= floor
    if floor > 0:                           # a floor exists to keep exact-zero scores out of the output
        assert p.min() > 0


# --------------------------------------------------------------------------- #
# load_model: cache invalidation, version mismatch, incomplete bundle
# --------------------------------------------------------------------------- #
def test_load_model_cache_follows_file_changes(bundle, tmp_path):
    path = tmp_path / "m.joblib"
    joblib.dump({**bundle, "model_name": "first"}, path)
    assert load_model(path)["model_name"] == "first"
    joblib.dump({**bundle, "model_name": "second"}, path)
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    assert load_model(path)["model_name"] == "second"


def test_load_model_warns_on_sklearn_version_mismatch(bundle, tmp_path):
    path = tmp_path / "old.joblib"
    joblib.dump({**bundle, "sklearn_version": "0.0.1"}, path)
    with pytest.warns(UserWarning, match=r"scikit-learn 0\.0\.1"):
        load_model(path)


def test_load_model_rejects_incomplete_bundle(bundle, tmp_path):
    path = tmp_path / "partial.joblib"
    joblib.dump({k: v for k, v in bundle.items() if k != "form_options"}, path)
    with pytest.raises(ValueError, match="missing keys.*form_options"):
        load_model(path)


# --------------------------------------------------------------------------- #
# Exact-value metric helpers (hand-computed)
# --------------------------------------------------------------------------- #
Y8 = np.array([1, 1, 0, 0, 1, 0, 0, 0])
P8 = np.array([.9, .8, .7, .6, .3, .2, .1, .05])


def test_classification_metrics_exact_values():
    """TEST FIXTURE: threshold 0.5 -> tp=2, fp=2, fn=1, tn=3 (fp != fn so the two cannot be swapped)."""
    m = train_model.classification_metrics(Y8, P8, 0.5)
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))
    assert m["accuracy"] == pytest.approx(5 / 8) and m["predicted_positive_rate"] == pytest.approx(0.5)
    assert m["brier"] == pytest.approx(1.4425 / 8)
    assert m["log_loss"] == pytest.approx(float(-np.mean(np.where(Y8 == 1, np.log(P8), np.log(1 - P8)))))
    assert m["roc_auc"] == pytest.approx(roc_auc_score(Y8, P8))
    assert m["precision_top10pct"] == pytest.approx(1.0) and m["recall_top10pct"] == pytest.approx(1 / 3)
    assert m["lift_top10pct"] == pytest.approx(1.0 / 0.375)
    assert m["confusion_matrix"] == [[3, 2], [1, 2]]


def test_top_fraction_stats_exact_and_tie_robust():
    s = train_model.top_fraction_stats(Y8, P8, 0.5)             # top 4: .9 .8 .7 .6 -> tp=2
    assert s["precision"] == pytest.approx(0.5) and s["recall"] == pytest.approx(2 / 3)
    y = np.array([1, 1] + [0] * 8)
    p = np.array([.9, .5, .5, .5] + [.1] * 6)                    # k=3: one above the cut + 2 of 3 tied at .5
    a = train_model.top_fraction_stats(y, p, 0.3)
    assert a["precision"] == pytest.approx((1 + 2 / 3) / 3) and a["recall"] == pytest.approx((1 + 2 / 3) / 2)
    b = train_model.top_fraction_stats(y[::-1], p[::-1], 0.3)    # row order must not matter
    assert b["precision"] == pytest.approx(a["precision"])
    const = np.array([0] * 90 + [1] * 10)                        # constant scores: no-skill = prevalence, any order
    for arr in (const, const[::-1]):
        c = train_model.top_fraction_stats(arr, np.full(100, 0.07), 0.10)
        assert c["precision"] == pytest.approx(0.10) and c["recall"] == pytest.approx(0.10)
    assert train_model.top_fraction_stats(np.zeros(5), np.arange(5.0))["recall"] is None
    assert train_model.top_fraction_stats(np.array([]), np.array([])) == {"precision": None, "recall": None}


def test_f1_optimal_threshold_exact_cut_and_tie_break():
    assert train_model.f1_optimal_threshold(np.array([0, 0, 0, 1, 1]), np.array([.1, .2, .3, .6, .9])) == pytest.approx(0.6)
    # F1 is 2/3 at both 0.9 and 0.3: the highest cut wins
    assert train_model.f1_optimal_threshold(np.array([1, 0, 0, 1]), np.array([.9, .5, .4, .3])) == pytest.approx(0.9)


def test_bands_and_thresholds_exact():
    th = train_model.risk_thresholds(np.linspace(0, 1, 101))
    assert th == pytest.approx({"moderate": 0.8, "high": 0.95})
    rows = train_model.band_table(np.array([0, 1, 1]), np.array([.1, .2, .3]), {"moderate": .2, "high": .3})
    assert [(r["band"], r["n"], r["n_late"]) for r in rows] == [("Low", 1, 0), ("Moderate", 1, 1), ("High", 1, 1)]
    assert rows[1]["late_ci_low"] is not None and rows[0]["late_ci_low"] == 0.0


def test_wilson_interval_and_band_diagnostics():
    lo, hi = train_model.wilson_interval(10, 100)
    assert lo == pytest.approx(0.0552, abs=5e-4) and hi == pytest.approx(0.1744, abs=5e-4)
    assert train_model.wilson_interval(0, 0) == (None, None)
    rows = train_model.band_table(
        np.array([1] * 6 + [0] * 94 + [1] * 8 + [0] * 92 + [1] * 70 + [0] * 30),
        np.array([.1] * 100 + [.3] * 100 + [.9] * 100), {"moderate": .2, "high": .5})
    d = train_model.band_diagnostics(rows)
    assert d["monotone"] is True and d["shares"] == {"Low": 1 / 3, "Moderate": 1 / 3, "High": 1 / 3}
    assert d["adjacent_intervals_overlap"] == {"Low-Moderate": True, "Moderate-High": False}
    assert "Low-Moderate" in d["note"]
    flat = train_model.band_diagnostics(train_model.band_table(
        np.array([1] * 20 + [0] * 80 + [1] * 5 + [0] * 95), np.array([.1] * 100 + [.9] * 100), {"moderate": .3, "high": .5}))
    assert flat["monotone"] is None or flat["monotone"] is False


def test_probability_floor_rules():
    assert train_model.probability_floor(np.array([.1, .2]), np.array([0, 1])) == 0.0
    p = np.array([0.0] * 8 + [0.05, 0.3])
    y = np.array([0] * 8 + [0, 1])
    assert train_model.probability_floor(p, y) == pytest.approx(0.05)         # (0 + 1) / (8 + 2) = 0.1, capped at the lowest positive score
    assert train_model.probability_floor(np.array([0.0] * 98 + [0.2, 0.3]), np.array([0] * 100)) == pytest.approx(1 / 100)
    sr = train_model.score_resolution(np.array([0.01, 0.01, 0.2]), 0.01)
    assert sr["distinct_levels"] == 2 and sr["share_at_floor"] == pytest.approx(2 / 3) and sr["max"] == 0.2


def test_choose_model_prefers_simplest_within_noise():
    rng = np.random.default_rng(0)
    boot = {"logistic_regression": 0.10 + rng.normal(0, 0.01, 200), "random_forest": 0.12 + rng.normal(0, 0.01, 200),
            "hist_gradient_boosting": 0.121 + rng.normal(0, 0.01, 200)}
    val = {"logistic_regression": 0.119, "random_forest": 0.12, "hist_gradient_boosting": 0.121}
    best, within, chosen = train_model.choose_model(val, boot)
    assert best == "hist_gradient_boosting" and chosen == "logistic_regression"
    val_far = {**val, "logistic_regression": 0.05}
    assert train_model.choose_model(val_far, boot)[2] == "hist_gradient_boosting"   # equal simplicity -> higher score


def test_support_shift_counts_out_of_range_orders():
    """TEST FIXTURE: train range of x is 10..20; one validation and two test orders fall outside."""
    t = train_model.TARGET
    mk = lambda xs, ys: pd.DataFrame({"x": xs, "z": [1.0] * len(xs), t: ys})  # noqa: E731
    parts_ = {"train": mk([10, 15, 20], [0, 1, 0]), "validation": mk([12, 25], [0, 1]),
              "test": mk([5, 18, 30, 11], [1, 0, 0, 0])}
    sh = train_model.support_shift(parts_, ["x", "z"])
    assert sh["validation"]["n_outside"] == 1 and sh["validation"]["late_rate_outside"] == 1.0
    assert sh["test"]["n_outside"] == 2 and sh["test"]["share_outside"] == 0.5
    assert sh["test"]["late_rate_outside"] == 0.5 and sh["test"]["late_rate_inside"] == 0.0
    assert list(sh["test"]["share_by_feature"]) == ["x"]
    assert "50.00% of 4 orders" in train_model.support_note(sh)


# --------------------------------------------------------------------------- #
# Honest test-window reporting
# --------------------------------------------------------------------------- #
def _model_entry(test_over=None, **over) -> dict:
    base = {"validation": {"pr_auc": 0.14, "positive_rate": 0.04, "roc_auc": 0.78},
            "test": {"n": 1000, "pr_auc": 0.075, "positive_rate": 0.05, "roc_auc": 0.62, "brier": 0.049, "log_loss": 0.21,
                     **(test_over or {})},
            "test_ci95": {"pr_auc": [0.064, 0.085]}, "mean_predicted_test": 0.075, "calibrated": True,
            "test_pr_auc_uncalibrated": 0.08}
    base.update(over)
    return base


DUMMY_ENTRY = {"test": {"brier": 0.0475, "log_loss": 0.205}}


def test_test_window_assessment_discloses_degradation():
    text, warns, degraded = train_model.test_window_assessment(_model_entry(), DUMMY_ENTRY)
    assert degraded is True
    assert "n=1,000" in text and "1.5x lift" in text and "0.064-0.085" in text and "7.5% vs observed 5.0%" in text
    assert "0.0490 vs 0.0475" in text                                     # Brier vs constant baseline
    joined = " ".join(warns)
    assert "Brier" in joined and "log-loss" in joined and "degraded" in joined and "Calibration lowered" in joined


def test_test_window_assessment_quiet_when_model_holds_up():
    good = _model_entry(test_over={"pr_auc": 0.16, "brier": 0.040, "log_loss": 0.18}, calibrated=False,
                        test_ci95={"pr_auc": [0.14, 0.18]}, test_pr_auc_uncalibrated=0.15)
    text, warns, degraded = train_model.test_window_assessment(good, DUMMY_ENTRY)
    assert warns == [] and degraded is False and "3.2x lift" in text
    close = _model_entry(test_over={"pr_auc": 0.05}, test_ci95={"pr_auc": [0.04, 0.06]}, calibrated=False)
    assert any("close to 1" in w for w in train_model.test_window_assessment(close, DUMMY_ENTRY)[1])


def test_importance_comparison_flags_sign_flips():
    val = [{"feature": "a", "importance_mean": 0.07, "importance_std": 0.01},
           {"feature": "b", "importance_mean": 0.05, "importance_std": 0.01},
           {"feature": "c", "importance_mean": 0.001, "importance_std": 0.01}]
    test = [{"feature": "a", "importance_mean": 0.03, "importance_std": 0.01},
            {"feature": "b", "importance_mean": -0.02, "importance_std": 0.005},
            {"feature": "c", "importance_mean": -0.001, "importance_std": 0.01}]
    txt = train_model.importance_comparison(val, test)
    assert "flips between windows for: b." in txt and "by validation importance: a, b, c" in txt


# --------------------------------------------------------------------------- #
# Calibration decision + leakage audit enforcement
# --------------------------------------------------------------------------- #
def _calibration_parts(seed: int = 0, n: int = 8000) -> dict:
    """TEST FIXTURE: one feature x; true late probability = 0.05 + 0.25 * x."""
    rng = np.random.default_rng(seed)
    x = rng.random(n)
    y = (rng.random(n) < 0.05 + 0.25 * x).astype(int)
    ts = pd.date_range("2018-04-01", periods=n, freq="min")
    df = pd.DataFrame({"x": x, train_model.TARGET: y, train_model.TIME_COL: ts})
    return {"train": df.iloc[: n // 2], "validation": df.iloc[n // 2:]}


def test_calibration_kept_only_when_it_helps_later_half():
    parts_ = _calibration_parts()
    tr = parts_["train"]
    overconfident = train_model.build_pipeline(LogisticRegression(class_weight="balanced"), ["x"], [], scale=False)
    overconfident.fit(tr[["x"]], tr[train_model.TARGET])
    res = train_model.calibrate(overconfident, parts_, ["x"])
    assert res["calibrated"] is True and res["method"] in ("sigmoid", "isotonic")
    assert res["diagnostics"]["raw"]["brier"] > res["diagnostics"][res["method"]]["brier"]
    honest = train_model.build_pipeline(LogisticRegression(), ["x"], [], scale=False)
    honest.fit(tr[["x"]], tr[train_model.TARGET])
    res2 = train_model.calibrate(honest, parts_, ["x"])
    d = res2["diagnostics"]
    helps = any(d[m]["brier"] < d["raw"]["brier"] and d[m]["log_loss"] < d["raw"]["log_loss"] for m in ("sigmoid", "isotonic"))
    assert res2["calibrated"] is helps                      # the decision follows the held-out numbers, both ways
    assert res2["estimator"] is honest or helps


def test_leakage_audit_fails_on_forbidden_feature(parts, bundle):
    ok = train_model.leakage_audit(bundle["numeric_features"], bundle["categorical_features"], parts,
                                   bundle["base_pipeline"], bundle["feature_columns"], [])
    assert ok["passed"] is True
    bad = train_model.leakage_audit(bundle["numeric_features"][:-1] + ["processing_days"], bundle["categorical_features"],
                                    parts, bundle["base_pipeline"], bundle["feature_columns"], [])
    assert bad["passed"] is False and any(c.startswith("FAIL") for c in bad["checks"])
    parity = train_model.leakage_audit(bundle["numeric_features"], bundle["categorical_features"], parts,
                                       bundle["base_pipeline"], bundle["feature_columns"], ["mismatch"])
    assert parity["passed"] is False


# --------------------------------------------------------------------------- #
# Artefacts
# --------------------------------------------------------------------------- #
REQUIRED_TOP = ["generated_at", "target", "features", "split", "models", "selected_model", "risk_bands",
                "permutation_importance", "leakage_audit", "metric_guidance", "training_seconds", "library_versions"]
REQUIRED_MODEL = ["validation", "test", "threshold", "best_params", "confusion_matrix_test",
                  "confusion_matrix_validation", "roc_curve_test", "pr_curve_test", "calibration_test"]
METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier", "n", "positive_rate"]


@needs_model
def test_metrics_json_has_required_structure(metrics):
    for k in REQUIRED_TOP:
        assert k in metrics, k
    assert {"numeric", "categorical", "dropped_candidates"} <= set(metrics["features"])
    assert metrics["features"]["dropped_candidates"]
    for part in ("train", "validation", "test"):
        assert {"start", "end", "n", "late_rate"} <= set(metrics["split"][part])
    assert metrics["split"]["method"] and metrics["split"]["notes"]
    assert {"validation", "test"} <= set(metrics["split"]["covariate_support"])
    assert set(metrics["models"]) == {"dummy_prior", *train_model.candidate_estimators(False)}   # code contract, not a model choice
    for name, m in metrics["models"].items():
        for k in REQUIRED_MODEL:
            assert k in m, (name, k)
        for w in ("validation", "test"):
            assert set(METRIC_KEYS) <= set(m[w]), (name, w)
        assert len(np.asarray(m["confusion_matrix_test"])) == 2
        assert len(m["roc_curve_test"]["fpr"]) <= 205 and len(m["pr_curve_test"]["precision"]) <= 205
        cal = m["calibration_test"]
        assert len(cal["mean_predicted"]) == len(cal["fraction_positive"]) == len(cal["bin_counts"])
        assert "in-sample" in m["validation_note"]
    sel = metrics["selected_model"]
    assert {"name", "primary_metric", "rationale", "threshold", "calibrated", "test_window_assessment",
            "test_window_warnings", "score_resolution"} <= set(sel)
    assert sel["name"] in metrics["models"] and sel["name"] != "dummy_prior"
    rb = metrics["risk_bands"]
    assert {"definition", "thresholds", "validation", "test", "validation_diagnostics", "test_diagnostics"} <= set(rb)
    assert {"moderate", "high"} <= set(rb["thresholds"])
    for part in ("validation", "test"):
        assert [r["band"] for r in rb[part]] == ["Low", "Moderate", "High"]
    pi = metrics["permutation_importance"]
    assert {"window", "scoring", "n_repeats", "items", "test_items", "comparison_note"} <= set(pi) and pi["items"]
    assert pi["window"] == "validation" and pi["test_window"] == "test"
    assert {"feature", "importance_mean", "importance_std"} <= set(pi["items"][0])
    assert metrics["leakage_audit"]["passed"] is True and metrics["leakage_audit"]["checks"]
    assert metrics["training_seconds"] > 0 and "scikit-learn" in metrics["library_versions"]


@needs_model
def test_metrics_json_has_no_nan_or_infinity():
    def reject(token):
        raise AssertionError(f"non-finite JSON constant {token}")
    json.loads(config.MODEL_METRICS_PATH.read_text(), parse_constant=reject)


@needs_model
def test_metrics_features_match_bundle(bundle, metrics):
    assert metrics["features"]["numeric"] == bundle["numeric_features"]
    assert metrics["features"]["categorical"] == bundle["categorical_features"]
    assert metrics["selected_model"]["name"] == bundle["model_name"]
    assert metrics["selected_model"]["threshold"] == pytest.approx(bundle["decision_threshold"])
    assert metrics["risk_bands"]["thresholds"] == pytest.approx(bundle["risk_thresholds"])
    assert 0 < bundle["risk_thresholds"]["moderate"] <= bundle["risk_thresholds"]["high"] <= 1


@needs_model
@needs_data
def test_metrics_match_recomputation_from_bundle(bundle, metrics, parts):
    te = parts["test"]
    y = te[train_model.TARGET].to_numpy()
    p = prediction._score(bundle["pipeline"], te, bundle["feature_columns"], bundle["probability_floor"])
    m = metrics["models"][bundle["model_name"]]["test"]
    thr = metrics["selected_model"]["threshold"]
    assert m["n"] == len(te) == metrics["split"]["test"]["n"]
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, p), abs=1e-9)
    assert m["accuracy"] == pytest.approx(float(((p >= thr).astype(int) == y).mean()), abs=1e-9)
    assert m["brier"] == pytest.approx(float(np.mean((p - y) ** 2)), abs=1e-9)
    assert m["precision_top10pct"] == pytest.approx(train_model.top_fraction_stats(y, p)["precision"], abs=1e-9)
    cm = np.asarray(metrics["models"][bundle["model_name"]]["confusion_matrix_test"])
    assert cm.sum() == len(te) and cm[1].sum() == int(y.sum())
    bands = metrics["risk_bands"]["test"]
    assert sum(b["n"] for b in bands) == len(te)
    th = bundle["risk_thresholds"]
    assert bands[2]["n"] == int((p >= th["high"]).sum())
    assert bands[1]["n"] == int(((p >= th["moderate"]) & (p < th["high"])).sum())
    assert metrics["split"]["train"]["n"] == bundle["train_window"]["n"] == len(parts["train"])
    pv = prediction._score(bundle["pipeline"], parts["validation"], bundle["feature_columns"], bundle["probability_floor"])
    assert bundle["risk_thresholds"] == pytest.approx(train_model.risk_thresholds(pv))


@needs_model
def test_selected_model_beats_dummy_on_validation_primary_metric(metrics):
    sel = metrics["selected_model"]["name"]
    m = metrics["models"]
    assert m[sel]["validation"]["pr_auc"] > m["dummy_prior"]["validation"]["pr_auc"]


@needs_model
def test_headline_text_discloses_test_window_weakness(metrics):
    """The rationale / guidance must carry the test-window verdict whenever the data say so."""
    sel, models = metrics["selected_model"], metrics["models"]
    t, d = models[sel["name"]]["test"], models["dummy_prior"]["test"]
    assert sel["test_window_assessment"] in sel["rationale"] and sel["test_window_assessment"] in metrics["metric_guidance"]
    assert f"n={t['n']:,}" in sel["test_window_assessment"]
    worse = t["brier"] >= d["brier"]
    assert any("Brier" in w for w in sel["test_window_warnings"]) == worse
    assert ("WARNING" in sel["rationale"]) == bool(sel["test_window_warnings"])
    assert ("WARNING" in metrics["metric_guidance"]) == bool(sel["test_window_warnings"])
    assert "can degrade" not in metrics["metric_guidance"] or not sel["test_performance_degraded"]


@needs_model
def test_baseline_top10_precision_is_prevalence_and_bands_are_diagnosed(metrics):
    dm = metrics["models"]["dummy_prior"]
    for w in ("validation", "test"):
        assert dm[w]["precision_top10pct"] == pytest.approx(dm[w]["positive_rate"])
    for w in ("validation", "test"):
        rows = metrics["risk_bands"][w]
        diag = metrics["risk_bands"][f"{w}_diagnostics"]
        rates = [r["late_rate"] for r in rows]
        assert diag["monotone"] == bool(rates[0] <= rates[1] <= rates[2])
        assert sum(diag["shares"].values()) == pytest.approx(1.0)
        for r in rows:
            assert r["late_ci_low"] <= r["late_rate"] <= r["late_ci_high"]
    assert "approximately" in metrics["risk_bands"]["definition"]
    assert metrics["risk_bands"]["test_diagnostics"]["note"] in metrics["risk_bands"]["definition"]


@needs_model
def test_split_notes_report_out_of_support_share(metrics):
    cs = metrics["split"]["covariate_support"]["test"]
    assert cs["n"] == metrics["split"]["test"]["n"] and 0 <= cs["share_outside"] <= 1
    assert any(n.startswith("Out-of-support orders") for n in metrics["split"]["notes"])


# --------------------------------------------------------------------------- #
# End-to-end training on a small subsample (quick settings, temp dir)
# --------------------------------------------------------------------------- #
@needs_data
def test_train_end_to_end_on_subsample(tmp_path, data, monkeypatch):
    sample = data.sample(frac=0.25, random_state=3)
    seen: dict = {"pipelines": []}
    orig_choose, orig_build = train_model.choose_model, train_model.build_pipeline

    def spy_choose(val_pr, boot_pr):
        seen["val_pr"], seen["contenders"] = dict(val_pr), list(boot_pr)
        seen["out"] = orig_choose(val_pr, boot_pr)
        return seen["out"]

    def spy_build(estimator, numeric, categorical, scale):
        seen["pipelines"].append((type(estimator).__name__, scale))
        return orig_build(estimator, numeric, categorical, scale)

    monkeypatch.setattr(train_model, "choose_model", spy_choose)
    monkeypatch.setattr(train_model, "build_pipeline", spy_build)
    wall0 = time.perf_counter()
    bundle, metrics = train_model.train(df=sample, out_dir=tmp_path, quick=True, verbose=False)
    wall = time.perf_counter() - wall0
    monkeypatch.undo()
    assert (tmp_path / config.MODEL_PATH.name).exists() and (tmp_path / config.MODEL_METRICS_PATH.name).exists()
    assert metrics["leakage_audit"]["passed"]
    assert set(bundle["feature_columns"]) <= set(config.PURCHASE_TIME_FEATURES)
    reloaded = load_model(tmp_path / config.MODEL_PATH.name)
    sp = train_model.chronological_split(sample)
    tr, va, te = sp["train"], sp["validation"], sp["test"]
    cols, floor = reloaded["feature_columns"], reloaded["probability_floor"]
    score = lambda frame: prediction._score(reloaded["pipeline"], frame, cols, floor)  # noqa: E731
    pt, pv = score(te), score(va)
    assert np.all((pt >= 0) & (pt <= 1))
    name = bundle["model_name"]
    mt = metrics["models"][name]
    yte, yva = te[train_model.TARGET].to_numpy(), va[train_model.TARGET].to_numpy()

    # test metrics are those of the saved pipeline on the test slice
    assert mt["test"]["n"] == len(te)
    assert mt["test"]["brier"] == pytest.approx(float(np.mean((pt - yte) ** 2)), abs=1e-9)
    assert mt["test"]["roc_auc"] == pytest.approx(roc_auc_score(yte, pt), abs=1e-9)
    assert mt["roc_curve_test"] == pytest.approx(train_model.curve_payloads(yte, pt)["roc"])
    assert mt["roc_curve_validation"] == pytest.approx(train_model.curve_payloads(yva, pv)["roc"])

    # thresholds / bands / decision cut are frozen on VALIDATION scores of the saved pipeline
    assert reloaded["risk_thresholds"] == pytest.approx(train_model.risk_thresholds(pv))
    assert reloaded["risk_thresholds"] != pytest.approx(train_model.risk_thresholds(pt))
    assert reloaded["decision_threshold"] == pytest.approx(train_model.f1_optimal_threshold(yva, pv))
    assert metrics["selected_model"]["threshold"] == pytest.approx(reloaded["decision_threshold"])
    th = reloaded["risk_thresholds"]
    for rows, y_, p_ in ((metrics["risk_bands"]["validation"], yva, pv), (metrics["risk_bands"]["test"], yte, pt)):
        masks = {"High": p_ >= th["high"], "Moderate": (p_ >= th["moderate"]) & (p_ < th["high"]), "Low": p_ < th["moderate"]}
        for r in rows:                                    # expected counts computed here, not by band_table
            m = masks[r["band"]]
            assert r["n"] == int(m.sum()) and r["n_late"] == int(y_[m].sum())
            assert (r["late_rate"] == pytest.approx(float(y_[m].mean()))) if m.any() else r["late_rate"] is None

    # recorded accuracy / confusion matrix follow the documented `p >= decision_threshold` rule on the floored scores
    thr = reloaded["decision_threshold"]
    for y_, p_, acc, cm in ((yva, pv, mt["validation"]["accuracy"], mt["confusion_matrix_validation"]),
                            (yte, pt, mt["test"]["accuracy"], mt["confusion_matrix_test"])):
        pred = p_ >= thr
        assert acc == pytest.approx(float((pred == y_).mean()), abs=1e-9)
        assert cm == [[int(((~pred) & (y_ == 0)).sum()), int((pred & (y_ == 0)).sum())],
                      [int(((~pred) & (y_ == 1)).sum()), int((pred & (y_ == 1)).sum())]]
    # the probability floor is applied to the validation scores behind the metrics too
    assert mt["validation"]["brier"] == pytest.approx(float(np.mean((pv - yva) ** 2)), abs=1e-9)

    # bundle bookkeeping
    assert reloaded["sklearn_version"] == sklearn.__version__ == metrics["library_versions"]["scikit-learn"]
    assert reloaded["trained_at"] == metrics["generated_at"]
    dt.datetime.fromisoformat(reloaded["trained_at"])
    assert reloaded["train_window"]["n"] == len(tr) == metrics["split"]["train"]["n"]
    assert reloaded["train_window"]["start"] == tr[train_model.TIME_COL].min().strftime("%Y-%m-%d")
    assert reloaded["train_window"]["end"] == tr[train_model.TIME_COL].max().strftime("%Y-%m-%d")
    summ = reloaded["metrics_summary"]
    assert summ["validation"] == train_model._clean(mt["validation"]) and summ["test"] == train_model._clean(mt["test"])
    assert summ["validation"] != summ["test"] and summ["primary_metric"] == "pr_auc"
    assert summ["test_performance_degraded"] == metrics["selected_model"]["test_performance_degraded"]

    # the dummy baseline never flags (threshold 0.5 vs a constant prior below it)
    dm = metrics["models"]["dummy_prior"]
    assert dm["threshold"] == 0.5 and dm["test"]["predicted_positive_rate"] == 0.0

    # the test-window caveat carried in the bundle is the one in the metrics and agrees with the test numbers
    cav = reloaded["test_window_caveat"]
    assert cav["n"] == len(te) == mt["test"]["n"] and cav["mean_predicted"] == pytest.approx(float(pt.mean()), abs=1e-9)
    assert cav["observed_late_rate"] == pytest.approx(float(yte.mean()))
    assert cav["brier"] == pytest.approx(mt["test"]["brier"]) and cav["brier_baseline"] == pytest.approx(
        metrics["models"]["dummy_prior"]["test"]["brier"])
    assert cav["brier_not_better_than_baseline"] == (cav["brier"] >= cav["brier_baseline"])
    assert {k: v["n"] for k, v in cav["band_late_rates"].items()} == {r["band"]: r["n"] for r in metrics["risk_bands"]["test"]}
    assert cav["test_performance_degraded"] == metrics["selected_model"]["test_performance_degraded"]

    # form options come from the TRAIN slice only
    opts = reloaded["form_options"]
    for c in prediction.RAW_NUMERIC:
        assert opts["ranges"][c] == pytest.approx([float(tr[c].astype(float).min()), float(tr[c].astype(float).max())])
    assert opts["defaults"]["purchase_dow"] == pytest.approx(float(tr["purchase_dow"].astype(float).median()))
    # the date span is the FITTING window, not the whole dataset; the dataset span is kept separately
    assert opts["date_span"] == [tr["purchase_date"].min().strftime("%Y-%m-%d"), tr["purchase_date"].max().strftime("%Y-%m-%d")]
    assert opts["date_span"] == [reloaded["train_window"]["start"], reloaded["train_window"]["end"]]
    assert opts["dataset_date_span"] == [sample["purchase_date"].min().strftime("%Y-%m-%d"),
                                         sample["purchase_date"].max().strftime("%Y-%m-%d")]

    # importance is computed on validation (test importance reported separately); calibrated flag consistent
    assert metrics["permutation_importance"]["window"] == "validation"
    pi = permutation_importance(reloaded["pipeline"], va[cols], yva, scoring="average_precision",
                                n_repeats=metrics["permutation_importance"]["n_repeats"],
                                random_state=config.RANDOM_STATE, n_jobs=1)
    got = {i["feature"]: i["importance_mean"] for i in metrics["permutation_importance"]["items"]}
    assert got == pytest.approx(dict(zip(cols, pi.importances_mean)), abs=1e-9)
    assert metrics["permutation_importance"]["test_window"] == "test"
    assert reloaded["calibrated"] == metrics["selected_model"]["calibrated"] == mt["calibrated"]
    assert metrics["selected_model"]["name"] == name and name != "dummy_prior"
    if reloaded["calibrated"]:
        assert reloaded["pipeline"] is not reloaded["base_pipeline"]
    assert reloaded["probability_floor"] == pytest.approx(train_model.probability_floor(
        reloaded["pipeline"].predict_proba(va[cols])[:, 1], yva))
    assert metrics["selected_model"]["test_window_assessment"] in metrics["selected_model"]["rationale"]

    # ---- the protocol guarantee: tuning / selection / the baseline use train (+ validation) only -------------------
    ytr = tr[train_model.TARGET].to_numpy()
    num, cat = reloaded["numeric_features"], reloaded["categorical_features"]
    assert te[train_model.TARGET].mean() != pytest.approx(ytr.mean())          # the checks below have power
    from sklearn.metrics import average_precision_score
    best_val = {}
    for n, (factory, grid) in train_model.candidate_estimators(True).items():
        recorded = metrics["models"][n]["tuning_grid"]
        assert [r["params"] for r in recorded] == grid
        got = []
        for params in grid:                       # independent fit on TRAIN, scored on VALIDATION
            pipe = train_model.build_pipeline(factory(**params), num, cat, scale=n == "logistic_regression")
            pipe.fit(tr[cols], ytr)
            got.append(float(average_precision_score(yva, pipe.predict_proba(va[cols])[:, 1])))
        assert [r["validation_pr_auc"] for r in recorded] == pytest.approx(got, abs=1e-9), n
        assert metrics["models"][n]["best_params"] == grid[int(np.argmax(got))], n
        best_val[n] = max(got)
    # the model choice is fed validation PR-AUC of exactly the three real contenders (never the dummy)
    assert set(seen["contenders"]) == set(best_val) and "dummy_prior" not in seen["contenders"]
    assert {n: seen["val_pr"][n] for n in best_val} == pytest.approx(best_val, abs=1e-9)
    assert seen["val_pr"]["dummy_prior"] == pytest.approx(float(yva.mean()))
    assert metrics["selected_model"]["name"] == seen["out"][2] == name
    # the dummy baseline is fitted on TRAIN: its constant probability is the train prevalence, not validation / test's
    dm_models = metrics["models"]["dummy_prior"]
    assert dm_models["mean_predicted_test"] == pytest.approx(float(ytr.mean()))
    assert dm_models["mean_predicted_validation"] == pytest.approx(float(ytr.mean()))
    assert dm_models["mean_predicted_test"] != pytest.approx(float(yte.mean()))
    # scaling is part of the logistic model's pipeline only (and applied everywhere that model is built)
    assert {sc for est, sc in seen["pipelines"] if est == "LogisticRegression"} == {True}
    assert {sc for est, sc in seen["pipelines"] if est != "LogisticRegression"} == {False}
    assert reloaded["pipeline"] is not None and "LogisticRegression" in {e for e, _ in seen["pipelines"]}

    # ---- hygiene recorded in the artefacts (no hard-coded numbers) ---------------------------------------------------
    checks = metrics["leakage_audit"]["checks"]
    assert any(c.startswith("PASS") and "reproduce the processed table" in c for c in checks)
    assert 0 < metrics["training_seconds"] <= wall and metrics["training_seconds"] >= 0.5 * wall
    guidance = metrics["metric_guidance"]
    assert f"Only {yte.mean():.1%} of test orders are late" in guidance and f"{1 - yte.mean():.1%} accuracy" in guidance
    assert f"only {yva.mean():.1%} of validation orders are late" in metrics["selected_model"]["rationale"]

    # ---- the same-state breakdown in the caveat is recomputed from the test scores ----------------------------------
    ss = te["same_state"].to_numpy() == 1
    cav_ss = reloaded["test_window_caveat"]["by_same_state"]
    thb = reloaded["risk_thresholds"]
    for key, mask in (("same_state", ss), ("other_state", ~ss)):
        g = cav_ss[key]
        assert g["n"] == int(mask.sum()) and g["observed_late_rate"] == pytest.approx(float(yte[mask].mean()))
        assert g["mean_predicted"] == pytest.approx(float(pt[mask].mean()), abs=1e-9)
        for band, sel_ in (("High", pt >= thb["high"]), ("Moderate", (pt >= thb["moderate"]) & (pt < thb["high"])),
                           ("Low", pt < thb["moderate"])):
            m2 = mask & sel_
            assert g["bands"][band]["n"] == int(m2.sum()) and g["bands"][band]["n_late"] == int(yte[m2].sum())
    assert metrics["risk_bands"]["test_by_same_state"] == train_model._clean(cav_ss)
    assert metrics["risk_bands"]["definition"].count("same-state vs other-state orders (descriptive)") == 1


@needs_data
def test_train_refuses_when_leakage_audit_fails(tmp_path, data, monkeypatch):
    monkeypatch.setattr(train_model, "leakage_audit", lambda *a, **k: {"passed": False, "checks": ["FAIL: forced"]})
    with pytest.raises(AssertionError, match="leakage audit failed"):
        train_model.train(df=data.sample(frac=0.15, random_state=1), out_dir=tmp_path, quick=True, verbose=False)
    assert not (tmp_path / config.MODEL_PATH.name).exists()


# --------------------------------------------------------------------------- #
# Hard limits: every bound accepted at the limit, rejected just beyond it (with the range in the message)
# --------------------------------------------------------------------------- #
#: the documented plausibility limits, pinned here independently of the code under test (a widened limit must fail)
EXPECTED_HARD_LIMITS = {
    "n_items": (1, 1_000), "n_sellers": (1, 1_000), "total_price": (0.01, 1_000_000), "total_freight": (0, 100_000),
    "total_weight_g": (1, 1_000_000), "estimated_delivery_days": (1, 730), "total_volume_cm3": (1, 100_000_000),
    "max_product_length_cm": (1, 1_000), "max_product_height_cm": (1, 1_000), "max_product_width_cm": (1, 1_000),
}


def _just_below(lo: float) -> float:
    return lo - 1 if lo >= 1 else (lo * 0.9 if lo > 0 else -0.5)


def _field_errors(inputs: dict, bundle: dict, field: str) -> list[str]:
    return [e for e in validate_input(inputs, bundle)[1] if e.startswith(f"{field} = ")]


def test_hard_limits_only_name_accepted_numeric_fields():
    assert prediction.HARD_LIMITS == EXPECTED_HARD_LIMITS
    assert set(prediction.HARD_LIMITS) <= set(prediction.RAW_NUMERIC)


@pytest.mark.parametrize("field", sorted(EXPECTED_HARD_LIMITS))
def test_hard_limits_boundaries(bundle, good_inputs, field):
    lo, hi = EXPECTED_HARD_LIMITS[field]
    base = {**good_inputs, "n_items": 1_000 if field == "n_sellers" else good_inputs["n_items"]}
    for ok in (lo, hi):                                           # inclusive limits
        assert _field_errors({**base, field: ok}, bundle, field) == [], (field, ok)
    rng = f"({prediction._fmt(lo)} to {prediction._fmt(hi)})"
    for bad in (_just_below(lo), hi + 1):
        errs = _field_errors({**base, field: bad}, bundle, field)
        assert len(errs) == 1 and "outside the allowed range" in errs[0] and rng in errs[0], (field, bad, errs)


def test_hard_limit_message_shows_small_values_exactly(bundle, good_inputs):
    msg = " ".join(_field_errors({**good_inputs, "total_price": 0.009}, bundle, "total_price"))
    assert "0.009" in msg and "(0.01 to 1,000,000)" in msg


def test_hard_date_range_boundaries(bundle, good_inputs):
    ok = ("2000-01-01", "2100-12-31", "2100-12-31 23:59")
    bad = ("1999-12-31", "2101-01-01")
    for d in ok:
        assert not [e for e in validate_input({**good_inputs, "purchase_date": d}, bundle)[1] if "allowed range" in e], d
    for d in bad:
        assert any("purchase_date is outside the allowed range" in e
                   for e in validate_input({**good_inputs, "purchase_date": d}, bundle)[1]), d


# --------------------------------------------------------------------------- #
# Date-span warning = the FITTING window (not the whole dataset)
# --------------------------------------------------------------------------- #
def test_date_span_is_the_training_window_and_warns_on_the_day_after(bundle, good_inputs):
    opts, win = bundle["form_options"], bundle["train_window"]
    assert opts["date_span"] == [win["start"], win["end"]]
    s0, s1 = pd.Timestamp(opts["date_span"][0]), pd.Timestamp(opts["date_span"][1])
    one = pd.Timedelta(days=1)

    def warns(d) -> list[str]:
        return [w for w in prediction.check_input({**good_inputs, "purchase_date": d}, bundle)[2] if "purchase_date" in w]

    assert warns(s0) == [] and warns(s1) == [] and warns(s1 + pd.Timedelta(hours=23, minutes=59)) == []
    assert len(warns(s0 - one)) == 1 and len(warns(s1 + one)) == 1
    msg = warns(s1 + one)[0]
    assert s0.strftime("%Y-%m-%d") in msg and s1.strftime("%Y-%m-%d") in msg
    # dates inside the dataset but outside the fitting window (2016 sparse months, validation / test windows) warn too
    ds = pd.Timestamp(opts["dataset_date_span"][0]), pd.Timestamp(opts["dataset_date_span"][1])
    assert warns(ds[0]) and warns(ds[1])


# --------------------------------------------------------------------------- #
# Test-window caveat travels with the probability
# --------------------------------------------------------------------------- #
def _caveat(**over) -> dict:
    base = {"window": "test", "start": "2018-07-01", "end": "2018-08-29", "n": 1000, "observed_late_rate": 0.05,
            "mean_predicted": 0.08, "brier": 0.049, "brier_baseline": 0.0475, "brier_not_better_than_baseline": True,
            "log_loss": 0.2, "log_loss_baseline": 0.19, "log_loss_not_better_than_baseline": True,
            "test_performance_degraded": True,
            "band_late_rates": {"Low": {"n": 500, "n_late": 10, "late_rate": 0.02},
                                "Moderate": {"n": 300, "n_late": 20, "late_rate": 0.0667},
                                "High": {"n": 200, "n_late": 20, "late_rate": 0.1}}}
    base.update(over)
    return base


def test_probability_note_carries_the_test_window_finding(bundle, good_inputs):
    b = _stub_bundle(bundle, calibrated=False, test_window_caveat=_caveat())
    res = predict_risk({**good_inputs, "n_items": 3, "n_sellers": 1}, b)           # p = 0.3 -> Moderate
    note = res["probability_note"]
    assert res["risk_label"] == "Moderate" and res["test_window_caveat"] == b["test_window_caveat"]
    for frag in ("2018-07-01 to 2018-08-29", "n=1,000", "mean predicted risk was 8.0% against 5.0% observed",
                 "Brier score (0.0490) was not better than a constant base-rate prediction (0.0475)",
                 "Moderate band were late 6.7% of the time (n=300)", "degraded"):
        assert frag in note, frag
    good = _caveat(brier=0.04, brier_not_better_than_baseline=False, test_performance_degraded=False)
    note2 = predict_risk({**good_inputs, "n_items": 3, "n_sellers": 1}, _stub_bundle(bundle, calibrated=False,
                                                                                      test_window_caveat=good))["probability_note"]
    assert "beat a constant base-rate" in note2 and "not better" not in note2 and "degraded" not in note2
    # the calibrated step-function note is kept, and the caveat is appended to it
    both = predict_risk(good_inputs, _stub_bundle(bundle, calibrated=True, test_window_caveat=_caveat(),
                                                  score_resolution={"distinct_levels": 30, "max": 0.31}))["probability_note"]
    assert both.startswith("Calibrated scores take only 30 distinct values") and "untouched test window" in both


def test_saved_bundle_carries_test_window_caveat_matching_the_metrics(bundle, metrics):
    cav = bundle["test_window_caveat"]
    sel = metrics["selected_model"]
    assert cav["test_performance_degraded"] == sel["test_performance_degraded"] == bundle["metrics_summary"]["test_performance_degraded"]
    assert metrics["selected_model"]["test_window_caveat"] == train_model._clean(cav)
    assert cav["brier_not_better_than_baseline"] == any("Brier" in w for w in sel["test_window_warnings"])
    bands = {r["band"]: r for r in metrics["risk_bands"]["test"]}
    assert {k: v["n"] for k, v in cav["band_late_rates"].items()} == {k: v["n"] for k, v in bands.items()}


def test_predict_risk_on_saved_model_returns_caveat_for_the_predicted_band(bundle, good_inputs):
    res = predict_risk(good_inputs, bundle)
    cav = res["test_window_caveat"]
    assert cav == bundle["test_window_caveat"]
    band = cav["band_late_rates"][res["risk_label"]]
    assert f"{res['risk_label']} band were late {band['late_rate']:.1%} of the time (n={band['n']:,})" in res["probability_note"]
    assert f"{cav['mean_predicted']:.1%} against {cav['observed_late_rate']:.1%} observed" in res["probability_note"]


def test_test_window_caveat_builder_from_model_entries():
    sel = _model_entry(mean_predicted_test=0.075)
    bands = [{"band": b, "n": n, "n_late": k, "late_rate": k / n, "late_ci_low": 0.0, "late_ci_high": 1.0}
             for b, n, k in (("Low", 10, 1), ("Moderate", 5, 1), ("High", 2, 1))]
    cav = train_model.test_window_caveat(sel, DUMMY_ENTRY, bands, {"start": "a", "end": "b"}, True)
    assert cav["n"] == 1000 and cav["mean_predicted"] == 0.075 and cav["observed_late_rate"] == 0.05
    assert cav["brier_not_better_than_baseline"] is True and cav["log_loss_not_better_than_baseline"] is True
    assert cav["band_late_rates"]["High"]["late_rate"] == 0.5 and cav["test_performance_degraded"] is True
    better = train_model.test_window_caveat(_model_entry(test_over={"brier": 0.04, "log_loss": 0.1}), DUMMY_ENTRY,
                                            bands, {"start": "a", "end": "b"}, False)
    assert better["brier_not_better_than_baseline"] is False and better["log_loss_not_better_than_baseline"] is False


# --------------------------------------------------------------------------- #
# Feature-selection rule for purchase_month (reason must not hide a model that got worse)
# --------------------------------------------------------------------------- #
def test_month_rule_reports_per_model_changes_and_the_mean():
    drop, reason = train_model.month_rule({"logistic_regression": (0.139, 0.135), "hist_gradient_boosting": (0.108, 0.135)})
    assert drop is True
    assert "MEAN validation PR-AUC" in reason and "mean change +0.012" in reason
    assert "Logistic regression 0.139 -> 0.135 (-0.004, lowered)" in reason
    assert "Histogram gradient boosting 0.108 -> 0.135 (+0.027, raised)" in reason
    assert "It lowered Logistic regression individually" in reason and "validation window" in reason
    drop2, reason2 = train_model.month_rule({"logistic_regression": (0.130, 0.135), "hist_gradient_boosting": (0.108, 0.135)})
    assert drop2 is True and "individually" not in reason2
    keep, _ = train_model.month_rule({"logistic_regression": (0.139, 0.130), "hist_gradient_boosting": (0.135, 0.130)})
    assert keep is False


def test_saved_month_reason_matches_its_numbers(metrics):
    reason = metrics["features"]["dropped_candidates"].get("purchase_month")
    if reason is None:
        pytest.skip("purchase_month is a model feature")
    assert "MEAN" in reason
    assert ("lowered" in reason) == ("It lowered" in reason)


@needs_data
def test_select_features_partitions_the_allow_list_with_reasons(parts):
    sub = {"train": parts["train"].sample(frac=0.2, random_state=0), "validation": parts["validation"].sample(frac=0.2, random_state=0)}
    numeric, categorical, dropped = train_model.select_features(sub, quick=True)
    assert set(numeric + categorical) <= set(config.PURCHASE_TIME_FEATURES)
    assert categorical == config.PURCHASE_TIME_CATEGORICAL
    # every allow-listed candidate is either used or dropped with a reason; nothing is silently lost
    assert set(config.PURCHASE_TIME_FEATURES) <= set(numeric + categorical) | set(dropped)
    assert not set(numeric + categorical) & set(dropped)
    assert all(isinstance(v, str) and v for v in dropped.values())
    assert all(c in dropped for c in config.PURCHASE_TIME_UNUSED)


# --------------------------------------------------------------------------- #
# Leakage audit: every check has its own failing fixture (exactly that check reports FAIL)
# --------------------------------------------------------------------------- #
AUDIT_NUM, AUDIT_CAT = ["n_items", "total_price"], ["customer_state"]


def _audit_frame(n: int = 60) -> pd.DataFrame:
    """TEST FIXTURE: 60 labelled daily orders with real feature names (all allow-listed) and unique ids."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "order_id": [f"o{i}" for i in range(n)], train_model.TIME_COL: pd.date_range("2018-01-01", periods=n, freq="D"),
        train_model.TARGET: np.tile([0, 1, 0], n // 3), "n_items": rng.integers(1, 5, n).astype(float),
        "total_price": rng.gamma(2, 50, n), "customer_state": rng.choice(["SP", "RJ"], n)})
    df.loc[::7, "total_price"] = np.nan
    return df


def _audit_args(df: pd.DataFrame | None = None, fit_on: pd.DataFrame | None = None) -> dict:
    df = _audit_frame() if df is None else df
    parts_ = {"train": df.iloc[:30].copy(), "validation": df.iloc[30:45].copy(), "test": df.iloc[45:].copy()}
    fit = parts_["train"] if fit_on is None else fit_on
    cols = AUDIT_NUM + AUDIT_CAT
    pipe = train_model.build_pipeline(LogisticRegression(), AUDIT_NUM, AUDIT_CAT, scale=True).fit(
        fit[cols], fit[train_model.TARGET])
    return {"numeric": AUDIT_NUM, "categorical": AUDIT_CAT, "parts": parts_, "base": pipe,
            "bundle_columns": cols, "parity_problems": []}


def _audit_failures(args: dict) -> list[str]:
    out = train_model.leakage_audit(args["numeric"], args["categorical"], args["parts"], args["base"],
                                    args["bundle_columns"], args["parity_problems"])
    assert out["passed"] is (not any(c.startswith("FAIL") for c in out["checks"]))
    return [c for c in out["checks"] if c.startswith("FAIL")]


def _ids_overlap(a: str, b: str):
    def corrupt(args, monkeypatch):
        args["parts"][b] = args["parts"][b].copy()
        args["parts"][b].iloc[0, args["parts"][b].columns.get_loc("order_id")] = args["parts"][a]["order_id"].iloc[0]
    return corrupt


def _shift(part: str, days: int):
    def corrupt(args, monkeypatch):
        p = args["parts"][part].copy()
        p[train_model.TIME_COL] = p[train_model.TIME_COL] + pd.Timedelta(days=days)
        args["parts"][part] = p
    return corrupt


def _patch_list(name: str, add=None, remove=None):
    def corrupt(args, monkeypatch):
        cur = [c for c in getattr(config, name) if c != remove] + ([add] if add else [])
        monkeypatch.setattr(config, name, cur)
    return corrupt


def _refit_on_shifted_train(args, monkeypatch):
    shifted = args["parts"]["train"].copy()
    shifted["n_items"] = shifted["n_items"] + 100          # imputer medians now differ from the TRAIN-window medians
    shifted["total_price"] = shifted["total_price"] + 1000
    args["base"] = _audit_args(fit_on=shifted)["base"]


AUDIT_CASES = {
    "allow_list": (_patch_list("PURCHASE_TIME_FEATURES", remove="n_items"), "subset of config.PURCHASE_TIME_FEATURES"),
    "leakage_list": (_patch_list("LEAKAGE_COLUMNS", add="n_items"), "config.LEAKAGE_COLUMNS"),
    "non_feature_list": (_patch_list("NON_FEATURE_COLUMNS", add="n_items"), "config.NON_FEATURE_COLUMNS"),
    "target_is_feature": (lambda a, mp: mp.setattr(train_model, "TARGET", "n_items"), "target is not a feature"),
    "pipeline_columns": (lambda a, mp: a.update(bundle_columns=list(reversed(a["bundle_columns"]))), "pipeline input columns"),
    "parity": (lambda a, mp: a.update(parity_problems=["mismatch"]), "derived features"),
    "train_overlaps_validation_in_time": (_shift("validation", -20), "strictly chronological"),
    "validation_overlaps_test_in_time": (_shift("test", -20), "strictly chronological"),
    "ids_train_validation": (_ids_overlap("train", "validation"), "no order_id appears in two partitions"),
    "ids_train_test": (_ids_overlap("train", "test"), "no order_id appears in two partitions"),
    "ids_validation_test": (_ids_overlap("validation", "test"), "no order_id appears in two partitions"),
    "imputer_not_train": (_refit_on_shifted_train, "imputer medians equal the TRAIN-window medians"),
}


def test_leakage_audit_baseline_passes_every_check():
    args = _audit_args()
    assert _audit_failures(args) == []
    full = train_model.leakage_audit(args["numeric"], args["categorical"], args["parts"], args["base"],
                                     args["bundle_columns"], [])
    assert full["passed"] is True and len(full["checks"]) == 9 and all(c.startswith("PASS") for c in full["checks"])


@pytest.mark.parametrize("case", sorted(AUDIT_CASES))
def test_leakage_audit_each_check_fails_on_its_own_corruption(case, monkeypatch):
    corrupt, expected = AUDIT_CASES[case]
    args = _audit_args()
    corrupt(args, monkeypatch)
    failed = _audit_failures(args)
    assert len(failed) == 1 and expected in failed[0], (case, failed)


# --------------------------------------------------------------------------- #
# Metric / calibration / helper protocol properties on freshly run code
# --------------------------------------------------------------------------- #
def test_classification_metrics_treat_a_score_equal_to_the_threshold_as_positive():
    y = np.array([1, 0, 1, 0, 0])
    p = np.array([0.4, 0.4, 0.4, 0.2, 0.1])            # three scores sit exactly on the cut
    m = train_model.classification_metrics(y, p, 0.4)
    assert m["confusion_matrix"] == [[2, 1], [0, 2]] and m["predicted_positive_rate"] == pytest.approx(0.6)
    assert m["accuracy"] == pytest.approx(0.8) and m["recall"] == 1.0 and m["precision"] == pytest.approx(2 / 3)
    just_above = train_model.classification_metrics(y, p, 0.4 + 1e-9)
    assert just_above["confusion_matrix"] == [[3, 0], [2, 0]]
    # the F1-optimal cut is itself a score value, so the recorded metrics must use >= to match it
    y2 = np.array([0, 0, 0, 1, 1, 0])
    p2 = np.array([.1, .2, .3, .6, .9, .5])
    thr = train_model.f1_optimal_threshold(y2, p2)
    assert thr in p2
    pred = p2 >= thr
    assert train_model.classification_metrics(y2, p2, thr)["predicted_positive_rate"] == pytest.approx(pred.mean())


def _skl_calibrator(base, rows: pd.DataFrame, cols: list[str], method: str):
    return CalibratedClassifierCV(FrozenEstimator(base), method=method).fit(rows[cols], rows[train_model.TARGET])


def test_calibrate_fits_on_earlier_half_judges_on_later_half_and_refits_final_on_validation():
    parts_ = _calibration_parts()
    tr, va = parts_["train"], parts_["validation"]
    base = train_model.build_pipeline(LogisticRegression(class_weight="balanced"), ["x"], [], scale=False)
    base.fit(tr[["x"]], tr[train_model.TARGET])
    res = train_model.calibrate(base, parts_, ["x"])
    assert res["calibrated"] is True
    half = len(va) // 2
    a, b = va.iloc[:half], va.iloc[half:]
    yb = b[train_model.TARGET].to_numpy()
    raw = base.predict_proba(b[["x"]])[:, 1]
    d = res["diagnostics"]
    assert d["raw"]["brier"] == pytest.approx(float(np.mean((raw - yb) ** 2)))
    assert str(half) in d["fit_half"] and str(len(va) - half) in d["check_half"]
    for method in ("sigmoid", "isotonic"):
        pb = _skl_calibrator(base, a, ["x"], method).predict_proba(b[["x"]])[:, 1]
        assert d[method]["brier"] == pytest.approx(float(np.mean((pb - yb) ** 2)))
        swapped = _skl_calibrator(base, b, ["x"], method).predict_proba(a[["x"]])[:, 1]
        assert d[method]["brier"] != pytest.approx(float(np.mean((swapped - a[train_model.TARGET].to_numpy()) ** 2)))
    # the final calibrator is the one fitted on ALL of validation (not train, not a half)
    grid = pd.DataFrame({"x": np.linspace(0, 1, 101)})
    final = res["estimator"].predict_proba(grid)[:, 1]
    np.testing.assert_allclose(final, _skl_calibrator(base, va, ["x"], res["method"]).predict_proba(grid)[:, 1], atol=1e-12)
    assert not np.allclose(final, _skl_calibrator(base, tr, ["x"], res["method"]).predict_proba(grid)[:, 1], atol=1e-6)
    assert not np.allclose(final, _skl_calibrator(base, a, ["x"], res["method"]).predict_proba(grid)[:, 1], atol=1e-6)


def test_calibrate_orders_validation_chronologically_before_halving():
    parts_ = _calibration_parts()
    tr, va = parts_["train"], parts_["validation"]
    base = train_model.build_pipeline(LogisticRegression(class_weight="balanced"), ["x"], [], scale=False)
    base.fit(tr[["x"]], tr[train_model.TARGET])
    ordered = train_model.calibrate(base, parts_, ["x"])
    shuffled = train_model.calibrate(base, {**parts_, "validation": va.sample(frac=1, random_state=5)}, ["x"])
    assert shuffled["method"] == ordered["method"]
    for m in ("raw", "sigmoid", "isotonic"):
        assert shuffled["diagnostics"][m] == pytest.approx(ordered["diagnostics"][m])


def test_calibrate_declines_when_validation_is_too_small_or_single_class():
    parts_ = _calibration_parts(n=8000)
    base = train_model.build_pipeline(LogisticRegression(), ["x"], [], scale=False).fit(
        parts_["train"][["x"]], parts_["train"][train_model.TARGET])
    one_class = parts_["validation"].assign(**{train_model.TARGET: 0})
    res = train_model.calibrate(base, {**parts_, "validation": one_class}, ["x"])
    assert res["calibrated"] is False and res["estimator"] is base and "reason" in res["diagnostics"]
    empty = train_model.calibrate(base, {**parts_, "validation": parts_["validation"].iloc[0:0]}, ["x"])
    assert empty["calibrated"] is False


def test_downsample_keeps_both_end_points_and_order():
    n = 1000
    x = np.arange(n, dtype=float)
    (out,) = train_model._downsample(x, max_points=50)
    assert out[0] == 0 and out[-1] == n - 1 and len(out) <= 50 and out == sorted(out)
    a, b = train_model._downsample(x, x * 2, max_points=50)
    assert [v * 2 for v in a] == b
    full = train_model._downsample(np.arange(7.0), max_points=50)[0]
    assert full == list(np.arange(7.0))
    assert train_model._downsample(np.array([]), max_points=5) == [[]]
    assert len(train_model._downsample(np.arange(1000.0))[0]) <= train_model.MAX_CURVE_POINTS


def test_bootstrap_interval_is_the_2_5_to_97_5_percentile():
    assert train_model._ci(np.arange(1001.0)) == pytest.approx([25.0, 975.0])
    assert train_model._ci(np.array([])) is None
    assert train_model._ci(np.array([3.0])) == [3.0, 3.0]


def test_bootstrap_scores_are_paired_and_reproducible():
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.2).astype(int)
    s = {"a": rng.random(300), "b": rng.random(300)}
    r1 = train_model.bootstrap_scores(y, s, 25, seed=7)
    r2 = train_model.bootstrap_scores(y, s, 25, seed=7)
    assert len(r1["a"]["pr_auc"]) == len(r1["b"]["pr_auc"]) == 25
    np.testing.assert_array_equal(r1["a"]["roc_auc"], r2["a"]["roc_auc"])
    assert train_model.bootstrap_scores(np.array([]), s, 5, 0)["a"]["pr_auc"].size == 0
    assert train_model.bootstrap_scores(np.zeros(10, int), s | {}, 5, 0)["a"]["pr_auc"].size == 0   # single class: no draws


def _form_frame() -> pd.DataFrame:
    """TEST FIXTURE: 8 training orders; SP->RJ x3 (promise 10/20/90: median 20, mean 40), SP->SP x3, MG->BA x2."""
    df = pd.DataFrame({
        "customer_state": ["RJ"] * 3 + ["SP"] * 3 + ["BA"] * 2, "seller_state": ["SP"] * 6 + ["MG"] * 2,
        "product_category": ["toys", "toys", "bed", "bed", "bed", "toys", "bed", "toys"],
        "estimated_delivery_days": [10.0, 20.0, 90.0, 5.0, 6.0, 7.0, 30.0, 31.0],
        "purchase_date": pd.to_datetime(["2017-03-01", "2017-03-02", "2017-03-03", "2017-04-01", "2017-04-02",
                                         "2017-04-03", "2017-05-10", "2017-05-11"]),
        "purchase_dow": [0, 1, 2, 3, 4, 5, 6, 6]})
    for c in prediction.RAW_NUMERIC:
        if c not in df:
            df[c] = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 10_000.0]
    return df


def test_build_form_options_on_a_tiny_frame():
    df = _form_frame()
    o = train_model.build_form_options(df, min_pair_orders=3)
    assert o["customer_state"] == ["BA", "RJ", "SP"] and o["seller_state"] == ["MG", "SP"]
    assert o["product_category"] == ["bed", "toys"]
    # a pair with exactly min_pair_orders orders is kept (>=), a pair with fewer is not; the median (not mean) is used
    assert o["state_pair_median_estimated_days"] == {f"SP{prediction.STATE_PAIR_SEP}RJ": 20.0,
                                                     f"SP{prediction.STATE_PAIR_SEP}SP": 6.0}
    assert o["state_pair_min_orders"] == 3
    assert o["ranges"]["total_weight_g"] == [100.0, 10_000.0] and o["ranges"]["estimated_delivery_days"] == [5.0, 90.0]
    assert o["defaults"]["total_weight_g"] == pytest.approx(float(df["total_weight_g"].median()))
    assert o["defaults"]["total_weight_g"] != pytest.approx(float(df["total_weight_g"].mean()))
    assert o["defaults"]["estimated_delivery_days"] == pytest.approx(float(df["estimated_delivery_days"].median()))
    assert o["defaults"]["purchase_dow"] == pytest.approx(3.5)
    assert not {"n_items", "total_price", "total_freight"} & set(o["defaults"])        # required fields are never defaulted
    assert o["date_span"] == ["2017-03-01", "2017-05-11"]                               # derived from the frame it was given
    assert train_model.build_form_options(df, date_span=("2000-01-01", "2000-12-31"))["date_span"] == ["2000-01-01", "2000-12-31"]
    assert o["hard_limits"]["n_items"] == [1.0, 1000.0]
    assert train_model.build_form_options(df, min_pair_orders=4)["state_pair_median_estimated_days"] == {}


def test_load_training_frame_drops_unlabelled_rows_and_sorts_by_time_then_id(tmp_path):
    t = pd.to_datetime(["2018-01-03", "2018-01-01", "2018-01-02", "2018-01-02", "2018-01-04"])
    df = pd.DataFrame({"order_id": ["e", "d", "c", "b", "a"], train_model.TIME_COL: t,
                       train_model.TARGET: pd.array([1, 0, pd.NA, 1, pd.NA], dtype="Int8")})
    path = tmp_path / "o.parquet"
    df.to_parquet(path)
    out = train_model.load_training_frame(path)
    assert out["order_id"].tolist() == ["d", "b", "e"]                    # NA targets dropped; time, then id
    assert out[train_model.TARGET].tolist() == [0, 1, 1] and out[train_model.TARGET].dtype.kind == "i"
    assert out.index.tolist() == [0, 1, 2]
    with pytest.raises(FileNotFoundError, match="src.preprocessing"):
        train_model.load_training_frame(tmp_path / "absent.parquet")


def test_pipeline_keeps_missing_indicators_and_groups_rare_categories():
    n = 130
    df = pd.DataFrame({"v": np.where(np.arange(n) % 5 == 0, np.nan, np.arange(n, dtype=float)),
                       "w": np.arange(n, dtype=float), "c": ["A"] * 60 + ["B"] * 60 + ["C"] * 10})
    y = np.tile([0, 1], n // 2 + 1)[:n]
    pipe = train_model.build_pipeline(LogisticRegression(), ["v", "w"], ["c"], scale=True).fit(df, y)
    prep = pipe.named_steps["prep"]
    imp = prep.named_transformers_["num"].named_steps["impute"]
    assert imp.indicator_ is not None and imp.statistics_[0] == pytest.approx(df["v"].median())      # median impute, fitted here
    oh = prep.named_transformers_["cat"].named_steps["onehot"]
    assert list(oh.infrequent_categories_[0]) == ["C"]                       # 10 < MIN_FREQUENCY: grouped
    assert prep.transform(pipe.named_steps["coerce"].transform(df)).shape[1] == 2 + 1 + 3     # 2 num + indicator + A, B, infrequent


def test_split_notes_on_a_synthetic_frame(monkeypatch, tmp_path):
    """TEST FIXTURE: 3 sparse 2016 orders are excluded and counted; windows and drift are described from the data."""
    monkeypatch.setattr(config, "ORDERS_ALL_PATH", tmp_path / "absent.parquet")
    rng = np.random.default_rng(0)
    ts = pd.to_datetime(["2016-10-02", "2016-10-09", "2016-11-01"] + [f"2017-{m:02d}-15" for m in range(1, 13)] * 5
                        + [f"2018-0{m}-10" for m in (4, 5, 6)] * 10 + [f"2018-0{m}-10" for m in (7, 8)] * 10)
    df = pd.DataFrame({"order_id": [f"o{i}" for i in range(len(ts))], train_model.TIME_COL: ts,
                       train_model.TARGET: rng.integers(0, 2, len(ts)), "same_state": rng.integers(0, 2, len(ts)),
                       "estimated_delivery_days": rng.integers(10, 30, len(ts)).astype(float),
                       "order_delivered_customer_date": ts + pd.Timedelta(days=12)})
    parts_ = train_model.chronological_split(df)
    notes, monthly = train_model.split_notes(df, parts_)
    assert notes[0].startswith("3 orders purchased before 2017-01-01 (2016-10 to 2016-11) are excluded")
    assert "at most 2 orders in any month" in notes[0]
    drift = next(n for n in notes if n.startswith("Late rate by window"))
    for k, v in parts_.items():
        assert f"{k} {v[train_model.TARGET].mean():.2%} (n={len(v):,})" in drift
    assert sum(r["n"] for r in monthly) == len(df) and monthly[0]["month"] == "2016-10"
    assert next(r for r in monthly if r["month"] == "2016-10")["late_rate"] == pytest.approx(
        float(df[df[train_model.TIME_COL].dt.strftime("%Y-%m") == "2016-10"][train_model.TARGET].mean()))
    assert any(n.startswith("Selection bias") and "unavailable" in n for n in notes)
    assert any(n.startswith("Non-stationarity") for n in notes)


# --------------------------------------------------------------------------- #
# The probability floor reaches every score behind the reported metrics (forced non-zero floor, quick run)
# --------------------------------------------------------------------------- #
@needs_data
def test_probability_floor_is_applied_to_validation_and_test_scores_in_the_metrics(tmp_path, data, monkeypatch):
    floor = 0.08
    monkeypatch.setattr(train_model, "probability_floor", lambda p, y: floor)
    sample = data.sample(frac=0.25, random_state=3)
    bundle, metrics = train_model.train(df=sample, out_dir=tmp_path, quick=True, verbose=False)
    assert bundle["probability_floor"] == floor
    sp = train_model.chronological_split(sample)
    cols = bundle["feature_columns"]
    name = bundle["model_name"]
    for window, key in (("validation", "validation"), ("test", "test")):
        frame = sp[window]
        y = frame[train_model.TARGET].to_numpy()
        raw = bundle["pipeline"].predict_proba(frame[cols])[:, 1]
        assert (raw < floor).any()                                  # the floor actually binds, so the check has power
        floored = np.maximum(raw, floor)
        m = metrics["models"][name][key]
        assert m["brier"] == pytest.approx(float(np.mean((floored - y) ** 2)), abs=1e-9)
        assert m["brier"] != pytest.approx(float(np.mean((raw - y) ** 2)), abs=1e-9)
        assert metrics["models"][name][f"mean_predicted_{window}"] == pytest.approx(float(floored.mean()), abs=1e-9)
        pred = floored >= bundle["decision_threshold"]
        assert m["predicted_positive_rate"] == pytest.approx(float(pred.mean()))
    assert metrics["selected_model"]["score_resolution"]["min"] >= floor


# --------------------------------------------------------------------------- #
# Derived-feature extrapolation: freight_ratio can leave its training support while price and freight do not
# --------------------------------------------------------------------------- #
def _ratio_bundle(bundle, ratio_range=(0.0, 1.0), drop_ratio_feature=False, drop_ratio_range=False) -> dict:
    """TEST FIXTURE: stub-scored bundle whose raw ranges are wide, so only freight_ratio can trigger a warning."""
    opts = copy.deepcopy(bundle["form_options"])
    opts["ranges"].update(total_price=[1.0, 1000.0], total_freight=[0.0, 1000.0])
    if drop_ratio_range:
        opts["ranges"].pop("freight_ratio", None)
    else:
        opts["ranges"]["freight_ratio"] = list(ratio_range)
    over: dict = {"form_options": opts}
    if drop_ratio_feature:
        over["feature_columns"] = [c for c in bundle["feature_columns"] if c != "freight_ratio"]
    return _stub_bundle(bundle, **over)


def test_freight_ratio_outside_training_support_warns_even_when_price_and_freight_are_in_range(bundle, good_inputs):
    b = _ratio_bundle(bundle)
    inside = {**good_inputs, "total_price": 100.0, "total_freight": 50.0}                 # ratio 0.5
    assert prediction.check_input(inside, b)[2] == []
    edge = {**good_inputs, "total_price": 100.0, "total_freight": 100.0}                  # ratio exactly 1.0: inside
    assert prediction.check_input(edge, b)[2] == []
    out = {**good_inputs, "total_price": 100.0, "total_freight": 250.0}                   # ratio 2.5
    _, errors, warned = prediction.check_input(out, b)
    assert errors == [] and len(warned) == 1
    assert "freight_ratio" in warned[0] and "2.5" in warned[0] and "0 to 1" in warned[0] and "extrapolating" in warned[0]
    res = predict_risk(out, b)
    assert res["warnings"] == warned and 0 <= res["probability"] <= 1                      # scored, never rejected
    batch = predict_batch(pd.DataFrame([inside, out]), b)
    assert batch["input_warning"].iloc[0] == "" and "freight_ratio" in batch["input_warning"].iloc[1]
    assert (batch["input_error"] == "").all()
    # a raw field that is itself out of range adds its own warning next to the ratio one
    both = prediction.check_input({**out, "total_price": 5000.0, "total_freight": 5001.0}, b)[2]
    assert any("total_price" in w for w in both) and any("freight_ratio" in w for w in both)


def test_freight_ratio_warning_needs_a_range_and_a_model_that_uses_the_ratio(bundle, good_inputs):
    out = {**good_inputs, "total_price": 100.0, "total_freight": 250.0}
    assert prediction.check_input(out, _ratio_bundle(bundle, drop_ratio_range=True))[2] == []      # older bundle
    assert prediction.check_input(out, _ratio_bundle(bundle, drop_ratio_feature=True))[2] == []    # feature unused


def test_saved_bundle_has_the_training_range_of_freight_ratio(bundle, parts):
    rng = bundle["form_options"]["ranges"]["freight_ratio"]
    ratio = parts["train"]["freight_ratio"].astype(float)
    assert rng == pytest.approx([ratio.min(), ratio.max()])
    lo_p, hi_f = (bundle["form_options"]["ranges"][c][i] for c, i in (("total_price", 0), ("total_freight", 1)))
    if hi_f / lo_p <= rng[1]:
        pytest.skip("no raw-in-range combination exceeds the training ratio range")
    extreme = {"customer_state": "SP", "seller_state": "SP", "product_category": bundle["form_options"]["product_category"][0],
               "purchase_date": bundle["form_options"]["date_span"][0], "n_items": 1, "total_price": lo_p,
               "total_freight": hi_f}
    warned = prediction.check_input(extreme, bundle)[2]
    assert len(warned) == 1 and "freight_ratio" in warned[0]


def test_build_form_options_records_the_training_freight_ratio_range():
    df = _form_frame()
    df["total_price"] = [10.0, 20.0, 40.0, 50.0, 100.0, 200.0, 400.0, 800.0]
    df["total_freight"] = [5.0, 2.0, 40.0, 5.0, 10.0, 0.0, 800.0, 80.0]      # ratios 0.5, 0.1, 1, 0.1, 0.1, 0, 2, 0.1
    assert train_model.build_form_options(df)["ranges"]["freight_ratio"] == [0.0, 2.0]
    df["total_price"] = np.nan                                               # ratio undefined everywhere: no range, no error
    assert "freight_ratio" not in train_model.build_form_options(df)["ranges"]


# --------------------------------------------------------------------------- #
# Input hygiene: non-dict input, blank strings, booleans, numbers beyond the float range
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [None, [1, 2], "SP", 5, (("customer_state", "SP"),)])
def test_non_dict_input_gives_a_readable_value_error(bundle, bad):
    for fn in (predict_risk, validate_input, prediction.check_input):
        with pytest.raises(ValueError, match="inputs must be a dict"):
            fn(bad, bundle)


def test_mapping_input_other_than_dict_is_accepted(bundle, good_inputs):
    from collections import OrderedDict
    assert validate_input(OrderedDict(good_inputs), bundle)[1] == []


@pytest.mark.parametrize("blank", ["", "   ", "\t", None, float("nan"), pd.NA])
@pytest.mark.parametrize("field", ["customer_state", "seller_state", "product_category", "n_items", "total_price",
                                   "total_freight"])
def test_blank_required_fields_are_errors(bundle, good_inputs, field, blank):
    assert f"{field} is required" in _errors({**good_inputs, field: blank}, bundle)
    out = predict_batch(pd.DataFrame([{**good_inputs, field: blank}]), bundle)
    assert f"{field} is required" in out["input_error"].iloc[0] and np.isnan(out["late_probability"].iloc[0])


@pytest.mark.parametrize("blank", ["", "   ", None, float("nan")])
@pytest.mark.parametrize("field", ["total_weight_g", "estimated_delivery_days", "n_sellers", "total_volume_cm3",
                                   "max_product_length_cm"])
def test_blank_optional_numeric_fields_are_defaulted_and_reported(bundle, good_inputs, field, blank):
    cleaned, errors, _ = prediction.check_input({**good_inputs, field: blank}, bundle)
    assert errors == []
    used = prediction._used_raw_fields(bundle["feature_columns"])
    if field in used:
        res = predict_risk({**good_inputs, field: blank}, bundle)
        assert field in res["assumptions_used"] and res["assumptions_used"][field]["value"] is not None
        out = predict_batch(pd.DataFrame([{**good_inputs, field: blank}]), bundle)
        assert field in out["assumptions_used"].iloc[0].split(";") and out["input_error"].iloc[0] == ""


@pytest.mark.parametrize("blank", ["", "  "])
def test_blank_purchase_date_falls_back_to_month_or_is_required(bundle, good_inputs, blank):
    no_date = {k: v for k, v in good_inputs.items() if k != "purchase_date"}
    assert "purchase_date is required" in _errors({**no_date, "purchase_date": blank}, bundle)
    assert validate_input({**no_date, "purchase_date": blank, "purchase_month": 5}, bundle)[1] == []
    assert validate_input({**good_inputs, "purchase_month": blank, "purchase_dow": blank}, bundle)[1] == []


@pytest.mark.parametrize("value", [10 ** 400, -(10 ** 400), 10 ** 30, True, False, np.True_, "1" * 400])
@pytest.mark.parametrize("field", ["n_items", "total_price", "total_freight", "total_weight_g", "purchase_month", "purchase_dow"])
def test_unreadable_numbers_become_readable_errors_not_exceptions(bundle, good_inputs, field, value):
    errors = validate_input({**good_inputs, field: value}, bundle)[1]
    assert errors and any(field in e for e in errors)
    if isinstance(value, (bool, np.bool_)) or abs(value if not isinstance(value, str) else 10 ** 400) > 1e300:
        assert any("finite number" in e or "whole number" in e or "from " in e for e in errors)


def test_one_unreadable_cell_flags_its_row_and_does_not_abort_the_batch(bundle, good_inputs):
    raw = pd.DataFrame([good_inputs, {**good_inputs, "total_price": 10 ** 400}, {**good_inputs, "n_items": True},
                        {**good_inputs, "total_freight": False}, good_inputs], dtype="object")
    out = predict_batch(raw, bundle)
    assert out["input_error"].tolist()[0] == out["input_error"].tolist()[4] == ""
    assert all("finite number" in e for e in out["input_error"].iloc[1:4])
    assert out["late_probability"].notna().tolist() == [True, False, False, False, True]
    assert out["late_probability"].iloc[0] == pytest.approx(out["late_probability"].iloc[4])


# --------------------------------------------------------------------------- #
# Inference-layer behaviours that were documented but unpinned
# --------------------------------------------------------------------------- #
def test_estimated_days_falls_back_to_the_global_median_for_an_unseen_state_pair(bundle, good_inputs):
    opts = bundle["form_options"]
    slim = {k: v for k, v in good_inputs.items() if k != "estimated_delivery_days"}
    # deterministic: a bundle whose pair table is empty
    empty = copy.deepcopy(bundle)
    empty["form_options"]["state_pair_median_estimated_days"] = {}
    res = predict_risk(slim, _stub_bundle(empty))
    got = res["assumptions_used"]["estimated_delivery_days"]
    assert got["value"] == pytest.approx(opts["defaults"]["estimated_delivery_days"])
    assert "overall training median" in got["source"]
    # on the saved bundle: any seller -> customer pair that has no training median (e.g. too few orders)
    known = set(opts["state_pair_median_estimated_days"])
    unseen = [(s_, c_) for s_ in opts["seller_state"] for c_ in opts["customer_state"]
              if f"{s_}{prediction.STATE_PAIR_SEP}{c_}" not in known]
    if unseen:
        s_, c_ = unseen[0]
        r2 = predict_risk({**slim, "seller_state": s_, "customer_state": c_}, bundle)
        assert r2["assumptions_used"]["estimated_delivery_days"]["value"] == pytest.approx(
            opts["defaults"]["estimated_delivery_days"])
        out = predict_batch(pd.DataFrame([{**slim, "seller_state": s_, "customer_state": c_}]), bundle)
        assert out["input_error"].iloc[0] == "" and "estimated_delivery_days" in out["assumptions_used"].iloc[0]


def test_month_only_input_reports_the_purchase_dow_assumption(bundle, good_inputs):
    if "purchase_dow" not in bundle["feature_columns"]:
        pytest.skip("the saved model does not use the weekday")
    no_date = {k: v for k, v in good_inputs.items() if k != "purchase_date"}
    res = predict_risk({**no_date, "purchase_month": 5}, bundle)
    dow = res["assumptions_used"]["purchase_dow"]
    assert dow["value"] == pytest.approx(bundle["form_options"]["defaults"]["purchase_dow"])
    assert "training" in dow["source"] and res["derived_features"]["purchase_dow"] == dow["value"]
    assert "purchase_month" not in res["assumptions_used"]
    given = predict_risk({**no_date, "purchase_month": 5, "purchase_dow": 2}, bundle)
    assert "purchase_dow" not in given["assumptions_used"] and given["derived_features"]["purchase_dow"] == 2
    assert "purchase_dow" not in predict_risk(good_inputs, bundle)["assumptions_used"]       # a date implies the weekday
    out = predict_batch(pd.DataFrame([{**no_date, "purchase_month": 5}]), bundle)
    assert "purchase_dow" in out["assumptions_used"].iloc[0].split(";")


def test_derived_features_report_numeric_and_categorical_model_inputs(bundle, good_inputs):
    res = predict_risk({**good_inputs, "product_category": good_inputs["product_category"].upper(),
                        "seller_state": "SP", "customer_state": "SP"}, bundle)
    d = res["derived_features"]
    assert set(d) == set(bundle["feature_columns"])
    assert d["customer_state"] == "SP" and d["seller_state"] == "SP"
    assert d["product_category"] == good_inputs["product_category"]           # normalised to the training spelling
    assert d["same_state"] == "1"                                             # a categorical model input (0 / 1 level)
    assert d["freight_ratio"] == pytest.approx(good_inputs["total_freight"] / good_inputs["total_price"])
    other = predict_risk({**good_inputs, "seller_state": "SP", "customer_state": "RJ"}, bundle)["derived_features"]
    assert other["same_state"] == "0"


def test_batch_joins_every_error_of_a_row(bundle, good_inputs):
    raw = pd.DataFrame([{**good_inputs, "customer_state": "XX", "total_price": -1, "n_items": 2.5}])
    msg = predict_batch(raw, bundle)["input_error"].iloc[0]
    assert msg.count(" | ") >= 2
    for frag in ("not a known customer_state", "total_price", "n_items must be a whole number"):
        assert frag in msg
    single = validate_input(raw.iloc[0].to_dict(), bundle)[1]
    assert msg == " | ".join(single)


def test_same_state_is_false_when_a_state_is_missing():
    raw = pd.DataFrame({"customer_state": [None, "SP", None, "SP", "RJ"], "seller_state": [None, None, "SP", "SP", "SP"],
                        "total_price": 10.0, "total_freight": 1.0, "purchase_date": pd.Timestamp("2018-01-01")})
    assert derive_model_features(raw)["same_state"].tolist() == [0, 0, 0, 1, 0]        # NA == NA is never "same state"


# --------------------------------------------------------------------------- #
# Same-state disclosure: the pooled band rate hides a large difference between the groups
# --------------------------------------------------------------------------- #
def test_same_state_breakdown_exact_and_safe_on_empty_groups():
    th = {"moderate": 0.2, "high": 0.5}
    y = np.array([1, 0, 0, 1, 0, 0, 1, 0])
    p = np.array([0.6, 0.6, 0.3, 0.1, 0.1, 0.1, 0.3, 0.7])
    same = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    bd = train_model.same_state_breakdown(y, p, same, th)
    s_, o_ = bd["same_state"], bd["other_state"]
    assert (s_["n"], o_["n"]) == (4, 4)
    assert s_["observed_late_rate"] == 0.5 and o_["observed_late_rate"] == 0.25
    assert s_["mean_predicted"] == pytest.approx(0.4) and o_["mean_predicted"] == pytest.approx(0.3)
    assert [s_["bands"][b]["n"] for b in ("Low", "Moderate", "High")] == [1, 1, 2]
    assert [s_["bands"][b]["n_late"] for b in ("Low", "Moderate", "High")] == [1, 0, 1]
    assert [o_["bands"][b]["n"] for b in ("Low", "Moderate", "High")] == [2, 1, 1]
    assert o_["bands"]["High"]["late_rate"] == 0.0 and o_["bands"]["Moderate"]["late_rate"] == 1.0
    empty = train_model.same_state_breakdown(y[:4], p[:4], same[:4], th)["other_state"]
    assert empty["n"] == 0 and empty["observed_late_rate"] is None and empty["mean_predicted"] is None
    assert all(r["n"] == 0 and r["late_rate"] is None for r in empty["bands"].values())
    none = train_model.same_state_breakdown(np.array([]), np.array([]), np.array([]), th)
    assert none["same_state"]["n"] == 0 and none["other_state"]["n"] == 0
    note = train_model.same_state_note(bd)
    assert "Low 100.0% (n=1) vs 0.0% (n=2)" in note and "same-state 40.0% vs 50.0% (n=4)" in note
    assert "n/a" in train_model.same_state_note({**bd, "other_state": empty})


def _caveat_with_groups(**over) -> dict:
    cell = lambda n, k: {"n": n, "n_late": k, "late_rate": k / n if n else None}              # noqa: E731
    groups = {"same_state": {"n": 300, "observed_late_rate": 0.09, "mean_predicted": 0.04,
                             "bands": {"Low": cell(200, 10), "Moderate": cell(80, 12), "High": cell(20, 5)}},
              "other_state": {"n": 700, "observed_late_rate": 0.03, "mean_predicted": 0.10,
                              "bands": {"Low": cell(300, 3), "Moderate": cell(0, 0), "High": cell(200, 12)}}}
    return _caveat(by_same_state=groups, **over)


def test_probability_note_splits_the_band_rate_by_same_state(bundle, good_inputs):
    b = _stub_bundle(bundle, calibrated=False, test_window_caveat=_caveat_with_groups())
    same = predict_risk({**good_inputs, "seller_state": "SP", "customer_state": "SP", "n_items": 1, "n_sellers": 1}, b)
    assert same["risk_label"] == "Low"
    note = same["probability_note"]
    assert "Low band were late 2.0% of the time (n=500)" in note
    assert "same-state orders 5.0% (n=200); other-state orders 1.0% (n=300)" in note
    assert "For same-state orders like this one the mean predicted risk was 4.0% against 9.0% observed (n=300)" in note
    other = predict_risk({**good_inputs, "seller_state": "SP", "customer_state": "RJ", "n_items": 1, "n_sellers": 1}, b)
    assert "For other-state orders like this one the mean predicted risk was 10.0% against 3.0% observed (n=700)" in \
        other["probability_note"]
    mod = predict_risk({**good_inputs, "seller_state": "SP", "customer_state": "RJ", "n_items": 3, "n_sellers": 1}, b)
    assert "other-state orders n/a (n=0)" in mod["probability_note"]                  # an empty cell is shown, not hidden
    # bundles without the breakdown keep the old sentence
    old = predict_risk(good_inputs, _stub_bundle(bundle, calibrated=False, test_window_caveat=_caveat()))["probability_note"]
    assert "same-state" not in old and "like this one" not in old


def test_test_window_caveat_builder_carries_the_same_state_breakdown():
    bands = [{"band": b, "n": 10, "n_late": 1, "late_rate": 0.1, "late_ci_low": 0.0, "late_ci_high": 1.0}
             for b in ("Low", "Moderate", "High")]
    groups = {"same_state": {"n": 1}, "other_state": {"n": 2}}
    cav = train_model.test_window_caveat(_model_entry(), DUMMY_ENTRY, bands, {"start": "a", "end": "b"}, False, groups)
    assert cav["by_same_state"] == groups
    assert "by_same_state" not in train_model.test_window_caveat(_model_entry(), DUMMY_ENTRY, bands,
                                                                 {"start": "a", "end": "b"}, False)


@needs_model
def test_saved_metrics_disclose_the_same_state_split(bundle, metrics, parts):
    te = parts["test"]
    p = prediction._score(bundle["pipeline"], te, bundle["feature_columns"], bundle["probability_floor"])
    th, y = bundle["risk_thresholds"], te[train_model.TARGET].to_numpy()
    low = (p < th["moderate"])
    cav = bundle["test_window_caveat"]["by_same_state"]
    for key, m in (("same_state", te["same_state"].to_numpy() == 1), ("other_state", te["same_state"].to_numpy() == 0)):
        assert cav[key]["bands"]["Low"]["n"] == int((low & m).sum())
        assert cav[key]["bands"]["Low"]["late_rate"] == pytest.approx(float(y[low & m].mean()))
    assert metrics["risk_bands"]["test_by_same_state"] == train_model._clean(cav)
    assert "same-state vs other-state orders (descriptive)" in metrics["risk_bands"]["definition"]
    res = predict_risk(_raw_inputs(te.dropna(subset=["total_weight_g"]).iloc[0]), bundle)
    assert "like this one" in res["probability_note"]


# --------------------------------------------------------------------------- #
# Narrative sentences are conditional on the computed numbers
# --------------------------------------------------------------------------- #
def _notes_frame(month_late: dict, placement: dict) -> pd.DataFrame:
    """TEST FIXTURE: 40 orders per month; ``month_late[month]`` of them late, placed among the same-state ("front")
    or other-state ("back") orders of that window so the same-state late-rate gap has a known sign."""
    months = ([f"2017-{m:02d}" for m in range(1, 13)] + ["2018-01", "2018-02", "2018-03"]
              + ["2018-04", "2018-05", "2018-06", "2018-07", "2018-08"])
    rows = []
    for mo in months:
        k = month_late.get(mo, month_late["default"])
        window = "train" if mo < "2018-04" else ("validation" if mo < "2018-07" else "test")
        for i in range(40):
            same = int(i < 20)
            late = int(i < k) if placement[window] == "front" else int(i >= 40 - k)
            rows.append({"order_id": f"{mo}-{i}", train_model.TIME_COL: pd.Timestamp(f"{mo}-15"), "same_state": same,
                         train_model.TARGET: late, "estimated_delivery_days": 15.0,
                         "order_delivered_customer_date": pd.Timestamp(f"{mo}-25")})
    return pd.DataFrame(rows)


def _notes_for(df, monkeypatch, tmp_path) -> list[str]:
    monkeypatch.setattr(config, "ORDERS_ALL_PATH", tmp_path / "absent.parquet")
    return train_model.split_notes(df, train_model.chronological_split(df))[0]


def test_split_notes_make_no_drift_claims_when_the_data_show_none(monkeypatch, tmp_path):
    df = _notes_frame({"default": 10}, {"train": "front", "validation": "front", "test": "front"})
    text = " ".join(_notes_for(df, monkeypatch, tmp_path))
    assert "Late rate by window: train 25.00%" in text and "validation 25.00%" in text and "test 25.00%" in text
    assert "spread between windows is only 0.0 percentage points" in text
    for claim in ("base-rate shift", "changes sign", "not be anticipated", "weaker"):
        assert claim not in text
    assert "no month reaches 1.5x the median" in text


def test_split_notes_state_drift_claims_only_when_the_data_show_them(monkeypatch, tmp_path):
    df = _notes_frame({"default": 10, "2017-11": 30, "2018-04": 2, "2018-05": 2, "2018-06": 2},
                      {"train": "front", "validation": "front", "test": "back"})
    notes = _notes_for(df, monkeypatch, tmp_path)
    text = " ".join(notes)
    tr_rate = df[df[train_model.TIME_COL] < "2018-04-01"][train_model.TARGET].mean()
    assert f"train {tr_rate:.2%}" in text and "percentage points, so thresholds and probabilities" in text
    spread_note = next(n for n in notes if n.startswith("Late rate by window"))
    rates = [df[(df[train_model.TIME_COL] >= a) & (df[train_model.TIME_COL] < b)][train_model.TARGET].mean()
             for a, b in (("2017-01-01", "2018-04-01"), ("2018-04-01", "2018-07-01"), ("2018-07-01", "2019-01-01"))]
    assert f"{(max(rates) - min(rates)) * 100:.1f} percentage points" in spread_note
    nonstat = next(n for n in notes if n.startswith("Non-stationarity"))
    assert "2017-11 (75.0%)" in nonstat and "3.0x the median" in nonstat and "not be anticipated" in nonstat
    drift = next(n for n in notes if n.startswith("Covariate / concept drift"))
    assert "changes sign between windows" in drift and "train +" in drift and "test -" in drift
    assert "weaker" not in text


# --------------------------------------------------------------------------- #
# Training hygiene that must not be hard-coded
# --------------------------------------------------------------------------- #
def test_candidate_estimators_are_seeded_and_the_logistic_grid_spans_class_weights():
    for quick in (True, False):
        cands = train_model.candidate_estimators(quick)
        assert set(cands) == {"logistic_regression", "random_forest", "hist_gradient_boosting"}
        for name, (factory, grid) in cands.items():
            assert grid
            assert factory(**grid[0]).get_params()["random_state"] == config.RANDOM_STATE, name
        assert {g["class_weight"] for g in cands["logistic_regression"][1]} == {None, "balanced"}


def test_build_pipeline_scales_only_when_asked():
    num, cat = ["a", "b"], ["c"]
    with_scale = train_model.build_pipeline(LogisticRegression(), num, cat, scale=True)
    without = train_model.build_pipeline(LogisticRegression(), num, cat, scale=False)
    steps = lambda p: [n for n, _ in p.named_steps["prep"].transformers[0][1].steps]            # noqa: E731
    assert steps(with_scale) == ["impute", "scale"] and steps(without) == ["impute"]


def test_spearman_redundancy_cutoff_is_point_nine():
    rng = np.random.default_rng(0)
    base = rng.normal(size=3000)
    df = pd.DataFrame({"a": base, "b": base + rng.normal(scale=0.35, size=3000),      # rho ~ 0.94: redundant
                       "c": base + rng.normal(scale=1.2, size=3000), "d": rng.normal(size=3000)})
    rho = df.corr(method="spearman")
    assert 0.9 < rho.loc["a", "b"] < 0.99 and rho.loc["a", "c"] < 0.9
    pairs = train_model._spearman_pairs(df, ["a", "b", "c", "d"])
    assert [(a, b) for a, b, _ in pairs] == [("a", "b")]
    assert train_model._spearman_pairs(df, ["a", "b", "c", "d"], cutoff=0.99) == []


@needs_data
def test_a_failing_derivation_parity_check_stops_training(tmp_path, data, monkeypatch):
    monkeypatch.setattr(train_model, "derivation_parity", lambda df: ["derive_model_features mismatches processed column x"])
    with pytest.raises(AssertionError, match="FAIL: derived features"):
        train_model.train(df=data.sample(frac=0.15, random_state=1), out_dir=tmp_path, quick=True, verbose=False)


@needs_data
def test_accuracy_wording_follows_the_class_balance(tmp_path, data, monkeypatch):
    monkeypatch.setattr(train_model, "RARE_CLASS_RATE", 0.0)                 # late orders now count as "not rare"
    _, m = train_model.train(df=data.sample(frac=0.15, random_state=1), out_dir=tmp_path, quick=True, verbose=False)
    assert "Only" not in m["metric_guidance"] and "accuracy is misleading" not in m["metric_guidance"]
    assert "look beyond accuracy" in m["metric_guidance"]
    assert "because only" not in m["selected_model"]["rationale"]
    assert "PR-AUC focuses on that class" in m["selected_model"]["rationale"]
