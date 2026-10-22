"""Inference layer for the delay-risk model (no Streamlit).

The prediction form supplies RAW purchase-time primitives (states, category, price,
freight, weight, purchase date, delivery promise, ...).  :func:`derive_model_features`
turns them into the model features and is the *single* implementation shared by
training-time verification and inference, so the two can never drift apart (a test
proves it reproduces the processed table on real rows).

Public API: :func:`load_model`, :func:`validate_input` / :func:`check_input`, :func:`predict_risk`,
:func:`predict_batch`, :func:`risk_label` / :func:`risk_labels`.  Everything is vectorised; a single order is
just a one-row frame.
"""
from __future__ import annotations

import re
import sys
import warnings
from collections.abc import Mapping
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from . import config

# --------------------------------------------------------------------------- #
# Raw input schema (what the form supplies)
# --------------------------------------------------------------------------- #
REQUIRED_CATEGORICAL = ["customer_state", "seller_state", "product_category"]
REQUIRED_NUMERIC = ["n_items", "total_price", "total_freight"]
#: optional numeric inputs; a missing value is filled from the training-set defaults
OPTIONAL_NUMERIC = [
    "total_weight_g", "estimated_delivery_days", "n_sellers", "total_volume_cm3",
    "max_product_length_cm", "max_product_height_cm", "max_product_width_cm",
]
INTEGER_FIELDS = {"n_items", "n_sellers", "estimated_delivery_days", "purchase_month", "purchase_dow"}
RAW_NUMERIC = REQUIRED_NUMERIC + OPTIONAL_NUMERIC
#: purchase_date is the preferred way to give the time; purchase_month (+ optional dow) is the fallback
TIME_FIELDS = ["purchase_date", "purchase_month", "purchase_dow"]
ACCEPTED_FIELDS = REQUIRED_CATEGORICAL + RAW_NUMERIC + TIME_FIELDS
STATE_PAIR_SEP = "→"

#: Hard plausibility limits (inclusive): physically / data-entry impossible values are errors.  These are NOT
#: derived from data; the TRAINING range (bundle ``form_options["ranges"]``) only triggers a non-blocking warning,
#: because real orders legitimately fall outside the window the model was fitted on.
HARD_LIMITS: dict[str, tuple[float, float]] = {
    "n_items": (1, 1_000), "n_sellers": (1, 1_000), "total_price": (0.01, 1_000_000), "total_freight": (0, 100_000),
    "total_weight_g": (1, 1_000_000), "estimated_delivery_days": (1, 730), "total_volume_cm3": (1, 100_000_000),
    "max_product_length_cm": (1, 1_000), "max_product_height_cm": (1, 1_000), "max_product_width_cm": (1, 1_000),
}
HARD_DATE_RANGE = (pd.Timestamp("2000-01-01"), pd.Timestamp("2100-12-31"))
#: ISO-8601 only: ambiguous dd/mm vs mm/dd strings and bare numbers are rejected, never guessed
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,9})?)?)?$")

#: raw fields feeding each model feature that is not simply a raw field itself
_DERIVED_FROM = {
    "freight_ratio": ["total_freight", "total_price"],
    "same_state": ["seller_state", "customer_state"],
    "purchase_quarter": ["purchase_date", "purchase_month"],
    "purchase_month": ["purchase_date", "purchase_month"],
    "purchase_dow": ["purchase_date", "purchase_dow"],
}
RISK_COLORS = {"Low": config.PALETTE["green"], "Moderate": config.PALETTE["amber"], "High": config.PALETTE["red"]}
TRAIN_COMMAND = f"{config.ROOT}/.venv/bin/python -m src.train_model"


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained model bundle has not been created yet."""


# --------------------------------------------------------------------------- #
# Feature derivation + dtype coercion (shared by training and inference)
# --------------------------------------------------------------------------- #
def derive_model_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Derive the model features that are functions of raw purchase-time inputs.

    Adds ``freight_ratio`` (freight / price, NaN when price <= 0), ``same_state``
    (seller state == customer state), and ``purchase_month`` / ``purchase_dow`` /
    ``purchase_quarter`` (from ``purchase_date`` when present, else from the supplied
    ``purchase_month`` / ``purchase_dow``).  All other columns pass through unchanged.
    """
    out = raw.copy()
    price = pd.to_numeric(out["total_price"], errors="coerce").astype("float64")
    freight = pd.to_numeric(out["total_freight"], errors="coerce").astype("float64")
    out["freight_ratio"] = (freight / price).where(price > 0)

    seller = out["seller_state"].astype("string")
    cust = out["customer_state"].astype("string")
    out["same_state"] = ((seller == cust).fillna(False)).astype("int8")

    if "purchase_date" in out.columns:
        dt = pd.to_datetime(out["purchase_date"], errors="coerce")
    else:
        dt = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    have = dt.notna()
    fb_month = pd.to_numeric(out.get("purchase_month"), errors="coerce") if "purchase_month" in out.columns else np.nan
    fb_dow = pd.to_numeric(out.get("purchase_dow"), errors="coerce") if "purchase_dow" in out.columns else np.nan
    month = dt.dt.month.astype("float64").where(have, fb_month)
    dow = dt.dt.day_of_week.astype("float64").where(have, fb_dow)
    out["purchase_month"] = month.astype("Int8")
    out["purchase_dow"] = dow.astype("Int8")
    out["purchase_quarter"] = ((month - 1) // 3 + 1).astype("Int8")
    return out


def coerce_feature_frame(X: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    """Return exactly the model columns with deterministic dtypes.

    Numeric -> float64 (NA -> NaN); categorical -> object strings with missing as NaN
    (so sklearn imputers/encoders never see pandas-3 string / category dtypes).  A
    missing model column raises rather than being silently dropped.  This is the first
    step of the saved pipeline, so the pipeline accepts raw feature frames of any dtype.
    """
    cols = list(numeric) + list(categorical)
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise ValueError(f"missing model feature column(s): {missing}")
    out = pd.DataFrame(index=X.index)
    for c in numeric:
        out[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    for c in categorical:
        s = X[c].astype("string")
        out[c] = s.astype(object).where(s.notna(), np.nan)
    return out


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
_BUNDLE_KEYS = ["pipeline", "feature_columns", "numeric_features", "categorical_features", "target",
                "model_name", "risk_thresholds", "form_options"]


def load_model(path: str | Path = config.MODEL_PATH) -> dict[str, Any]:
    """Load the model bundle; raise :class:`ModelNotFoundError` with the training command."""
    path = Path(path)
    if not path.exists():
        raise ModelNotFoundError(
            f"Trained model not found at {path}. Create it with:\n    {TRAIN_COMMAND}"
        )
    return _load_cached(str(path), path.stat().st_mtime_ns)


@lru_cache(maxsize=4)
def _load_cached(path: str, _mtime_ns: int) -> dict[str, Any]:
    bundle = joblib.load(path)
    missing = [k for k in _BUNDLE_KEYS if k not in bundle]
    if missing:
        raise ValueError(f"Model bundle {path} is missing keys {missing}; retrain with: {TRAIN_COMMAND}")
    banned = set(bundle["feature_columns"]) & (set(config.LEAKAGE_COLUMNS) | set(config.NON_FEATURE_COLUMNS))
    if banned:
        raise ValueError(f"Model bundle uses forbidden (leakage / non-feature) columns: {sorted(banned)}")
    import sklearn
    saved = bundle.get("sklearn_version")
    if saved and saved != sklearn.__version__:
        warnings.warn(f"Model trained with scikit-learn {saved}, running {sklearn.__version__}; "
                      f"retrain with: {TRAIN_COMMAND}", stacklevel=2)
    return bundle


def _bundle(bundle: dict | None) -> dict:
    return bundle if bundle is not None else load_model()


# --------------------------------------------------------------------------- #
# Risk labels
# --------------------------------------------------------------------------- #
def risk_labels(p: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    """Vectorised ``Low`` / ``Moderate`` / ``High`` labels (a probability equal to a cut takes the higher band)."""
    p = np.asarray(p, dtype=float)
    return np.where(p >= thresholds["high"], "High", np.where(p >= thresholds["moderate"], "Moderate", "Low"))


def risk_label(p: float, thresholds: dict[str, float]) -> str:
    """Map a probability to ``Low`` / ``Moderate`` / ``High`` using the bundle thresholds."""
    p = float(p)
    if not np.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError(f"probability must be a finite number in [0, 1], got {p!r}")
    return str(risk_labels(np.array([p]), thresholds)[0])


# --------------------------------------------------------------------------- #
# Validation (vectorised; one row = one order)
# --------------------------------------------------------------------------- #
def _blank(s: pd.Series) -> pd.Series:
    """True where a value is missing or an empty / whitespace-only string."""
    txt = s.astype("string")
    return s.isna() | (txt.str.strip() == "").fillna(True)


def _unreadable_number(v: Any) -> bool:
    """True for values that must not be read as numbers: booleans and Python ints beyond the float range."""
    if isinstance(v, (bool, np.bool_)):
        return True
    return isinstance(v, int) and abs(v) > sys.float_info.max


def _to_number(src: pd.Series, blank: pd.Series) -> pd.Series:
    """Parse a column to float64 without ever raising: unreadable values (text, booleans, ints too large for a
    float) become NaN and are reported by the caller as 'must be a number'."""
    x = src.where(~blank)
    if x.dtype == object:
        x = x.where(~x.map(_unreadable_number).astype(bool))
    return pd.to_numeric(x, errors="coerce").astype("float64")


def _fmt(x: float) -> str:
    """Number for messages: thousands separators, up to 6 decimals (so 0.009 is not shown as 0.01), no trailing zeros."""
    return f"{x:,.6f}".rstrip("0").rstrip(".")


def _parse_dates(s: pd.Series) -> pd.Series:
    """Strict date parsing: datetime-like objects and ISO ``YYYY-MM-DD[ HH:MM[:SS]]`` strings only.

    Numbers (e.g. 20180710 would otherwise be read as epoch nanoseconds) and ambiguous formats such as
    ``10/07/2018`` become NaT instead of being silently misread.
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        dt = pd.to_datetime(s, errors="coerce")
        return dt.dt.tz_localize(None) if dt.dt.tz is not None else dt
    obj = s.astype(object).to_numpy()
    out = np.full(len(obj), np.datetime64("NaT", "us"), dtype="datetime64[us]")
    is_str = np.array([isinstance(v, str) for v in obj], dtype=bool)
    is_dt = np.array([isinstance(v, (date, np.datetime64)) for v in obj], dtype=bool)
    if is_str.any():
        txt = pd.Series(obj[is_str]).str.strip()
        parsed = pd.to_datetime(txt.where(txt.str.match(_ISO_DATE)), format="ISO8601", errors="coerce")
        out[is_str] = parsed.astype("datetime64[us]").to_numpy()
    if is_dt.any():
        parsed = pd.to_datetime(pd.Series(obj[is_dt]), errors="coerce")
        if parsed.dtype.kind == "M":
            out[is_dt] = (parsed.dt.tz_localize(None) if parsed.dt.tz is not None else parsed).astype(
                "datetime64[us]").to_numpy()
    return pd.Series(out, index=s.index)


def _used_raw_fields(feature_columns: list[str]) -> set[str]:
    used: set[str] = set()
    for f in feature_columns:
        used.update(_DERIVED_FROM.get(f, [f]))
    return used


def _resolve_frame(raw: pd.DataFrame, bundle: dict) -> tuple[pd.DataFrame, list[list[str]], pd.DataFrame, list[list[str]]]:
    """Validate + complete raw inputs.

    Returns ``(features, errors, assumed, warnings)``: the model feature frame (all rows;
    invalid rows are still present but must not be scored), one list of human-readable
    error messages per row (empty list = valid), a boolean frame marking which raw fields
    were filled from documented training defaults, and one list of non-blocking warnings
    per row (value outside the training range / dataset date span: the model extrapolates).
    """
    opts = bundle["form_options"]
    ranges = opts.get("ranges", {})
    defaults = opts.get("defaults", {})
    used = _used_raw_fields(bundle["feature_columns"])
    idx = raw.index
    msgs: list[list[str]] = []
    warns: list[list[str]] = []

    def flag(mask: pd.Series, text: str | pd.Series) -> None:
        msgs.append(np.where(mask, text, "").tolist())

    def warn(mask: pd.Series, text: str | pd.Series) -> None:
        warns.append(np.where(mask, text, "").tolist())

    def column(name: str) -> pd.Series:
        return raw[name] if name in raw.columns else pd.Series(np.nan, index=idx, dtype="object")

    cl = pd.DataFrame(index=idx)
    assumed = pd.DataFrame(False, index=idx, columns=ACCEPTED_FIELDS)

    # categoricals: normalised (strip; states upper-case, category lower-case); never defaulted
    for col in REQUIRED_CATEGORICAL:
        src = column(col)
        blank = _blank(src)
        s = src.astype("string").str.strip()
        s = s.str.lower() if col == "product_category" else s.str.upper()
        valid = set(opts.get(col, []))
        flag(blank, f"{col} is required.")
        flag(~blank & ~s.isin(valid),
             "'" + s.fillna("") + f"' is not a known {col} (choose one of the {len(valid)} values seen in training).")
        cl[col] = s.where(~blank, pd.NA)

    # numerics
    pair = cl["seller_state"].astype("object").fillna("?") + STATE_PAIR_SEP + cl["customer_state"].astype("object").fillna("?")
    for col in RAW_NUMERIC:
        src = column(col)
        blank = _blank(src)
        num = _to_number(src, blank)
        notnum = ~blank & (num.isna() | ~np.isfinite(num))
        flag(notnum, f"{col} must be a finite number.")
        num = num.where(~notnum)
        if col in INTEGER_FIELDS:
            nonint = num.notna() & (num != np.round(num))
            flag(nonint, f"{col} must be a whole number.")
            num = num.where(~nonint)
        if col in HARD_LIMITS:
            lo, hi = HARD_LIMITS[col]
            out_of = num.notna() & ((num < lo) | (num > hi))
            flag(out_of, f"{col} = " + num.map(_fmt) + f" is outside the allowed range ({_fmt(lo)} to {_fmt(hi)}).")
            num = num.where(~out_of)
        if col in ranges and col in used:
            lo, hi = ranges[col]
            beyond = num.notna() & ((num < lo) | (num > hi))
            warn(beyond, f"{col} = " + num.map(_fmt) + f" is outside the range the model was trained on "
                 f"({_fmt(lo)} to {_fmt(hi)}): the model is extrapolating, so treat this estimate with extra caution.")
        if col in REQUIRED_NUMERIC:
            flag(blank, f"{col} is required.")
        elif col in used:
            if col == "estimated_delivery_days":
                fill = pair.map(opts.get("state_pair_median_estimated_days", {})).astype("float64")
                fill = fill.fillna(defaults.get(col, np.nan))
            else:
                fill = pd.Series(defaults.get(col, np.nan), index=idx, dtype="float64")
            assumed[col] = blank
            num = num.where(~blank, fill)
            flag(blank & num.isna(), f"{col} is missing and no training default is available.")
        cl[col] = num

    flag((cl["n_sellers"] > cl["n_items"]).fillna(False), "n_sellers cannot exceed n_items.")

    # purchase time: purchase_date preferred; else purchase_month (required) [+ dow optional]
    d_src = column("purchase_date")
    d_blank = _blank(d_src)
    dt = _parse_dates(d_src.where(~d_blank))
    flag(~d_blank & dt.isna(), "purchase_date could not be read as a date (use YYYY-MM-DD).")
    lo_d, hi_d = HARD_DATE_RANGE
    absurd = dt.notna() & ((dt < lo_d) | (dt >= hi_d + pd.Timedelta(days=1)))
    flag(absurd, f"purchase_date is outside the allowed range ({lo_d:%Y-%m-%d} to {hi_d:%Y-%m-%d}).")
    dt = dt.where(~absurd)
    span = opts.get("date_span")
    if span:
        s0, s1 = pd.Timestamp(span[0]), pd.Timestamp(span[1])
        warn(dt.notna() & ((dt < s0) | (dt >= s1 + pd.Timedelta(days=1))),
             f"purchase_date is outside the dates the model was trained on ({s0:%Y-%m-%d} to {s1:%Y-%m-%d}, its "
             "fitting window); it only learned the day-of-week pattern seen in that period.")
    cl["purchase_date"] = dt
    have_date = dt.notna()
    for col in ("purchase_month", "purchase_dow"):
        src = column(col)
        blank = _blank(src)
        num = _to_number(src, blank)
        lo, hi = (1, 12) if col == "purchase_month" else (0, 6)
        bad = ~blank & (num.isna() | (num < lo) | (num > hi) | (num != np.round(num)))
        hint = " (1 = January)." if lo == 1 else " (0 = Monday)."
        flag(bad, f"{col} must be a whole number from {lo} to {hi}{hint}")
        num = num.where(~bad)
        truth = dt.dt.month if col == "purchase_month" else dt.dt.day_of_week
        flag(have_date & num.notna() & (num != truth.astype("float64")), f"{col} conflicts with purchase_date.")
        if col == "purchase_month":
            flag(~have_date & blank & d_blank, "purchase_date is required (or purchase_month, 1-12).")
        else:
            miss = ~have_date & blank
            assumed[col] = miss & (col in used)
            num = num.where(~miss, defaults.get(col, np.nan))
        cl[col] = num

    feats = derive_model_features(cl)
    # a derived feature can leave its training support although each raw input is inside its own range
    if "freight_ratio" in ranges and "freight_ratio" in bundle["feature_columns"]:
        lo, hi = ranges["freight_ratio"]
        ratio = feats["freight_ratio"].astype("float64")
        warn(ratio.notna() & ((ratio < lo) | (ratio > hi)),
             "freight_ratio (freight / price) = " + ratio.map(_fmt) + f" is outside the range the model was trained on "
             f"({_fmt(lo)} to {_fmt(hi)}): the model is extrapolating, so treat this estimate with extra caution.")
    errors = [[m for m in row if m] for row in zip(*msgs)]
    warnings_ = [[m for m in row if m] for row in zip(*warns)] if warns else [[] for _ in range(len(idx))]
    return feats, errors, assumed, warnings_


def _assumption_records(row_assumed: pd.Series, feats: pd.Series) -> dict[str, dict[str, Any]]:
    """Describe the defaults applied to one row (field -> value + where it came from)."""
    out: dict[str, dict[str, Any]] = {}
    for col in row_assumed.index[row_assumed.values]:
        if col == "estimated_delivery_days":
            src = ("median estimated delivery days of training orders for this seller state -> customer state "
                   "pair, else the overall training median")
        else:
            src = "training-set median"
        val = feats[col] if col in feats.index else np.nan
        out[col] = {"value": None if pd.isna(val) else float(val), "source": src}
    return out


def _resolve_one(inputs: dict[str, Any], bundle: dict) -> tuple[pd.Series, list[str], pd.Series, list[str]]:
    """Resolve a single order dict -> (feature row, error messages, assumed flags, warnings)."""
    if not isinstance(inputs, Mapping):
        raise ValueError("inputs must be a dict of field -> value")
    unknown = sorted(set(inputs) - set(ACCEPTED_FIELDS))
    row = pd.DataFrame([{k: v for k, v in inputs.items() if k in ACCEPTED_FIELDS}], dtype="object")
    feats, errors, assumed, warned = _resolve_frame(row, bundle)
    errs = list(errors[0])
    if unknown:
        errs.append(f"Unrecognised input field(s): {unknown}. Accepted fields: {ACCEPTED_FIELDS}.")
    return feats.iloc[0], errs, assumed.iloc[0], list(warned[0])


def check_input(inputs: dict[str, Any], bundle: dict | None = None) -> tuple[dict[str, Any], list[str], list[str]]:
    """Validate one order: ``(cleaned, errors, warnings)``.

    ``cleaned`` holds the normalised raw values (plus the derived model features) with optional
    fields completed from the training defaults.  ``errors`` (empty = valid) block scoring: a
    missing required field, an unknown state / category, an impossible value (outside
    :data:`HARD_LIMITS`), a non-ISO date or an unrecognised field name.  ``warnings`` never block:
    a value outside the training range / date span means the model is extrapolating.
    """
    feats, errors, _, warned = _resolve_one(inputs, _bundle(bundle))
    cleaned = {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in feats.items()}
    return cleaned, errors, warned


def validate_input(inputs: dict[str, Any], bundle: dict | None = None) -> tuple[dict[str, Any], list[str]]:
    """Validate one order's raw inputs; returns ``(cleaned, errors)`` (see :func:`check_input` for warnings)."""
    cleaned, errors, _ = check_input(inputs, bundle)
    return cleaned, errors


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def _score(pipeline: Any, feats: pd.DataFrame, feature_columns: list[str], floor: float = 0.0) -> np.ndarray:
    """Late probabilities in [0, 1]; scores below ``floor`` (isotonic exact zeros) are raised to it."""
    proba = np.asarray(pipeline.predict_proba(feats[feature_columns]))[:, 1]
    if not np.all(np.isfinite(proba)) or proba.min() < 0 or proba.max() > 1:
        raise ValueError("model produced a probability outside [0, 1]")
    return np.maximum(proba, floor)


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def _test_window_sentence(b: dict, label: str, same_state: bool | None = None) -> str | None:
    """Data-derived caveat from the untouched test window (None for bundles that do not carry one).

    ``same_state`` (seller state == customer state) adds the per-group figures: the pooled band rate can hide a
    large difference between same-state and other-state orders, and the model's calibration differs by group.
    """
    c = b.get("test_window_caveat")
    if not c:
        return None
    text = (f"On the untouched test window ({c['start']} to {c['end']}, n={c['n']:,}, {c['observed_late_rate']:.1%} late) "
            f"the model's mean predicted risk was {c['mean_predicted']:.1%} against {c['observed_late_rate']:.1%} observed")
    if c["brier_not_better_than_baseline"]:
        text += (f", and its Brier score ({c['brier']:.4f}) was not better than a constant base-rate prediction "
                 f"({c['brier_baseline']:.4f}), so read the probability as a rough ranking, not an exact chance")
    else:
        text += f"; its Brier score ({c['brier']:.4f}) beat a constant base-rate prediction ({c['brier_baseline']:.4f})"
    band = (c.get("band_late_rates") or {}).get(label)
    if band and band.get("n"):
        text += f". Test orders in the {label} band were late {band['late_rate']:.1%} of the time (n={band['n']:,})"
        groups = c.get("by_same_state") or {}
        if groups:
            parts = []
            for key, name in (("same_state", "same-state"), ("other_state", "other-state")):
                cell = ((groups.get(key) or {}).get("bands") or {}).get(label) or {}
                parts.append(f"{name} orders {_pct(cell.get('late_rate'))} (n={cell.get('n', 0):,})")
            text += " (" + "; ".join(parts) + ")"
    text += "."
    g = (c.get("by_same_state") or {}).get("same_state" if same_state else "other_state") if same_state is not None else None
    if g and g.get("n"):
        text += (f" For {'same-state' if same_state else 'other-state'} orders like this one the mean predicted risk was "
                 f"{_pct(g['mean_predicted'])} against {_pct(g['observed_late_rate'])} observed (n={g['n']:,}).")
    if c.get("test_performance_degraded"):
        text += " Performance degraded on this later window, so expect weaker results than on validation."
    return text


def _probability_note(b: dict, p: float, label: str | None = None, same_state: bool | None = None) -> str | None:
    """Plain-language caveats on the probability: step-function / floored scores and the test-window finding."""
    parts: list[str] = []
    res = b.get("score_resolution")
    if b.get("calibrated") and res:
        note = (f"Calibrated scores take only {res['distinct_levels']} distinct values (a step function, highest "
                f"{res['max']:.1%}), so nearby inputs often get the same probability; read it as a band, not a precise "
                "figure.")
        floor = float(b.get("probability_floor", 0.0))
        if floor > 0 and p <= floor:
            note += f" The lowest level is shown as {floor:.2%} rather than 0% because no order is certain to arrive on time."
        parts.append(note)
    test = _test_window_sentence(b, label, same_state) if label else None
    if test:
        parts.append(test)
    return " ".join(parts) or None


def predict_risk(inputs: dict[str, Any], bundle: dict | None = None) -> dict[str, Any]:
    """Predict the late-delivery probability for ONE order from raw purchase-time inputs.

    Returns ``probability, risk_label, risk_color, thresholds, model_name, features_used``
    plus ``assumptions_used`` (fields filled from training defaults), ``derived_features``,
    ``decision_threshold``, ``calibrated``, ``warnings`` (non-blocking: the input lies outside the
    training range, so the model extrapolates), ``probability_note`` (step-function caveat plus the test-window
    finding, or None) and ``test_window_caveat`` (the structured test-window figures the note is built from, or None).
    Raises ``ValueError`` listing every problem when the input is invalid (or when ``inputs`` is not a dict).
    """
    b = _bundle(bundle)
    feats, errors, assumed, warned = _resolve_one(inputs, b)
    if errors:
        raise ValueError("Invalid input: " + " | ".join(errors))
    frame = feats.to_frame().T.infer_objects()
    p = float(_score(b["pipeline"], frame, b["feature_columns"], float(b.get("probability_floor", 0.0)))[0])
    label = risk_label(p, b["risk_thresholds"])
    derived = {c: (None if pd.isna(feats[c]) else float(feats[c])) for c in b["numeric_features"]}
    derived.update({c: str(feats[c]) for c in b["categorical_features"]})
    return {
        "probability": p,
        "risk_label": label,
        "risk_color": RISK_COLORS[label],
        "thresholds": dict(b["risk_thresholds"]),
        "model_name": b["model_name"],
        "features_used": list(b["feature_columns"]),
        "assumptions_used": _assumption_records(assumed, feats),
        "derived_features": derived,
        "decision_threshold": b.get("decision_threshold"),
        "calibrated": bool(b.get("calibrated", False)),
        "warnings": warned,
        "probability_note": _probability_note(b, p, label, bool(feats["same_state"])),
        "test_window_caveat": b.get("test_window_caveat"),
    }


def predict_batch(df: pd.DataFrame, bundle: dict | None = None) -> pd.DataFrame:
    """Score many orders (raw-input columns as in :func:`predict_risk`).

    Returns a copy of ``df`` with ``late_probability``, ``risk_label``, ``assumptions_used``
    (``;``-joined filled fields), ``input_warning`` (non-blocking extrapolation warnings) and
    ``input_error``.  Rows that fail validation get a NaN
    probability and the reason in ``input_error`` (never a silent default); a frame that is
    missing a required *column* raises ``ValueError``.
    """
    b = _bundle(bundle)
    missing_cols = [c for c in REQUIRED_CATEGORICAL + REQUIRED_NUMERIC if c not in df.columns]
    if "purchase_date" not in df.columns and "purchase_month" not in df.columns:
        missing_cols.append("purchase_date (or purchase_month)")
    if missing_cols:
        raise ValueError(f"Batch input is missing required column(s): {missing_cols}")
    out = df.copy()
    if df.empty:
        for c, dt in [("late_probability", "float64"), ("risk_label", "object"),
                      ("assumptions_used", "object"), ("input_warning", "object"), ("input_error", "object")]:
            out[c] = pd.Series(dtype=dt)
        return out
    raw = df[[c for c in ACCEPTED_FIELDS if c in df.columns]].reset_index(drop=True).astype("object")
    feats, err, assumed, warned = _resolve_frame(raw, b)
    ok = np.array([not e for e in err])
    prob = np.full(len(df), np.nan)
    if ok.any():
        prob[ok] = _score(b["pipeline"], feats.loc[ok], b["feature_columns"], float(b.get("probability_floor", 0.0)))
    labels = np.full(len(df), "", dtype=object)
    labels[ok] = risk_labels(prob[ok], b["risk_thresholds"])
    out["late_probability"] = prob
    out["risk_label"] = pd.Series(labels, index=out.index).replace("", np.nan)
    out["assumptions_used"] = pd.Series(
        [";".join(assumed.columns[r]) for r in assumed.to_numpy()], index=out.index)
    out["input_warning"] = pd.Series([" | ".join(w) for w in warned], index=out.index)
    out["input_error"] = pd.Series([" | ".join(e) for e in err], index=out.index)
    return out
