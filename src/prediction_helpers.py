"""Pure helpers behind the Delay Prediction page (no Streamlit).

Everything the page *displays* about the model or about a prediction is computed here from
``models/model_metrics.json``, the model bundle and the processed data, so that the numbers can be
unit-tested without a browser.  Nothing in this module is hardcoded: rates, windows and AUCs come
from the metrics file; defaults come from the training data.

Contents: field labels + plain-language validation messages, form defaults, risk-band tables, the
limitations / interpretation text, a delivery-promise what-if, and the CSV batch-scoring helpers
(template, parsing, scoring, invalid-row report).
"""
from __future__ import annotations

import csv
import io
import math
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

from . import analytics, charts, prediction

# --------------------------------------------------------------------------- #
# Field labels and plain-language validation messages
# --------------------------------------------------------------------------- #
FIELD_LABELS: dict[str, str] = {
    "customer_state": "Customer state",
    "seller_state": "Seller state",
    "product_category": "Product category",
    "purchase_date": "Purchase date",
    "n_items": "Number of items",
    "total_price": "Total product value",
    "total_freight": "Total freight value",
    "total_weight_g": "Total product weight",
    "estimated_delivery_days": "Delivery promise",
    "n_sellers": "Number of sellers",
    "total_volume_cm3": "Package volume",
    "max_product_length_cm": "Longest product length",
    "max_product_height_cm": "Tallest product height",
    "max_product_width_cm": "Widest product width",
    "purchase_month": "Purchase month",
    "purchase_dow": "Purchase weekday",
}
_FIELD_RE = re.compile(r"\b(" + "|".join(sorted(FIELD_LABELS, key=len, reverse=True)) + r")\b")


def humanise_error(message: str) -> tuple[str | None, str]:
    """Turn a ``prediction`` validation message into ``(field, plain-language text)``.

    ``field`` is the first raw field name the message mentions (None when it names none, e.g. an
    unrecognised-field error).  Field identifiers are replaced by their labels
    (``total_price is required.`` -> ``Total product value is required.``).
    """
    msg = str(message).strip()
    first = _FIELD_RE.search(msg)
    field = first.group(1) if first else None

    def repl(m: re.Match) -> str:
        label = FIELD_LABELS[m.group(1)]
        if m.start() == 0:
            return label
        return label[:1].lower() + label[1:]

    text = _FIELD_RE.sub(repl, msg)
    return field, text


def field_errors(errors: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    """Split validation errors into ``({field: [messages]}, [messages naming no field])``."""
    by_field: dict[str, list[str]] = {}
    general: list[str] = []
    for e in errors:
        f, t = humanise_error(e)
        if f is None:
            general.append(t)
        else:
            by_field.setdefault(f, []).append(t)
    return by_field, general


# --------------------------------------------------------------------------- #
# Small accessors over the metrics JSON / bundle (all tolerate missing keys)
# --------------------------------------------------------------------------- #
def selected_name(bundle: Mapping | None, metrics: Mapping | None) -> str | None:
    sel = ((metrics or {}).get("selected_model") or {}).get("name")
    return sel or (bundle or {}).get("model_name")


def selected_entry(bundle: Mapping | None, metrics: Mapping | None) -> Mapping:
    name = selected_name(bundle, metrics)
    return ((metrics or {}).get("models") or {}).get(name) or {}


def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def pct(x: Any, digits: int = 1) -> str:
    v = _num(x)
    return "n/a" if v is None else f"{100 * v:.{digits}f}%"


def model_label(bundle: Mapping | None, metrics: Mapping | None) -> str:
    entry = selected_entry(bundle, metrics)
    name = selected_name(bundle, metrics) or "model"
    return str(entry.get("label") or str(name).replace("_", " ").capitalize())


def window_rates(metrics: Mapping | None) -> dict[str, dict]:
    """``{'train'|'validation'|'test': {late_rate, n, start, end}}`` from ``metrics['split']``."""
    split = (metrics or {}).get("split") or {}
    out = {}
    for w in ("train", "validation", "test"):
        d = split.get(w)
        if isinstance(d, Mapping) and _num(d.get("late_rate")) is not None:
            out[w] = {"late_rate": float(d["late_rate"]), "n": int(d.get("n") or 0),
                      "n_late": int(d.get("n_late") or 0), "start": d.get("start"), "end": d.get("end")}
    return out


def model_info(bundle: Mapping, metrics: Mapping | None) -> dict[str, Any]:
    """Headline facts about the model in use (None where the metrics file lacks them)."""
    entry = selected_entry(bundle, metrics)
    val, test = entry.get("validation") or {}, entry.get("test") or {}
    ci = (entry.get("test_ci95") or {})
    wins = window_rates(metrics)
    tw = bundle.get("train_window") or {}
    method = entry.get("calibration_method") if entry else None
    return {
        "name": model_label(bundle, metrics),
        "calibrated": bool(bundle.get("calibrated", False)),
        "calibration_method": method,
        "train_start": tw.get("start") or (wins.get("train") or {}).get("start"),
        "train_end": tw.get("end") or (wins.get("train") or {}).get("end"),
        "train_n": tw.get("n") or (wins.get("train") or {}).get("n"),
        "test_start": (wins.get("test") or {}).get("start"), "test_end": (wins.get("test") or {}).get("end"),
        "test_n": test.get("n") or (wins.get("test") or {}).get("n"),
        "test_roc_auc": _num(test.get("roc_auc")), "test_roc_ci": ci.get("roc_auc"),
        "test_pr_auc": _num(test.get("pr_auc")), "test_pr_ci": ci.get("pr_auc"),
        "test_baseline": _num(test.get("positive_rate")),
        "test_lift": (_num(test.get("pr_auc")) / _num(test.get("positive_rate"))
                      if _num(test.get("pr_auc")) is not None and _num(test.get("positive_rate")) else None),
        "val_roc_auc": _num(val.get("roc_auc")), "val_pr_auc": _num(val.get("pr_auc")),
        "val_baseline": _num(val.get("positive_rate")),
        "windows": wins,
        "degraded": bool((metrics or {}).get("selected_model", {}).get("test_performance_degraded")
                         or entry.get("test_performance_degraded")),
    }


def top_features(metrics: Mapping | None, window: str = "validation", k: int = 3) -> list[str]:
    pi = (metrics or {}).get("permutation_importance") or {}
    items = pi.get("test_items") if window == "test" else pi.get("items")
    if not isinstance(items, list):
        return []
    ok = [i for i in items if _num(i.get("importance_mean")) is not None]
    ok.sort(key=lambda i: -float(i["importance_mean"]))
    return [str(i["feature"]) for i in ok[:k]]


# --------------------------------------------------------------------------- #
# Form defaults
# --------------------------------------------------------------------------- #
def _mode(s: pd.Series, allowed: list[str]) -> str | None:
    if s is None or len(s) == 0:
        return allowed[0] if allowed else None
    vc = s.dropna().astype(str).value_counts()
    for v in vc.index:
        if v in allowed:
            return str(v)
    return allowed[0] if allowed else None


def form_defaults(df_all: pd.DataFrame | None, bundle: Mapping) -> dict[str, Any]:
    """Prefilled form values: most common states / category and median order values in the data.

    Computed from the processed orders (not hardcoded); fall back to the first option / None when
    the frame is empty.
    """
    opts = bundle["form_options"]
    d = df_all if df_all is not None else pd.DataFrame()
    out: dict[str, Any] = {}
    for col in prediction.REQUIRED_CATEGORICAL:
        out[col] = _mode(d[col] if col in d.columns else None, list(opts.get(col, [])))

    def med(col: str) -> float | None:
        if col not in d.columns or d[col].dropna().empty:
            return None
        return float(pd.to_numeric(d[col], errors="coerce").median())

    n_items, price, freight = med("n_items"), med("total_price"), med("total_freight")
    out["n_items"] = max(1, int(round(n_items))) if n_items is not None else 1
    out["total_price"] = round(price, 2) if price is not None else None
    out["total_freight"] = round(freight, 2) if freight is not None else None
    span = opts.get("date_span") or opts.get("dataset_date_span")
    out["purchase_date"] = pd.Timestamp(span[1]).date() if span else None
    return out


def pair_promise(bundle: Mapping, seller_state: str, customer_state: str) -> tuple[int, str]:
    """Prefilled delivery promise (days) for a route and where it came from (``pair`` | ``overall``)."""
    opts = bundle["form_options"]
    key = f"{seller_state}{prediction.STATE_PAIR_SEP}{customer_state}"
    table = opts.get("state_pair_median_estimated_days") or {}
    if key in table and _num(table[key]) is not None:
        return int(round(float(table[key]))), "pair"
    return int(round(float((opts.get("defaults") or {}).get("estimated_delivery_days", 24.0)))), "overall"


# --------------------------------------------------------------------------- #
# Risk bands
# --------------------------------------------------------------------------- #
def band_ranges(thresholds: Mapping[str, float]) -> dict[str, str]:
    mod, high = float(thresholds["moderate"]), float(thresholds["high"])
    return {"Low": f"below {mod:.1%}", "Moderate": f"{mod:.1%} to below {high:.1%}", "High": f"{high:.1%} and above"}


def _band_rows(metrics: Mapping | None, window: str) -> dict[str, Mapping]:
    rb = (metrics or {}).get("risk_bands") or {}
    rows = rb.get(window)
    return {r["band"]: r for r in rows} if isinstance(rows, list) else {}


def band_table(thresholds: Mapping[str, float], metrics: Mapping | None, current: str | None = None) -> pd.DataFrame:
    """One row per band: probability range and observed late rate on the validation and test windows.

    Rates are fractions; ``Test 95% CI`` is a ready-made text.  ``This order`` marks ``current``.
    """
    val, test = _band_rows(metrics, "validation"), _band_rows(metrics, "test")
    ranges = band_ranges(thresholds)
    recs = []
    for band in ("Low", "Moderate", "High"):
        v, t = val.get(band, {}), test.get(band, {})
        lo, hi = _num(t.get("late_ci_low")), _num(t.get("late_ci_high"))
        recs.append({
            "Risk band": band, "Probability range": ranges[band],
            "Validation late rate": _num(v.get("late_rate")), "Validation n": v.get("n"),
            "Test late rate": _num(t.get("late_rate")),
            "Test 95% CI": f"{pct(lo)} to {pct(hi)}" if lo is not None and hi is not None else "n/a",
            "Test n": t.get("n"),
            "This order": "◀ this order" if current == band else "",
        })
    return pd.DataFrame(recs)


def band_separation_text(metrics: Mapping | None) -> str | None:
    """Plain statement of how well the bands separate on the test window (from the stored diagnostics)."""
    test = _band_rows(metrics, "test")
    if not all(b in test for b in ("Low", "Moderate", "High")):
        return None
    diag = ((metrics or {}).get("risk_bands") or {}).get("test_diagnostics") or {}
    ov = diag.get("adjacent_intervals_overlap") or {}
    low, mod, high = (test[b] for b in ("Low", "Moderate", "High"))
    txt = (f"On the test window the observed late rate was {pct(low['late_rate'])} in the Low band (n={int(low['n']):,}), "
           f"{pct(mod['late_rate'])} in Moderate (n={int(mod['n']):,}) and {pct(high['late_rate'])} in High (n={int(high['n']):,}).")
    if ov.get("Low-Moderate") is False:
        txt += " Low is clearly lower than Moderate (the 95% intervals do not overlap)."
    elif ov.get("Low-Moderate"):
        txt += " The Low and Moderate bands are not clearly separated (their 95% intervals overlap)."
    if ov.get("Moderate-High"):
        txt += " The Moderate and High bands are barely separated: their 95% intervals overlap, so do not read High as clearly worse than Moderate."
    elif ov.get("Moderate-High") is False:
        txt += " Moderate and High are separated (non-overlapping 95% intervals)."
    return txt


# --------------------------------------------------------------------------- #
# Limitations and interpretation (all numbers computed)
# --------------------------------------------------------------------------- #
def limitations(bundle: Mapping, metrics: Mapping | None, *, include_calibration: bool = True,
                include_bands: bool = True) -> list[str]:
    """The limitations callout lines, built from the stored evaluation."""
    mi = model_info(bundle, metrics)
    lines: list[str] = []
    if mi["test_roc_auc"] is not None and mi["test_pr_auc"] is not None:
        ci = mi["test_roc_ci"]
        ci_txt = f" (95% CI {ci[0]:.2f} to {ci[1]:.2f})" if isinstance(ci, (list, tuple)) and len(ci) == 2 else ""
        base = f" against {mi['test_baseline']:.3f} for a no-skill model" if mi["test_baseline"] is not None else ""
        lift = f", about {mi['test_lift']:.1f}x better than chance" if mi["test_lift"] else ""
        lines.append(
            f"**Modest signal.** On the untouched test window ({mi['test_start']} to {mi['test_end']}, n = {int(mi['test_n'] or 0):,}) "
            f"ROC-AUC was {mi['test_roc_auc']:.2f}{ci_txt}, where 0.5 means no skill, and PR-AUC {mi['test_pr_auc']:.3f}{base}{lift}. "
            "It ranks orders only somewhat better than chance; it cannot say whether a particular order will be late.")
        if mi["val_roc_auc"] is not None:
            lines.append(f"**Weaker on later data.** On the validation window (also used to tune the model) ROC-AUC was "
                         f"{mi['val_roc_auc']:.2f}; on the later test window it fell to {mi['test_roc_auc']:.2f}.")
    w = mi["windows"]
    if all(k in w for k in ("train", "validation", "test")):
        lines.append(f"**The late rate shifted over time:** {pct(w['train']['late_rate'])} in the training window, "
                     f"{pct(w['validation']['late_rate'])} in validation and {pct(w['test']['late_rate'])} in test, so a model fitted on "
                     "one period does not transfer cleanly to the next.")
    cav = bundle.get("test_window_caveat") or {}
    if include_calibration and cav:
        txt = (f"**Probabilities are rough.** On the test window the mean predicted risk was {pct(cav.get('mean_predicted'))} "
               f"against {pct(cav.get('observed_late_rate'))} observed")
        if cav.get("brier_not_better_than_baseline"):
            txt += (f", and the Brier score ({cav['brier']:.4f}) was not better than always predicting the base rate "
                    f"({cav['brier_baseline']:.4f})")
        lines.append(txt + ". Read the probability as a rough ranking, not an exact chance.")
    sep = band_separation_text(metrics)
    if include_bands and sep and "barely separated" in sep:
        lines.append("**Bands overlap.** " + sep.split("The Moderate and High")[0].strip()
                     + " The Moderate and High bands are barely separated on the test window.")
    lines.append("**Historical data only.** The model learned from Olist orders placed in 2016-2018; it says nothing certain about "
                 "today's carriers, routes or customers. It finds associations in that data, not causes, and a prediction is "
                 "never a guarantee for any single order.")
    return lines


def historical_context(df_all: pd.DataFrame | None, seller_state: str, customer_state: str, category: str,
                       min_orders: int = 100) -> dict[str, Any]:
    """Observed late rates (with n and Wilson interval) for the route, the category and all orders."""
    def block(d: pd.DataFrame | None) -> dict[str, Any]:
        k = analytics.kpi_summary(d if d is not None else pd.DataFrame())
        n = k["n_orders"]
        lo, hi = analytics.wilson_ci(k["n_late"], n) if n else (np.nan, np.nan)
        return {"n": n, "late_rate": k["late_rate"], "n_late": k["n_late"], "ci_low": lo, "ci_high": hi,
                "median_duration": k["median_duration"], "small": n < min_orders}
    if df_all is None or df_all.empty:
        empty = block(None)
        return {"route": empty, "category": empty, "overall": empty, "min_orders": min_orders}
    route = df_all[(df_all["seller_state"].astype(str) == seller_state) & (df_all["customer_state"].astype(str) == customer_state)]
    cat = df_all[df_all["product_category"].astype(str) == category]
    return {"route": block(route), "category": block(cat), "overall": block(df_all), "min_orders": min_orders}


def what_if_promise(inputs: Mapping[str, Any], bundle: Mapping, current_days: float,
                    extra_days: tuple[int, ...] = (0, 3, 7, 14)) -> pd.DataFrame:
    """Re-score one order with a longer delivery promise (model sensitivity, an association -- not a cause).

    Skips values that would fall outside the promise range seen in training (the model would be
    extrapolating).  Columns: ``extra_days``, ``promise_days``, ``probability``, ``risk_label``.
    """
    recs = []
    for extra in extra_days:
        trial = dict(inputs)
        trial["estimated_delivery_days"] = int(round(current_days)) + int(extra)
        try:
            res = prediction.predict_risk(trial, dict(bundle))
        except ValueError:
            continue
        if any("estimated_delivery_days" in w for w in res["warnings"]):
            continue
        recs.append({"extra_days": extra, "promise_days": trial["estimated_delivery_days"],
                     "probability": res["probability"], "risk_label": res["risk_label"]})
    return pd.DataFrame(recs, columns=["extra_days", "promise_days", "probability", "risk_label"])


def interpretation(result: Mapping[str, Any], bundle: Mapping, metrics: Mapping | None, history: Mapping | None,
                   route: tuple[str, str] | None = None, what_if: pd.DataFrame | None = None) -> list[dict[str, str]]:
    """Labelled callouts for a prediction: model prediction, observed history, association, suggested action.

    Each item: ``{"tag", "kind", "title", "body"}``.  Every number comes from the result, the metrics
    file or the historical data.
    """
    p, label = float(result["probability"]), str(result["risk_label"])
    wins = window_rates(metrics)
    out: list[dict[str, str]] = []

    # ---- model prediction: compare with base rates and with the band's history
    parts = [f"The model estimates a **{p:.1%}** chance that an order like this arrives after its promised date, which puts it in the "
             f"**{label}** risk band."]
    t = wins.get("test")
    if t and t["late_rate"] > 0:
        ratio = p / t["late_rate"]
        names = {"train": "training", "validation": "validation", "test": "test"}
        base_txt = ", ".join(f"{pct(wins[w]['late_rate'])} in the {names[w]} window" for w in ("train", "validation", "test") if w in wins)
        rel = "about the same as" if 0.9 <= ratio <= 1.1 else f"{ratio:.1f}x"
        parts.append(f"For comparison, the share of late orders was {base_txt}. This estimate is {rel} the test-window rate "
                     f"of {pct(t['late_rate'])}.")
    bands_val, bands_test = _band_rows(metrics, "validation"), _band_rows(metrics, "test")
    if label in bands_val and label in bands_test:
        v, tt = bands_val[label], bands_test[label]
        parts.append(f"Historically, orders the model put in the {label} band were late {pct(v['late_rate'])} of the time on the validation "
                     f"window (n = {int(v['n']):,}) and {pct(tt['late_rate'])} on the later test window (n = {int(tt['n']):,}).")
    out.append({"tag": "Model prediction", "kind": "info", "title": "Model prediction", "body": " ".join(parts)})

    # ---- observed history of the route / category
    if history:
        r, c, o = history["route"], history["category"], history["overall"]
        seller, customer = route or ("seller state", "customer state")
        lines = []
        if r["n"]:
            lines.append(f"Orders shipped {seller} to {customer} were late {pct(r['late_rate'])} of the time "
                         f"(95% CI {pct(r['ci_low'])} to {pct(r['ci_high'])}, n = {r['n']:,}; all orders {pct(o['late_rate'])}, n = {o['n']:,}).")
            if r["small"]:
                lines.append(f"Small sample (n = {r['n']:,}, below {history['min_orders']:,}): treat this route rate as indicative only.")
        else:
            lines.append(f"There are no historical delivered orders on the route {seller} to {customer}, so no observed rate is available.")
        if c["n"]:
            lines.append(f"The product category was late {pct(c['late_rate'])} of the time (n = {c['n']:,}).")
        out.append({"tag": "Observed", "kind": "info", "title": "Observed in the historical data", "body": " ".join(lines)})

    # ---- association: promise sensitivity
    if what_if is not None and len(what_if) > 1:
        first, last = what_if.iloc[0], what_if.iloc[-1]
        diff = float(last["probability"]) - float(first["probability"])
        direction = "lower" if diff < -1e-9 else ("higher" if diff > 1e-9 else "unchanged")
        body = (f"Keeping everything else fixed, the model's estimate goes from {first['probability']:.1%} at a {int(first['promise_days'])}-day "
                f"promise to {last['probability']:.1%} at {int(last['promise_days'])} days ({direction}). ")
        body += ("The delivery promise is the strongest predictor in this model. " if top_features(metrics, "validation", 1) == ["estimated_delivery_days"] else "")
        body += ("This is how the model associates promises with late deliveries in past data; changing a promise does not by itself make "
                 "a delivery on time, and the model has no information about why promises were set that way.")
        out.append({"tag": "Statistical association", "kind": "info", "title": "Statistical association: the delivery promise", "body": body})

    # ---- suggested actions
    tb = bands_test.get(label)
    if label == "Low":
        act = ("Routine monitoring is enough on this signal alone. A Low band is not a guarantee: "
               + (f"Low-band orders were still late {pct(tb['late_rate'])} of the time on the test window (n = {int(tb['n']):,})." if tb else "some Low-band orders are late."))
    else:
        act = ("Worth a closer look rather than an alarm. Things an operations manager could investigate: the seller handover time and "
               "carrier capacity on this route, whether the delivery promise is realistic for it, and proactive status updates to the customer. ")
        if tb:
            act += f"On the test window {pct(tb['late_rate'])} of {label}-band orders were late (n = {int(tb['n']):,}); "
        sep = band_separation_text(metrics)
        act += ("Moderate and High are barely distinguishable there, so use the band as a rough triage, not a precise ranking." if sep and "barely separated" in sep
                else "use the band as a rough triage, not a precise ranking.")
    out.append({"tag": "Suggested actions", "kind": "success" if label == "Low" else "watch", "title": "Suggested actions", "body": act})
    return out


# --------------------------------------------------------------------------- #
# Batch scoring (CSV)
# --------------------------------------------------------------------------- #
BATCH_ID_COL = "order_ref"
BATCH_REQUIRED = ["customer_state", "seller_state", "product_category", "n_items", "total_price", "total_freight", "purchase_date"]
BATCH_OPTIONAL = ["estimated_delivery_days", "total_weight_g", "n_sellers", "total_volume_cm3",
                  "max_product_length_cm", "max_product_height_cm", "max_product_width_cm"]
BATCH_COLUMNS = [BATCH_ID_COL] + BATCH_REQUIRED + BATCH_OPTIONAL
MAX_BATCH_ROWS = 20_000
MAX_BATCH_BYTES = 5 * 1024 * 1024


class BatchFileError(ValueError):
    """The uploaded CSV cannot be used at all (message is shown to the user as is)."""


def build_template(df_all: pd.DataFrame | None, bundle: Mapping, n: int = 3) -> pd.DataFrame:
    """Template with the exact headers and ``n`` EXAMPLE rows built from the most common routes in the data."""
    opts = bundle["form_options"]
    defaults = form_defaults(df_all, bundle)
    d = df_all if df_all is not None else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    if len(d) and {"seller_state", "customer_state"} <= set(d.columns):
        pairs = d.groupby([d["seller_state"].astype(str), d["customer_state"].astype(str)], observed=True).size().sort_values(ascending=False)
        for (s, c), _ in pairs.head(n).items():
            sub = d[(d["seller_state"].astype(str) == s) & (d["customer_state"].astype(str) == c)]
            cats = [x for x in sub["product_category"].astype(str).value_counts().index if x in opts["product_category"]]
            rows.append({
                "customer_state": c, "seller_state": s, "product_category": cats[0] if cats else defaults["product_category"],
                "n_items": 1, "total_price": round(float(sub["total_price"].median()), 2),
                "total_freight": round(float(sub["total_freight"].median()), 2),
                "purchase_date": str(defaults["purchase_date"]) if defaults["purchase_date"] else "",
                "estimated_delivery_days": pair_promise(bundle, s, c)[0],
                "total_weight_g": round(float(sub["total_weight_g"].median()), 0) if sub["total_weight_g"].notna().any() else None,
            })
    while len(rows) < n:
        rows.append({"customer_state": defaults["customer_state"], "seller_state": defaults["seller_state"],
                     "product_category": defaults["product_category"], "n_items": 1, "total_price": defaults["total_price"],
                     "total_freight": defaults["total_freight"],
                     "purchase_date": str(defaults["purchase_date"]) if defaults["purchase_date"] else ""})
    out = pd.DataFrame(rows).reindex(columns=BATCH_COLUMNS)
    out[BATCH_ID_COL] = [f"EXAMPLE-{i + 1}" for i in range(len(out))]
    return out


def template_csv(df_all: pd.DataFrame | None, bundle: Mapping) -> bytes:
    return build_template(df_all, bundle).to_csv(index=False).encode("utf-8")


def _norm_header(h: str) -> str:
    return re.sub(r"[\s\-]+", "_", str(h).strip().lower())


def parse_batch_csv(data: bytes) -> tuple[pd.DataFrame, list[str]]:
    """Parse an uploaded CSV into a frame of the documented columns (all values kept as text).

    Returns ``(frame, notes)``; ``notes`` lists ignored columns.  Raises :class:`BatchFileError`
    with a plain-language message when the file is empty, too large, unreadable or lacks required
    columns.  Row-level problems are NOT raised here -- ``prediction.predict_batch`` reports them
    per row.
    """
    if data is None or len(data) == 0:
        raise BatchFileError("The file is empty.")
    if len(data) > MAX_BATCH_BYTES:
        raise BatchFileError(f"The file is {len(data) / 1e6:.1f} MB; the limit is {MAX_BATCH_BYTES / 1e6:.0f} MB. Split it into smaller files.")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    first = text.splitlines()[0] if text.strip() else ""
    sep = ";" if first.count(";") > first.count(",") else ","
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, skip_blank_lines=True, keep_default_na=True)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        raise BatchFileError(f"The file could not be read as a CSV table ({exc}). Check that every row has the same number of columns.") from exc
    # pandas renames repeated headers (n_items -> n_items.1), so look for duplicates in the raw header line
    raw_header = [_norm_header(c) for c in next(csv.reader(io.StringIO(text), delimiter=sep), [])]
    dups = sorted({c for c in raw_header if raw_header.count(c) > 1})
    if dups:
        raise BatchFileError("The file has duplicate column names: " + ", ".join(dups) + ".")
    df.columns = [_norm_header(c) for c in df.columns]
    missing = [c for c in BATCH_REQUIRED if c not in df.columns]
    if missing:
        raise BatchFileError("Required column(s) missing: " + ", ".join(missing) + ". Download the template to see the exact headers.")
    if len(df) == 0:
        raise BatchFileError("The file has a header but no rows.")
    if len(df) > MAX_BATCH_ROWS:
        raise BatchFileError(f"The file has {len(df):,} rows; the limit is {MAX_BATCH_ROWS:,}. Split it into smaller files.")
    ignored = [c for c in df.columns if c not in BATCH_COLUMNS]
    notes = []
    if ignored:
        notes.append("Ignored column(s) not used by the model and not exported: " + ", ".join(ignored) + ".")
    keep = [c for c in BATCH_COLUMNS if c in df.columns]
    out = df[keep].copy()
    for c in out.columns:
        out[c] = out[c].str.strip()
    return out.reset_index(drop=True), notes


def score_batch(df: pd.DataFrame, bundle: Mapping) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score every valid row; invalid rows keep an empty probability and their reasons.

    Returns ``(scored, summary)``.  ``scored`` has ``csv_row`` (line number in the file, header = 1),
    the input columns, ``late_probability`` (fraction, 4 decimals), ``risk_label``, ``assumptions_used``,
    ``input_warning`` and ``input_error``.
    """
    scored = prediction.predict_batch(df, dict(bundle))
    scored.insert(0, "csv_row", np.arange(len(scored)) + 2)
    scored["late_probability"] = scored["late_probability"].round(4)
    ok = scored["input_error"].fillna("").eq("")
    # rows that were not scored carry no assumptions / warnings (nothing was assumed or extrapolated for them)
    for col in ("assumptions_used", "input_warning"):
        scored[col] = scored[col].fillna("").where(ok, "")
    bands = scored.loc[ok, "risk_label"].value_counts()
    summary = {
        "n_rows": int(len(scored)), "n_valid": int(ok.sum()), "n_invalid": int((~ok).sum()),
        "bands": {b: int(bands.get(b, 0)) for b in ("Low", "Moderate", "High")},
        "mean_probability": float(scored.loc[ok, "late_probability"].mean()) if ok.any() else float("nan"),
        "n_warnings": int((ok & scored["input_warning"].ne("")).sum()),
        "n_assumed": int((ok & scored["assumptions_used"].ne("")).sum()),
    }
    return scored, summary


def invalid_rows_report(scored: pd.DataFrame) -> pd.DataFrame:
    """One line per problem for rows that could not be scored: ``csv_row``, ``order_ref`` (if any), ``problem``."""
    if scored is None or scored.empty or "input_error" not in scored.columns:
        return pd.DataFrame(columns=["csv_row", "problem"])
    bad = scored[scored["input_error"].fillna("") != ""]
    recs = []
    for _, r in bad.iterrows():
        for e in str(r["input_error"]).split(" | "):
            rec = {"csv_row": int(r["csv_row"]), "problem": humanise_error(e)[1]}
            if BATCH_ID_COL in bad.columns:
                rec[BATCH_ID_COL] = r[BATCH_ID_COL]
            recs.append(rec)
    cols = ["csv_row"] + ([BATCH_ID_COL] if BATCH_ID_COL in bad.columns else []) + ["problem"]
    return pd.DataFrame(recs, columns=cols)


_IDENT_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")


def friendly_feature_text(text: str | None) -> str:
    """Replace raw column identifiers (``estimated_delivery_days``) in a stored note by their display names."""
    return _IDENT_RE.sub(lambda m: charts.pretty_feature_name(m.group(0)), str(text or ""))


PREVIEW_ORDER = ["csv_row", BATCH_ID_COL, "late_probability", "risk_label"]


def batch_preview(scored: pd.DataFrame, limit: int = 200, blank: str = "—") -> pd.DataFrame:
    """The 'Scored orders' preview: only rows that got a probability, results first, blanks shown as a dash.

    Rows that could not be scored are listed separately (see :func:`invalid_rows_report`).  The
    ``input_error`` column is dropped (empty for every scored row).
    """
    if scored is None or scored.empty or "input_error" not in scored.columns:
        return pd.DataFrame()
    ok = scored[scored["input_error"].fillna("").eq("")].head(limit)
    first = [c for c in PREVIEW_ORDER if c in ok.columns]
    rest = [c for c in ok.columns if c not in first and c not in ("input_error", "assumptions_used", "input_warning")]
    tail = [c for c in ("assumptions_used", "input_warning") if c in ok.columns]
    out = ok[first + rest + tail].copy()
    for c in rest + tail + ([BATCH_ID_COL] if BATCH_ID_COL in out.columns else []):
        out[c] = out[c].astype("object").where(out[c].notna() & out[c].astype(str).ne(""), blank)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Display text
# --------------------------------------------------------------------------- #
def read_first(bundle: Mapping, metrics: Mapping | None) -> str:
    """Short up-front honesty statement with the real test numbers."""
    mi = model_info(bundle, metrics)
    if mi["test_roc_auc"] is None or mi["test_pr_auc"] is None:
        return ("This is a rough screening aid, not a forecast. The evaluation file with the test-window results is missing, so "
                "the model's accuracy cannot be shown here: re-create it with `.venv/bin/python -m src.train_model`.")
    base = f" against {mi['test_baseline']:.3f} for a no-skill model" if mi["test_baseline"] is not None else ""
    val = (f", much weaker than on the validation window ({mi['val_roc_auc']:.2f})"
           if mi["val_roc_auc"] is not None and mi["val_roc_auc"] - mi["test_roc_auc"] > 0.05 else "")
    return (f"This is a rough screening aid, not a forecast. On the untouched test window (n = {int(mi['test_n'] or 0):,}) the model's "
            f"ROC-AUC was **{mi['test_roc_auc']:.2f}** (0.5 = no skill) and its PR-AUC **{mi['test_pr_auc']:.3f}**{base}: a modest but "
            f"real signal{val}. Use a result as a prompt to look closer, never as a promise about one order.")


def money(x: Any) -> str:
    v = _num(x)
    return "n/a" if v is None else f"R$ {v:,.2f}"


def describe_order(inputs: Mapping[str, Any]) -> str:
    """One-line description of an order's inputs (no identifiers) for pills / scenario tables."""
    def lab(c: Any) -> str:
        s = str(c).replace("_", " ").strip()
        return s[:1].upper() + s[1:]
    parts = [f"{inputs.get('seller_state', '?')} to {inputs.get('customer_state', '?')}", lab(inputs.get("product_category", "?"))]
    n = _num(inputs.get("n_items"))
    if n is not None:
        parts.append(f"{int(n)} item{'s' if int(n) != 1 else ''}")
    parts.append(f"{money(inputs.get('total_price'))} + {money(inputs.get('total_freight'))} freight")
    d = _num(inputs.get("estimated_delivery_days"))
    if d is not None:
        parts.append(f"{int(d)}-day promise")
    if inputs.get("purchase_date") is not None:
        parts.append(f"purchased {inputs['purchase_date']}")
    return " · ".join(parts)
