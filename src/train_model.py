"""Train and evaluate the delivery-delay classifier (CLI: ``python -m src.train_model``).

Task: predict ``is_late`` (delivered after the promised calendar date) using ONLY
information known at purchase time (``config.PURCHASE_TIME_FEATURES``).

Protocol (all choices are written into ``models/model_metrics.json``)
---------------------------------------------------------------------
* chronological split by ``order_purchase_timestamp`` -- train / validation / test, no
  order in two partitions; the test window is evaluated exactly once, after every
  fitting, tuning, calibration, threshold and band decision has been frozen on
  train / validation;
* every preprocessing step (imputation, scaling, one-hot encoding) lives inside the
  sklearn ``Pipeline`` and is fitted on the training window only;
* hyper-parameters are tuned on the validation window (small grids, PR-AUC);
* probability calibration (``FrozenEstimator`` + ``CalibratedClassifierCV``) is fitted on
  validation and kept only if it improves a held-out later half of validation;
* the final model is chosen on validation PR-AUC, preferring the simpler model when the
  difference is within bootstrap noise.

No custom transformer is defined here: the only custom callable in the saved pipeline is
:func:`src.prediction.coerce_feature_frame`, so ``joblib.load`` works in any process.
"""
from __future__ import annotations

import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config
from .prediction import HARD_LIMITS, RAW_NUMERIC, STATE_PAIR_SEP, coerce_feature_frame, derive_model_features

TARGET = config.TARGET_COL
TIME_COL = "order_purchase_timestamp"

# Chronological windows (start inclusive, end exclusive).  Chosen from the monthly counts / late
# rates (see split notes): 2016 is dropped (sparse), the Black-Friday and Feb-Mar 2018 delay spikes
# sit inside the TRAIN window so the model has seen a disrupted regime, and validation / test each
# hold >= 12k orders and several hundred late orders.
TRAIN_START = "2017-01-01"
VALIDATION_START = "2018-04-01"
TEST_START = "2018-07-01"

MIN_FREQUENCY = 50          # rarer categories are grouped into one "infrequent" level per column
HIGH_TOP_FRACTION = 0.05    # risk bands: High = top 5 % of validation scores ...
MODERATE_TOP_FRACTION = 0.20  # ... Moderate = next 15 % (top 20 % flagged in total)
MAX_CURVE_POINTS = 200
RATE_SPREAD_NOTABLE = 0.01    # late-rate spread across windows (absolute) above which a base-rate shift is called out
ELEVATED_MONTH_FACTOR = 1.5   # a month is "elevated" when its late rate is >= this multiple of the window median
RARE_CLASS_RATE = 0.2         # late rate below which accuracy is described as misleading
Z95 = float(scipy.stats.norm.ppf(0.975))
VALIDATION_CAVEAT = (
    "Validation figures are tuned / in-sample for the selected model, not an out-of-sample estimate: the "
    "hyper-parameters, the calibrator (refitted on all of validation), the F1 threshold, the risk-band cuts, "
    "permutation importance and the model choice were all decided on this window. Use the test window for an "
    "honest estimate.")
SIMPLICITY_RANK = {"logistic_regression": 1, "random_forest": 2, "hist_gradient_boosting": 2}
MODEL_LABELS = {
    "dummy_prior": "Dummy baseline (class prior)",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "hist_gradient_boosting": "Histogram gradient boosting",
}

# Why allow-listed / unused candidates are not model inputs (static reasons; computed ones are added at runtime)
UNUSED_REASONS = {
    "customer_city": "high cardinality; state captures the coarse geography",
    "seller_city": "high cardinality; state captures the coarse geography",
    "state_pair": "high-cardinality interaction of seller_state x customer_state; the two states plus same_state already carry it",
    "n_products": "secondary order-size aggregate overlapping n_items",
    "n_categories": "secondary order-size aggregate overlapping n_items / product_category",
    "category_missing": "covered by the 'unknown' product_category level",
    "weight_missing": "covered by the median imputer's missing-value indicator on the weight / dimension columns",
    "purchase_year": "excluded by design: a year column only identifies the period (time-drift proxy) and can never generalise to unseen years",
    "purchase_date": "excluded by design: raw date is a time-drift proxy; calendar effects enter only through day-of-week",
    "purchase_year_month": "excluded by design: a month identifier is a time-drift proxy",
}


# --------------------------------------------------------------------------- #
# Data, split, features
# --------------------------------------------------------------------------- #
def load_training_frame(path: Path = config.ORDERS_DELIVERED_PATH) -> pd.DataFrame:
    """Load the analysis set (delivered, labelled orders); raise with the fix command if absent."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Create it with:\n    {config.ROOT}/.venv/bin/python -m src.preprocessing")
    df = pd.read_parquet(path)
    df = df[df[TARGET].notna()].copy()
    df[TARGET] = df[TARGET].astype(int)
    return df.sort_values([TIME_COL, "order_id"], kind="stable").reset_index(drop=True)


def chronological_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split by purchase timestamp into train / validation / test (orders are the unit)."""
    ts = df[TIME_COL]
    parts = {
        "train": df[(ts >= TRAIN_START) & (ts < VALIDATION_START)],
        "validation": df[(ts >= VALIDATION_START) & (ts < TEST_START)],
        "test": df[ts >= TEST_START],
    }
    return {k: v.copy() for k, v in parts.items()}


def _window_summary(part: pd.DataFrame) -> dict[str, Any]:
    if part.empty:
        return {"start": None, "end": None, "n": 0, "n_late": 0, "late_rate": None}
    return {
        "start": part[TIME_COL].min().strftime("%Y-%m-%d"),
        "end": part[TIME_COL].max().strftime("%Y-%m-%d"),
        "n": int(len(part)),
        "n_late": int(part[TARGET].sum()),
        "late_rate": float(part[TARGET].mean()),
    }


def split_notes(df: pd.DataFrame, parts: dict[str, pd.DataFrame]) -> tuple[list[str], list[dict]]:
    """Plain-language, data-derived notes on the split (sparse 2016, drift, censoring) + monthly table."""
    notes: list[str] = []
    monthly = (df.assign(month=df[TIME_COL].dt.strftime("%Y-%m"))
                 .groupby("month", observed=True)[TARGET].agg(n="size", late_rate="mean").reset_index())
    monthly_rows = [{"month": r.month, "n": int(r.n), "late_rate": float(r.late_rate)} for r in monthly.itertuples()]

    early = df[df[TIME_COL] < TRAIN_START]
    if len(early):
        notes.append(
            f"{len(early):,} orders purchased before {TRAIN_START} ({early[TIME_COL].min():%Y-%m} to "
            f"{early[TIME_COL].max():%Y-%m}) are excluded: the marketplace had almost no volume yet "
            f"(at most {int(monthly[monthly['month'] < TRAIN_START[:7]]['n'].max()):,} orders in any month), "
            "too sparse to learn from or evaluate on.")
    s = {k: _window_summary(v) for k, v in parts.items()}
    rates = [v["late_rate"] for v in s.values() if v["n"]]
    spread = max(rates) - min(rates) if rates else 0.0
    shift_txt = (f" The spread between the highest and lowest window is {spread * 100:.1f} percentage points, so "
                 "thresholds and probabilities fitted on one regime are applied to another and test precision / "
                 "calibration can differ from validation partly because of the base-rate shift."
                 if spread >= RATE_SPREAD_NOTABLE else
                 f" The spread between windows is only {spread * 100:.1f} percentage points.")
    notes.append("Late rate by window: "
                 + ", ".join(f"{k} {v['late_rate']:.2%} (n={v['n']:,})" for k, v in s.items() if v["n"]) + "." + shift_txt)
    m = monthly.set_index("month")["late_rate"]
    in_train = [mm for mm in m.index if TRAIN_START[:7] <= mm < VALIDATION_START[:7]]
    if in_train:
        top = m.loc[in_train].sort_values(ascending=False).head(3)
        med = float(m.loc[in_train].median())
        txt = ("Late rate by month inside the training window: highest "
               + ", ".join(f"{k} ({v:.1%})" for k, v in top.items()) + f" versus a median of {med:.1%}")
        if med > 0 and top.iloc[0] >= ELEVATED_MONTH_FACTOR * med:
            txt += (f". The top month is {top.iloc[0] / med:.1f}x the median: episodes of elevated delay like this may "
                    "not be anticipated from purchase-time order attributes alone, which would limit any "
                    "purchase-time model.")
        else:
            txt += f"; no month reaches {ELEVATED_MONTH_FACTOR:g}x the median."
        notes.append("Non-stationarity: " + txt)
    drift, gaps = [], {}
    for k, v in parts.items():
        if v.empty:
            continue
        same = v.groupby("same_state", observed=True)[TARGET].mean()
        gaps[k] = same.get(1, np.nan) - same.get(0, np.nan)
        drift.append(f"{k}: median promise {v['estimated_delivery_days'].astype('float64').median():.0f} days, late rate "
                     f"same-state {same.get(1, np.nan):.1%} vs other-state {same.get(0, np.nan):.1%}")
    if drift:
        signs = {np.sign(g) for g in gaps.values() if np.isfinite(g) and g != 0}
        flip = (" The same-state minus other-state late-rate gap changes sign between windows (" +
                ", ".join(f"{k} {g * 100:+.1f} pp" for k, g in gaps.items() if np.isfinite(g)) +
                "): an association learned on the earlier window may not transfer to the later one."
                if len(signs) > 1 else "")
        notes.append("Covariate / concept drift (descriptive, computed per window): " + "; ".join(drift) + "." + flip)
    span_years = (df[TIME_COL].max() - pd.Timestamp(TRAIN_START)).days / 365.25
    notes.append(
        "Month-of-year / quarter are tested as features (see features.dropped_candidates)"
        + (f": the modelled history spans {span_years:.1f} years, so a month effect cannot be separated from one-off "
           "events in a single year." if span_years < 2 else "."))
    try:
        allo = pd.read_parquet(config.ORDERS_ALL_PATH, columns=["order_status", TIME_COL])
        w = allo[allo[TIME_COL] >= TEST_START]
        open_status = {"created", "approved", "invoiced", "processing", "shipped"}
        n_open = int(w["order_status"].astype(str).isin(open_status).sum())
        last_delivery = df["order_delivered_customer_date"].max()
        notes.append(
            f"Selection bias / right-censoring: labels exist only for delivered orders. In the test window "
            f"{n_open:,} of {len(w):,} orders ({n_open / max(len(w), 1):.2%}) are still in an open status "
            f"(created/approved/invoiced/processing/shipped) at the end of the data and cannot be labelled; "
            f"if those orders are disproportionately late, the late rate of the latest months is understated "
            f"(by at most the share of open orders). Latest recorded delivery: {last_delivery:%Y-%m-%d}. Cancelled / unavailable "
            f"orders are also unlabelled.")
    except Exception:  # orders_all is optional context only
        notes.append("Selection bias: labels exist only for delivered orders (orders_all.parquet unavailable for counts).")
    notes.append("The final model is fitted on the training window only (not refitted on train+validation) so that "
                 "validation-based calibration, thresholds and bands remain honest; this ignores the most recent "
                 "weeks before the test window.")
    return notes, monthly_rows


def _spearman_pairs(train: pd.DataFrame, cols: list[str], cutoff: float = 0.9) -> list[tuple[str, str, float]]:
    """Pairs of numeric candidates with |Spearman rho| >= cutoff on the training window."""
    c = train[cols].astype("float64").corr(method="spearman")
    out = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = c.loc[a, b]
            if np.isfinite(r) and abs(r) >= cutoff:
                out.append((a, b, float(r)))
    return out


# --------------------------------------------------------------------------- #
# Pipelines
# --------------------------------------------------------------------------- #
def build_pipeline(estimator: Any, numeric: list[str], categorical: list[str], scale: bool) -> Pipeline:
    """coerce dtypes -> (median impute [+indicator] [+scale]) / (constant impute + one-hot) -> estimator."""
    num_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat_steps = [
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=MIN_FREQUENCY, sparse_output=False)),
    ]
    prep = ColumnTransformer([("num", Pipeline(num_steps), numeric), ("cat", Pipeline(cat_steps), categorical)])
    coerce = FunctionTransformer(coerce_feature_frame, kw_args={"numeric": numeric, "categorical": categorical},
                                 validate=False)
    return Pipeline([("coerce", coerce), ("prep", prep), ("model", estimator)])


def candidate_estimators(quick: bool) -> dict[str, tuple[Any, list[dict]]]:
    """name -> (estimator factory(params), parameter grid).  Small grids; quick mode shrinks them for tests."""
    rs = config.RANDOM_STATE
    n_trees = 40 if quick else 300
    lr_grid = [{"C": c, "class_weight": w} for c, w in product([0.03, 0.3] if quick else [0.01, 0.03, 0.1, 0.3, 1.0],
                                                               [None, "balanced"])]
    rf_grid = [{"min_samples_leaf": leaf, "max_features": mf}
               for leaf, mf in product([30] if quick else [10, 30, 100], ["sqrt"] if quick else ["sqrt", 0.3])]
    hgb_grid = [{"max_iter": it, "max_leaf_nodes": leaves}
                for it, leaves in product([50] if quick else [100, 250], [8] if quick else [8, 31])]
    return {
        "logistic_regression": (lambda **p: LogisticRegression(max_iter=3000, random_state=rs, **p), lr_grid),
        "random_forest": (lambda **p: RandomForestClassifier(n_estimators=n_trees, n_jobs=-1, random_state=rs, **p),
                          rf_grid),
        "hist_gradient_boosting": (lambda **p: HistGradientBoostingClassifier(
            learning_rate=0.05, min_samples_leaf=100, early_stopping=False, random_state=rs, **p), hgb_grid),
    }


def month_rule(scores: dict[str, tuple[float, float]]) -> tuple[bool, str]:
    """Decide on ``purchase_month`` from validation PR-AUC ``{model: (with, without)}``; return ``(drop, reason)``.

    The month is dropped when the MEAN change over the models is >= 0; the reason lists each model's own change so
    a model that got worse is not hidden behind the mean.
    """
    deltas = {n: wo - w for n, (w, wo) in scores.items()}
    mean_delta = float(np.mean(list(deltas.values()))) if deltas else 0.0
    detail = "; ".join(
        f"{MODEL_LABELS.get(n, n)} {w:.3f} -> {wo:.3f} ({deltas[n]:+.3f}, {'raised' if deltas[n] >= 0 else 'lowered'})"
        for n, (w, wo) in scores.items())
    lowered = [MODEL_LABELS.get(n, n) for n, d in deltas.items() if d < 0]
    reason = (
        "calendar month cannot be separated from one-off events when training covers little more than one year; "
        f"removing it did not lower the MEAN validation PR-AUC across {' and '.join(MODEL_LABELS.get(n, n) for n in scores)} "
        f"(with -> without: {detail}; mean change {mean_delta:+.3f})"
        + (f". It lowered {', '.join(lowered)} individually, so the drop is driven by the other model(s)" if lowered else "")
        + ". The rule is applied on the validation window, which is also used for tuning and model choice.")
    return mean_delta >= 0, reason


def select_features(parts: dict[str, pd.DataFrame], quick: bool) -> tuple[list[str], list[str], dict[str, str]]:
    """Choose model inputs among the purchase-time allow-list; return (numeric, categorical, dropped->reason)."""
    train, val = parts["train"], parts["validation"]
    numeric = list(config.PURCHASE_TIME_NUMERIC)
    categorical = list(config.PURCHASE_TIME_CATEGORICAL)
    dropped = {c: UNUSED_REASONS[c] for c in config.PURCHASE_TIME_UNUSED + config.NON_FEATURE_COLUMNS
               if c in UNUSED_REASONS}

    for col in ("customer_city", "seller_city"):
        if col in train.columns:
            k = train[col].nunique()
            dropped[col] = (f"high cardinality ({k:,} distinct values in {len(train):,} training orders, median "
                            f"{train[col].value_counts().median():.0f} orders each); state captures the coarse geography")
    for col in ("n_products", "n_categories"):
        if col in train.columns:
            r = float(train[[col, "n_items"]].astype("float64").corr(method="spearman").iloc[0, 1])
            dropped[col] = (f"secondary order-size aggregate overlapping n_items (Spearman rho = {r:.2f} on the training "
                              "window); kept for analytics only to limit redundancy")

    # 1. redundancy: drop the later of any pair with |Spearman rho| >= 0.9 on train
    for a, b, r in _spearman_pairs(train, numeric):
        if b in numeric:
            numeric.remove(b)
            dropped[b] = f"redundant with {a} (Spearman rho = {r:.2f} on the training window)"

    # 2. calendar month: keep only if it helps validation PR-AUC (LR + HGB, mean change)
    if "purchase_month" in numeric:
        scores: dict[str, tuple[float, float]] = {}
        for name in ("logistic_regression", "hist_gradient_boosting"):
            factory = candidate_estimators(quick)[name][0]
            params = {"C": 0.1} if name == "logistic_regression" else {"max_iter": 100, "max_leaf_nodes": 15}
            res = {}
            for label, cols in {"with": numeric, "without": [c for c in numeric if c != "purchase_month"]}.items():
                pipe = build_pipeline(factory(**params), cols, categorical, scale=name == "logistic_regression")
                pipe.fit(train[cols + categorical], train[TARGET])
                res[label] = average_precision_score(val[TARGET], pipe.predict_proba(val[cols + categorical])[:, 1])
            scores[name] = (float(res["with"]), float(res["without"]))
        drop, reason = month_rule(scores)
        if drop:
            numeric.remove("purchase_month")
            dropped["purchase_month"] = reason
        else:
            dropped.pop("purchase_month", None)
    return numeric, categorical, dropped


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def f1_optimal_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Probability cut maximising F1 (ties -> highest cut); 0.5 if undefined (one class / constant scores)."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(y) == 0 or y.sum() == 0 or np.ptp(p) == 0:
        return 0.5
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    best = np.flatnonzero(f1 >= f1.max() - 1e-12)[-1]
    return float(thr[best])


def top_fraction_stats(y: np.ndarray, p: np.ndarray, frac: float = 0.10) -> dict[str, float | None]:
    """Expected precision / recall among the ``frac`` highest-scored orders (None when undefined).

    Tied scores at the cut are split in expectation (as if broken uniformly at random) instead of by
    row order, so a constant-score model scores exactly the prevalence and tied calibrated scores are
    not flattered or penalised by chronology.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = len(y)
    if n == 0:
        return {"precision": None, "recall": None}
    k = max(1, int(np.ceil(frac * n)))
    cut = np.partition(p, n - k)[n - k]
    above, tie = p > cut, p == cut
    expected_tp = y[above].sum() + (k - int(above.sum())) * y[tie].mean()
    pos = float(y.sum())
    return {"precision": float(expected_tp / k), "recall": float(expected_tp / pos) if pos else None}


def classification_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    """Threshold metrics + ranking / probability metrics; safe for empty or one-class input (None)."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    n = len(y)
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier", "log_loss"]
    if n == 0:
        return {**dict.fromkeys(keys), "n": 0, "positive_rate": None, "predicted_positive_rate": None,
                "precision_top10pct": None, "recall_top10pct": None, "lift_top10pct": None,
                "confusion_matrix": [[0, 0], [0, 0]]}
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    both = y.min() != y.max()
    top = top_fraction_stats(y, p, 0.10)
    return {
        "accuracy": float((tn + tp) / n), "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "pr_auc": float(average_precision_score(y, p)) if y.sum() else None,
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])),
        "n": int(n), "positive_rate": float(y.mean()), "predicted_positive_rate": float(pred.mean()),
        "precision_top10pct": top["precision"], "recall_top10pct": top["recall"],
        "lift_top10pct": float(top["precision"] / y.mean()) if y.mean() > 0 else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def _downsample(*arrays: np.ndarray, max_points: int = MAX_CURVE_POINTS) -> list[list[float]]:
    """Evenly thin curve arrays to <= max_points, always keeping both end points."""
    n = len(arrays[0])
    if n <= max_points:
        sel = np.arange(n)
    else:
        sel = np.unique(np.linspace(0, n - 1, max_points).round().astype(int))
    return [np.asarray(a, dtype=float)[sel].tolist() for a in arrays]


def curve_payloads(y: np.ndarray, p: np.ndarray) -> dict[str, dict[str, list]]:
    """ROC / PR / decile-calibration curve points (<= 200 points; empty lists if undefined)."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    empty = {"roc": {"fpr": [], "tpr": []}, "pr": {"precision": [], "recall": []},
             "cal": {"mean_predicted": [], "fraction_positive": [], "bin_counts": []}}
    if len(y) == 0:
        return empty
    out = {k: dict(v) for k, v in empty.items()}
    if y.min() != y.max():
        fpr, tpr, _ = roc_curve(y, p)
        out["roc"]["fpr"], out["roc"]["tpr"] = _downsample(fpr, tpr)
        prec, rec, _ = precision_recall_curve(y, p)
        out["pr"]["precision"], out["pr"]["recall"] = _downsample(prec, rec)
    if np.ptp(p) == 0:  # constant scores (e.g. the prior-only baseline): a single bin
        bins = pd.Series(["all"] * len(p), dtype="category")
    else:
        bins = pd.qcut(pd.Series(p), q=10, duplicates="drop")
    g = pd.DataFrame({"p": p, "y": y, "bin": bins}).groupby("bin", observed=True)
    stats = g.agg(mean_predicted=("p", "mean"), fraction_positive=("y", "mean"), bin_counts=("y", "size"))
    out["cal"] = {"mean_predicted": stats["mean_predicted"].tolist(),
                  "fraction_positive": stats["fraction_positive"].tolist(),
                  "bin_counts": [int(v) for v in stats["bin_counts"]]}
    return out


def bootstrap_scores(y: np.ndarray, score_sets: dict[str, np.ndarray], n_boot: int, seed: int) -> dict[str, dict[str, np.ndarray]]:
    """Paired bootstrap of ROC-AUC and PR-AUC (same resamples for every model)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    res = {k: {"roc_auc": [], "pr_auc": []} for k in score_sets}
    if n == 0:
        return {k: {m: np.array([]) for m in v} for k, v in res.items()}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        for k, s in score_sets.items():
            res[k]["roc_auc"].append(roc_auc_score(yb, s[idx]))
            res[k]["pr_auc"].append(average_precision_score(yb, s[idx]))
    return {k: {m: np.asarray(v) for m, v in d.items()} for k, d in res.items()}


def _ci(a: np.ndarray) -> list[float] | None:
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else None


# --------------------------------------------------------------------------- #
# Fitting, tuning, calibration
# --------------------------------------------------------------------------- #
def fit_and_tune(name: str, factory: Any, grid: list[dict], parts: dict[str, pd.DataFrame],
                 numeric: list[str], categorical: list[str]) -> dict[str, Any]:
    """Fit every grid point on train, score validation PR-AUC (time-aware: validation is later than train)."""
    cols = numeric + categorical
    tr, va = parts["train"], parts["validation"]
    rows, best = [], None
    for params in grid:
        pipe = build_pipeline(factory(**params), numeric, categorical, scale=name == "logistic_regression")
        pipe.fit(tr[cols], tr[TARGET])
        pr = float(average_precision_score(va[TARGET], pipe.predict_proba(va[cols])[:, 1]))
        rows.append({"params": params, "validation_pr_auc": pr})
        if best is None or pr > best[0]:
            best = (pr, params, pipe)
    return {"base": best[2], "best_params": best[1], "grid": rows}


def calibrate(base: Pipeline, parts: dict[str, pd.DataFrame], cols: list[str]) -> dict[str, Any]:
    """Keep calibration only if it beats raw scores on a later held-out half of validation.

    Calibrators (sigmoid / isotonic) are fitted on the EARLIER half and judged on the LATER half
    (Brier and log-loss); if one wins on both, the final calibrator is refitted on all of
    validation.  Returns {"estimator", "calibrated", "method", "diagnostics"}.
    """
    va = parts["validation"].sort_values(TIME_COL, kind="stable")
    half = len(va) // 2
    a, b = va.iloc[:half], va.iloc[half:]
    diag: dict[str, Any] = {"fit_half": f"{len(a)} orders (earlier half of validation)",
                            "check_half": f"{len(b)} orders (later half of validation)"}
    if len(a) == 0 or a[TARGET].nunique() < 2 or b[TARGET].nunique() < 2:
        return {"estimator": base, "calibrated": False, "method": None,
                "diagnostics": {**diag, "reason": "validation too small / single-class to calibrate"}}
    yb = b[TARGET].to_numpy()
    raw = base.predict_proba(b[cols])[:, 1]
    diag["raw"] = {"brier": float(np.mean((raw - yb) ** 2)), "log_loss": float(log_loss(yb, np.clip(raw, 1e-15, 1 - 1e-15)))}
    best_method, best_brier = None, diag["raw"]["brier"]
    for method in ("sigmoid", "isotonic"):
        cal = CalibratedClassifierCV(FrozenEstimator(base), method=method).fit(a[cols], a[TARGET])
        pb = cal.predict_proba(b[cols])[:, 1]
        m = {"brier": float(np.mean((pb - yb) ** 2)), "log_loss": float(log_loss(yb, np.clip(pb, 1e-15, 1 - 1e-15)))}
        diag[method] = m
        if m["brier"] < best_brier and m["log_loss"] < diag["raw"]["log_loss"]:
            best_method, best_brier = method, m["brier"]
    if best_method is None:
        return {"estimator": base, "calibrated": False, "method": None,
                "diagnostics": {**diag, "decision": "raw probabilities kept (calibration did not improve the later half)"}}
    final = CalibratedClassifierCV(FrozenEstimator(base), method=best_method).fit(
        parts["validation"][cols], parts["validation"][TARGET])
    return {"estimator": final, "calibrated": True, "method": best_method,
            "diagnostics": {**diag, "decision": f"{best_method} calibration kept; refitted on full validation"}}


def choose_model(val_pr: dict[str, float], boot_pr: dict[str, np.ndarray]) -> tuple[str, list[str], str]:
    """Pick the final model from VALIDATION PR-AUC: best score, then the simplest model within bootstrap noise.

    ``boot_pr`` holds paired bootstrap PR-AUC draws per contender.  Returns ``(best, within_noise, chosen)``.
    """
    contenders = list(boot_pr)
    best = max(contenders, key=lambda n: val_pr[n])
    n_paired = min(len(boot_pr[n]) for n in contenders)
    within = []
    for n in contenders:
        diff = boot_pr[best][:n_paired] - boot_pr[n][:n_paired]
        noise = float(np.std(diff)) if n != best and n_paired else 0.0
        if val_pr[best] - val_pr[n] <= noise:
            within.append(n)
    return best, within, min(within, key=lambda n: (SIMPLICITY_RANK[n], -val_pr[n]))


def risk_thresholds(p_val: np.ndarray) -> dict[str, float]:
    """Moderate / High probability cuts = validation score quantiles (top 20 % / top 5 %)."""
    p = np.asarray(p_val, dtype=float)
    if len(p) == 0:
        return {"moderate": 0.5, "high": 0.5}
    high = float(np.quantile(p, 1 - HIGH_TOP_FRACTION))
    moderate = float(min(np.quantile(p, 1 - MODERATE_TOP_FRACTION), high))
    return {"moderate": moderate, "high": high}


def wilson_interval(k: int, n: int) -> tuple[float | None, float | None]:
    """95 % Wilson score interval for a proportion (None, None when n == 0)."""
    if n <= 0:
        return None, None
    ph = k / n
    denom = 1 + Z95 ** 2 / n
    centre = (ph + Z95 ** 2 / (2 * n)) / denom
    half = Z95 * np.sqrt(ph * (1 - ph) / n + Z95 ** 2 / (4 * n ** 2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def band_table(y: np.ndarray, p: np.ndarray, th: dict[str, float]) -> list[dict[str, Any]]:
    """Per risk band: n, late count, observed late rate and its 95 % Wilson interval (None when empty)."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    band = np.where(p >= th["high"], "High", np.where(p >= th["moderate"], "Moderate", "Low"))
    rows = []
    for name in ("Low", "Moderate", "High"):
        m = band == name
        n = int(m.sum())
        k = int(y[m].sum())
        lo, hi = wilson_interval(k, n)
        rows.append({"band": name, "n": n, "n_late": k, "late_rate": float(y[m].mean()) if n else None,
                     "late_ci_low": lo, "late_ci_high": hi})
    return rows


def band_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Do the bands actually rank risk in this window?  Monotonicity, interval overlap, actual shares."""
    total = sum(r["n"] for r in rows)
    shares = {r["band"]: (r["n"] / total if total else None) for r in rows}
    complete = all(r["n"] > 0 for r in rows)
    rates = [r["late_rate"] for r in rows]
    monotone = bool(rates[0] <= rates[1] <= rates[2]) if complete else None
    overlaps: dict[str, bool | None] = {}
    for a, b in zip(rows[:-1], rows[1:]):
        key = f"{a['band']}-{b['band']}"
        overlaps[key] = (bool(a["late_ci_high"] >= b["late_ci_low"] and b["late_ci_high"] >= a["late_ci_low"])
                         if a["n"] and b["n"] else None)
    overlapping = [k for k, v in overlaps.items() if v]
    if monotone is None:
        note = "A band is empty in this window; band ordering cannot be assessed."
    else:
        note = (f"Observed late rate {'rises' if monotone else 'does NOT rise'} monotonically from Low to High. "
                + (f"The 95% Wilson intervals of adjacent bands overlap for {', '.join(overlapping)}: those bands cannot "
                   "be told apart at this sample size." if overlapping
                   else "Adjacent bands have non-overlapping 95% Wilson intervals."))
    return {"monotone": monotone, "shares": shares, "adjacent_intervals_overlap": overlaps, "note": note}


def same_state_breakdown(y: np.ndarray, p: np.ndarray, same: np.ndarray, th: dict[str, float]) -> dict[str, Any]:
    """Per group (same-state vs other-state orders): n, observed late rate, mean predicted risk and the band table.

    The association between same-state delivery and lateness changed sign between the windows, so a pooled band
    rate can hide a large difference between the groups.  Empty groups give ``n = 0`` and ``None`` rates.
    """
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    same = np.asarray(same, dtype=float) == 1
    out: dict[str, Any] = {}
    for key, m in (("same_state", same), ("other_state", ~same)):
        n = int(m.sum())
        out[key] = {"n": n, "observed_late_rate": float(y[m].mean()) if n else None,
                    "mean_predicted": float(p[m].mean()) if n else None,
                    "bands": {r["band"]: {k: r[k] for k in ("n", "n_late", "late_rate", "late_ci_low", "late_ci_high")}
                              for r in band_table(y[m], p[m], th)}}
    return out


def same_state_note(bd: dict[str, Any]) -> str:
    """Descriptive sentence on the test window's late rate per band within each same-state group (data-derived)."""
    def cell(g: str, band: str) -> str:
        c = bd[g]["bands"][band]
        return f"{_fmt(c['late_rate'], True)} (n={c['n']:,})"
    per_band = "; ".join(f"{b} {cell('same_state', b)} vs {cell('other_state', b)}" for b in ("Low", "Moderate", "High"))
    return (f"Test-window late rate by band, same-state vs other-state orders (descriptive): {per_band}. Mean predicted "
            f"vs observed risk: same-state {_fmt(bd['same_state']['mean_predicted'], True)} vs "
            f"{_fmt(bd['same_state']['observed_late_rate'], True)} (n={bd['same_state']['n']:,}), other-state "
            f"{_fmt(bd['other_state']['mean_predicted'], True)} vs {_fmt(bd['other_state']['observed_late_rate'], True)} "
            f"(n={bd['other_state']['n']:,}); the pooled band rates above average over these two groups.")


def probability_floor(p_val: np.ndarray, y_val: np.ndarray) -> float:
    """Floor for exact-zero (isotonic) scores: the Laplace-smoothed validation late rate of that level.

    0.0 when no score is exactly zero.  Capped at the lowest positive score so the order of scores is kept.
    """
    p = np.asarray(p_val, dtype=float)
    y = np.asarray(y_val, dtype=int)
    zero = p <= 0
    if not zero.any():
        return 0.0
    floor = float((y[zero].sum() + 1) / (zero.sum() + 2))
    positive = p[p > 0]
    return float(min(floor, positive.min())) if positive.size else floor


def _proba(est: Any, X: pd.DataFrame, floor: float = 0.0) -> np.ndarray:
    """Late probabilities with the exact-zero floor applied (identical to ``prediction._score``)."""
    return np.maximum(est.predict_proba(X)[:, 1], floor)


def score_resolution(p_val: np.ndarray, floor: float) -> dict[str, Any]:
    """Describe the step-function nature of the deployed scores on the validation window."""
    p = np.asarray(p_val, dtype=float)
    return {"window": "validation", "distinct_levels": int(len(np.unique(p))), "min": float(p.min()),
            "max": float(p.max()), "probability_floor": float(floor),
            "share_at_floor": float((p <= floor).mean()) if floor > 0 else 0.0}


def support_shift(parts: dict[str, pd.DataFrame], numeric: list[str]) -> dict[str, dict[str, Any]]:
    """Per later window: share of orders with a numeric feature outside the TRAIN min-max range (extrapolation)."""
    tr = parts["train"][numeric].astype("float64")
    lo, hi = tr.min(), tr.max()
    out: dict[str, dict[str, Any]] = {}
    for k in ("validation", "test"):
        v = parts[k]
        X = v[numeric].astype("float64")
        outside = X.lt(lo) | X.gt(hi)
        any_out = outside.any(axis=1).to_numpy()
        y = v[TARGET].to_numpy()
        n_out = int(any_out.sum())
        share_by = {c: float(outside[c].mean()) for c in numeric if outside[c].any()} if len(v) else {}
        out[k] = {
            "n": int(len(v)), "n_outside": n_out, "share_outside": float(n_out / len(v)) if len(v) else None,
            "late_rate_outside": float(y[any_out].mean()) if n_out else None,
            "late_rate_inside": float(y[~any_out].mean()) if (~any_out).any() else None,
            "share_by_feature": dict(sorted(share_by.items(), key=lambda kv: -kv[1])),
        }
    return out


def support_note(shift: dict[str, dict[str, Any]]) -> str:
    """Plain-language note on out-of-training-support shares (data-derived)."""
    bits = []
    for k, v in shift.items():
        if not v["n"]:
            continue
        txt = f"{k}: {v['share_outside']:.2%} of {v['n']:,} orders"
        if v["n_outside"]:
            top = next(iter(v["share_by_feature"].items()))
            txt += (f" (mostly {top[0]}, {top[1]:.2%}); late rate {v['late_rate_outside']:.1%} (n={v['n_outside']:,}) "
                    f"vs {_fmt(v['late_rate_inside'], True)} for the rest")
        bits.append(txt)
    return ("Out-of-support orders (a numeric feature outside the training-window min-max range, where the "
            "model can only extrapolate): " + "; ".join(bits) + ".")


# --------------------------------------------------------------------------- #
# Form options (what the prediction UI may offer) and leakage audit
# --------------------------------------------------------------------------- #
def build_form_options(train: pd.DataFrame, min_pair_orders: int = 10,
                       date_span: tuple[str, str] | None = None) -> dict[str, Any]:
    """Valid categories, per-state-pair median delivery promise, defaults and ranges (training window only).

    ``ranges`` are the TRAINING ranges: inputs outside them are scored with a warning (extrapolation), only
    ``hard_limits`` (impossible values) are rejected.  ``date_span`` is the purchase-date span the model was FITTED on
    (default: the min / max ``purchase_date`` of ``train``); the caller may add the wider dataset span separately.
    """
    opts: dict[str, Any] = {
        c: sorted(train[c].astype(str).unique().tolist()) for c in ("customer_state", "seller_state", "product_category")
    }
    pair = train["seller_state"].astype(str) + STATE_PAIR_SEP + train["customer_state"].astype(str)
    g = train.groupby(pair, observed=True)["estimated_delivery_days"].agg(["median", "size"])
    g = g[g["size"] >= min_pair_orders]
    opts["state_pair_median_estimated_days"] = {k: float(v) for k, v in g["median"].items()}
    opts["state_pair_min_orders"] = int(min_pair_orders)
    num = train[RAW_NUMERIC].astype("float64")
    opts["ranges"] = {c: [float(num[c].min()), float(num[c].max())] for c in RAW_NUMERIC}
    ratio = derive_model_features(train[["customer_state", "seller_state", "total_price", "total_freight"]])["freight_ratio"]
    if ratio.notna().any():  # derived feature: ratio can leave its support while price and freight are each in range
        opts["ranges"]["freight_ratio"] = [float(ratio.min()), float(ratio.max())]
    defaults = {c: float(num[c].median()) for c in RAW_NUMERIC if c not in ("n_items", "total_price", "total_freight")}
    defaults["purchase_dow"] = float(train["purchase_dow"].astype("float64").median())
    opts["defaults"] = defaults
    opts["hard_limits"] = {c: [float(a), float(b)] for c, (a, b) in HARD_LIMITS.items()}
    if date_span is None and "purchase_date" in train.columns and train["purchase_date"].notna().any():
        date_span = (train["purchase_date"].min().strftime("%Y-%m-%d"), train["purchase_date"].max().strftime("%Y-%m-%d"))
    opts["date_span"] = list(date_span) if date_span else None
    opts["defaults_note"] = "medians of the training window; used only for optional inputs left blank"
    return opts


def derivation_parity(df: pd.DataFrame) -> list[str]:
    """Check that ``derive_model_features`` reproduces the processed feature columns; return problems."""
    raw_cols = ["customer_state", "seller_state", "total_price", "total_freight", "purchase_date"]
    derived = derive_model_features(df[raw_cols])
    problems = []
    for c in ("freight_ratio", "same_state", "purchase_month", "purchase_dow", "purchase_quarter"):
        a = pd.to_numeric(derived[c], errors="coerce").astype("float64").to_numpy()
        b = pd.to_numeric(df[c], errors="coerce").astype("float64").to_numpy()
        if not np.allclose(a, b, equal_nan=True, rtol=1e-9, atol=1e-12):
            problems.append(f"derive_model_features mismatches processed column {c}")
    return problems


def leakage_audit(numeric: list[str], categorical: list[str], parts: dict[str, pd.DataFrame],
                  base: Pipeline, bundle_columns: list[str], parity_problems: list[str]) -> dict[str, Any]:
    """Run and record the leakage / hygiene checks; ``passed`` is False if any fails."""
    feats = set(numeric + categorical)
    checks: list[tuple[str, bool]] = [
        ("features are a subset of config.PURCHASE_TIME_FEATURES", feats <= set(config.PURCHASE_TIME_FEATURES)),
        ("no feature appears in config.LEAKAGE_COLUMNS", not feats & set(config.LEAKAGE_COLUMNS)),
        ("no feature appears in config.NON_FEATURE_COLUMNS (ids, raw timestamps, year / date)",
         not feats & set(config.NON_FEATURE_COLUMNS)),
        ("target is not a feature", TARGET not in feats),
        ("pipeline input columns equal bundle feature_columns",
         list(base.named_steps["prep"].feature_names_in_) == list(bundle_columns)),
        ("derived features (freight_ratio, same_state, month/dow/quarter) reproduce the processed table row by row",
         not parity_problems),
        ("partitions are strictly chronological (max train < min validation, max validation < min test)",
         parts["train"][TIME_COL].max() < parts["validation"][TIME_COL].min()
         and parts["validation"][TIME_COL].max() < parts["test"][TIME_COL].min()),
        ("no order_id appears in two partitions",
         not (set(parts["train"]["order_id"]) & set(parts["validation"]["order_id"])
              or set(parts["train"]["order_id"]) & set(parts["test"]["order_id"])
              or set(parts["validation"]["order_id"]) & set(parts["test"]["order_id"]))),
    ]
    imputer = base.named_steps["prep"].named_transformers_["num"].named_steps["impute"]
    fitted = pd.Series(imputer.statistics_, index=numeric)
    train_med = parts["train"][numeric].astype("float64").median()
    checks.append(("imputer medians equal the TRAIN-window medians (preprocessing fitted on train only)",
                   bool(np.allclose(fitted.to_numpy(), train_med.reindex(numeric).to_numpy(), equal_nan=True))))
    return {"passed": bool(all(ok for _, ok in checks)),
            "checks": [f"{'PASS' if ok else 'FAIL'}: {t}" for t, ok in checks]}


# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #
def _clean(obj: Any) -> Any:
    """Recursively convert numpy / pandas scalars to JSON-safe python (NaN / inf -> None)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _fmt(x: float | None, pct: bool = False) -> str:
    if x is None:
        return "n/a"
    return f"{x:.1%}" if pct else f"{x:.3f}"


def test_window_assessment(sel: dict[str, Any], dummy: dict[str, Any]) -> tuple[str, list[str], bool]:
    """Data-derived summary of the selected model on the untouched test window.

    ``sel`` / ``dummy`` are ``models[name]`` entries.  Returns ``(text, warnings, degraded)``: the text states
    lift with its bootstrap interval, Brier / log-loss against the constant baseline and mean predicted vs
    observed risk; a warning is added whenever the model is not better than the baseline on a probability metric,
    its lift interval is close to 1, or it falls clearly below its validation lift.
    """
    t, v, d = sel["test"], sel["validation"], dummy["test"]
    base_t = t["positive_rate"]
    lift_t = t["pr_auc"] / base_t if base_t and t["pr_auc"] is not None else None
    lift_v = v["pr_auc"] / v["positive_rate"] if v["positive_rate"] and v["pr_auc"] is not None else None
    ci = sel["test_ci95"]["pr_auc"]
    text = (f"Test window (untouched until the end; n={t['n']:,}, {base_t:.1%} late): ROC-AUC {_fmt(t['roc_auc'])} "
            f"(validation {_fmt(v['roc_auc'])}); PR-AUC {_fmt(t['pr_auc'])} vs no-skill {base_t:.3f}")
    if lift_t is not None:
        text += f" = {lift_t:.1f}x lift (validation {lift_v:.1f}x on the same scores)" if lift_v is not None else f" = {lift_t:.1f}x lift"
    if ci:
        text += f", 95% bootstrap interval {ci[0]:.3f}-{ci[1]:.3f} ({ci[0] / base_t:.1f}x-{ci[1] / base_t:.1f}x)"
    text += (f"; Brier {t['brier']:.4f} vs {d['brier']:.4f} for the constant baseline; log-loss {t['log_loss']:.4f} vs "
             f"{d['log_loss']:.4f}; mean predicted risk {sel['mean_predicted_test']:.1%} vs observed {base_t:.1%}.")
    warns: list[str] = []
    if t["brier"] >= d["brier"]:
        warns.append("On the test window the model's Brier score is not better than the constant baseline, so its "
                     "probabilities are not better calibrated than simply quoting the base rate.")
    if t["log_loss"] >= d["log_loss"]:
        warns.append("On the test window the model's log-loss is not better than the constant baseline.")
    if ci and ci[0] < base_t * 1.25:
        warns.append("The test lift interval is close to 1: the ranking barely beats a constant prediction.")
    degraded = bool(ci and lift_v is not None and ci[1] / base_t < lift_v)
    if degraded:
        warns.append(f"Test lift (interval {ci[0] / base_t:.1f}x-{ci[1] / base_t:.1f}x) is clearly below validation lift "
                     f"({lift_v:.1f}x): performance degraded on the later window; expect weaker results in use than "
                     "the validation figures suggest.")
    if sel["calibrated"] and sel["test_pr_auc_uncalibrated"] > t["pr_auc"]:
        warns.append(f"Calibration lowered test PR-AUC ({sel['test_pr_auc_uncalibrated']:.3f} uncalibrated -> "
                     f"{t['pr_auc']:.3f}).")
    return text, warns, degraded


def test_window_caveat(sel: dict[str, Any], dummy: dict[str, Any], test_bands: list[dict[str, Any]],
                       window: dict[str, Any], degraded: bool,
                       by_same_state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Structured test-window facts stored in the bundle so the prediction UI can show them next to a probability."""
    t, d = sel["test"], dummy["test"]
    out = {
        "window": "test", "start": window["start"], "end": window["end"], "n": t["n"],
        "observed_late_rate": t["positive_rate"], "mean_predicted": sel["mean_predicted_test"],
        "brier": t["brier"], "brier_baseline": d["brier"],
        "brier_not_better_than_baseline": bool(t["brier"] >= d["brier"]),
        "log_loss": t["log_loss"], "log_loss_baseline": d["log_loss"],
        "log_loss_not_better_than_baseline": bool(t["log_loss"] >= d["log_loss"]),
        "test_performance_degraded": bool(degraded),
        "band_late_rates": {r["band"]: {k: r[k] for k in ("n", "n_late", "late_rate", "late_ci_low", "late_ci_high")}
                            for r in test_bands},
    }
    if by_same_state is not None:
        out["by_same_state"] = by_same_state
    return out


def importance_comparison(val_items: list[dict], test_items: list[dict]) -> str:
    """Compare permutation-importance rankings of the two windows (data-derived note)."""
    top = lambda items: [i["feature"] for i in sorted(items, key=lambda d: -d["importance_mean"])[:3]]  # noqa: E731
    tv = {i["feature"]: i for i in test_items}
    flips = [i["feature"] for i in val_items
             if i["feature"] in tv and abs(i["importance_mean"]) > i["importance_std"]
             and abs(tv[i["feature"]]["importance_mean"]) > tv[i["feature"]]["importance_std"]
             and (i["importance_mean"] > 0) != (tv[i["feature"]]["importance_mean"] > 0)]
    txt = (f"Top-3 by validation importance: {', '.join(top(val_items))}; by test importance: "
           f"{', '.join(top(test_items))}.")
    if flips:
        txt += f" The sign of the importance flips between windows for: {', '.join(flips)}."
    return txt + " The validation ranking does not necessarily transfer to later windows."


def train(df: pd.DataFrame | None = None, out_dir: Path | None = None, quick: bool = False,
          write: bool = True, verbose: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the full protocol; return ``(bundle, metrics)`` and (optionally) save both artefacts.

    ``quick=True`` shrinks grids / trees / bootstrap for the unit tests; the artefacts the app
    uses come from the default (full) run.
    """
    t0 = time.perf_counter()
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=FutureWarning)
    df = load_training_frame() if df is None else df
    out_dir = Path(out_dir) if out_dir is not None else config.MODELS_DIR
    n_boot = 40 if quick else 300

    parts = chronological_split(df)
    for k, v in parts.items():
        if len(v) == 0 or v[TARGET].nunique() < 2:
            raise ValueError(f"{k} window is empty or single-class; cannot train / evaluate")
    notes, monthly = split_notes(df, parts)
    log("split:", {k: _window_summary(v) for k, v in parts.items()})

    # ---- features (train / validation only) -----------------------------------------------------
    numeric, categorical, dropped = select_features(parts, quick)
    feature_columns = numeric + categorical
    log("features:", feature_columns)
    shift = support_shift(parts, numeric)
    notes.append(support_note(shift))
    tr, va, te = parts["train"], parts["validation"], parts["test"]
    ytr, yva = tr[TARGET].to_numpy(), va[TARGET].to_numpy()

    # ---- fit + tune + calibrate (train / validation only) ---------------------------------------
    fitted: dict[str, dict[str, Any]] = {}
    dummy = build_pipeline(DummyClassifier(strategy="prior"), numeric, categorical, scale=False)
    dummy.fit(tr[feature_columns], ytr)
    fitted["dummy_prior"] = {"base": dummy, "best_params": {"strategy": "prior"}, "grid": [],
                             "calibration": {"estimator": dummy, "calibrated": False, "method": None,
                                             "diagnostics": {"reason": "constant prior probability; nothing to calibrate"}}}
    for name, (factory, grid) in candidate_estimators(quick).items():
        t1 = time.perf_counter()
        res = fit_and_tune(name, factory, grid, parts, numeric, categorical)
        res["calibration"] = calibrate(res["base"], parts, feature_columns)
        fitted[name] = res
        log(f"  {name}: best {res['best_params']} val PR-AUC "
            f"{max(r['validation_pr_auc'] for r in res['grid']):.4f} "
            f"calibration={res['calibration']['method']} ({time.perf_counter() - t1:.0f}s)")

    # ---- validation scores, thresholds ------------------------------------------------------------
    val_scores_base = {n: r["base"].predict_proba(va[feature_columns])[:, 1] for n, r in fitted.items()}
    val_raw = {n: r["calibration"]["estimator"].predict_proba(va[feature_columns])[:, 1] for n, r in fitted.items()}
    floors = {n: probability_floor(val_raw[n], yva) for n in fitted}
    val_scores = {n: np.maximum(val_raw[n], floors[n]) for n in fitted}
    thresholds = {n: (0.5 if n == "dummy_prior" else f1_optimal_threshold(yva, val_scores[n])) for n in fitted}

    # ---- selection on validation (rank metric of the uncalibrated scores) ---------------------------
    boot = bootstrap_scores(yva, val_scores_base, n_boot, config.RANDOM_STATE)
    val_pr = {n: float(average_precision_score(yva, val_scores_base[n])) for n in fitted}
    contenders = [n for n in fitted if n != "dummy_prior"]
    best_name, within, chosen = choose_model(val_pr, {n: boot[n]["pr_auc"] for n in contenders})
    prevalence_val = float(yva.mean())
    dummy_pr = val_pr["dummy_prior"]
    lift_ci = _ci(boot[chosen]["pr_auc"])
    rationale = (
        "Primary metric: validation PR-AUC (average precision), because "
        + (f"only {prevalence_val:.1%} of validation orders are late" if prevalence_val < RARE_CLASS_RATE
           else f"{prevalence_val:.1%} of validation orders are late and PR-AUC focuses on that class")
        + f". Best uncalibrated score: {MODEL_LABELS[best_name]} ({val_pr[best_name]:.3f}); models within one "
        f"paired-bootstrap standard deviation of it: {', '.join(MODEL_LABELS[n] for n in within)}. "
        f"{MODEL_LABELS[chosen]} was chosen as the simplest of those (uncalibrated validation PR-AUC {val_pr[chosen]:.3f}, "
        f"95% bootstrap interval {lift_ci[0]:.3f}-{lift_ci[1]:.3f}) versus the {dummy_pr:.3f} no-skill baseline "
        f"(= validation late rate), i.e. a {val_pr[chosen] / max(dummy_pr, 1e-12):.1f}x lift.")
    if lift_ci[0] < dummy_pr * 1.25:
        rationale += (" The interval is close to the baseline: the model barely beats a constant prediction on "
                      "validation.")

    chosen_est = fitted[chosen]["calibration"]["estimator"]
    th_bands = risk_thresholds(val_scores[chosen])
    val_bands = band_table(yva, val_scores[chosen], th_bands)

    # ---- permutation importance on VALIDATION (original columns, before the ColumnTransformer) ---------
    n_repeats = 3 if quick else 10
    pi = permutation_importance(chosen_est, va[feature_columns], yva, scoring="average_precision",
                                n_repeats=n_repeats, random_state=config.RANDOM_STATE, n_jobs=1)
    pi_items = sorted(({"feature": f, "importance_mean": float(m), "importance_std": float(s)}
                       for f, m, s in zip(feature_columns, pi.importances_mean, pi.importances_std)),
                      key=lambda d: -d["importance_mean"])
    log("validation done; evaluating test window once")

    # ---- the single pass over the TEST window --------------------------------------------------------
    yte = te[TARGET].to_numpy()
    test_scores = {n: _proba(r["calibration"]["estimator"], te[feature_columns], floors[n]) for n, r in fitted.items()}
    test_base = {n: r["base"].predict_proba(te[feature_columns])[:, 1] for n, r in fitted.items()}
    boot_te = bootstrap_scores(yte, test_scores, n_boot, config.RANDOM_STATE + 1)
    models: dict[str, Any] = {}
    for n, r in fitted.items():
        mv = classification_metrics(yva, val_scores[n], thresholds[n])
        mt = classification_metrics(yte, test_scores[n], thresholds[n])
        cv, ct = curve_payloads(yva, val_scores[n]), curve_payloads(yte, test_scores[n])
        cal = r["calibration"]
        models[n] = {
            "label": MODEL_LABELS[n],
            "validation": {k: v for k, v in mv.items() if k != "confusion_matrix"},
            "validation_note": VALIDATION_CAVEAT,
            "probability_floor": floors[n],
            "test": {k: v for k, v in mt.items() if k != "confusion_matrix"},
            "threshold": thresholds[n],
            "threshold_criterion": ("0.5 (majority-class baseline: constant prior probability, never flags)"
                                    if n == "dummy_prior" else "F1-optimal cut on validation scores"),
            "best_params": r["best_params"],
            "tuning_grid": r["grid"],
            "calibrated": bool(cal["calibrated"]),
            "calibration_method": cal["method"],
            "calibration_diagnostics": cal["diagnostics"],
            "validation_pr_auc_uncalibrated": val_pr[n],
            "test_pr_auc_uncalibrated": float(average_precision_score(yte, test_base[n])),
            "validation_pr_auc_ci95": _ci(boot[n]["pr_auc"]),
            "test_ci95": {"roc_auc": _ci(boot_te[n]["roc_auc"]), "pr_auc": _ci(boot_te[n]["pr_auc"])},
            "mean_predicted_validation": float(val_scores[n].mean()),
            "mean_predicted_test": float(test_scores[n].mean()),
            "confusion_matrix_test": mt["confusion_matrix"],
            "confusion_matrix_validation": mv["confusion_matrix"],
            "roc_curve_test": ct["roc"],
            "pr_curve_test": ct["pr"],
            "calibration_test": ct["cal"],
            "roc_curve_validation": cv["roc"],
            "pr_curve_validation": cv["pr"],
            "calibration_validation": cv["cal"],
        }
    test_bands = band_table(yte, test_scores[chosen], th_bands)
    test_by_state = same_state_breakdown(yte, test_scores[chosen], te["same_state"].to_numpy(), th_bands)
    # diagnostic only: nothing is chosen from the test-window importances
    pi_te = permutation_importance(chosen_est, te[feature_columns], yte, scoring="average_precision",
                                   n_repeats=n_repeats, random_state=config.RANDOM_STATE, n_jobs=1)
    pi_test_items = sorted(({"feature": f, "importance_mean": float(m), "importance_std": float(s)}
                            for f, m, s in zip(feature_columns, pi_te.importances_mean, pi_te.importances_std)),
                           key=lambda d: -d["importance_mean"])

    # ---- calibration outcome sentence (honest about drift) -------------------------------------------
    sel = models[chosen]
    cal_text = (f"{sel['calibration_method']} calibration kept (fitted on validation)." if sel["calibrated"]
                else "Calibration did not improve the later half of validation; raw probabilities kept.")
    rationale += (f" {cal_text} Mean predicted risk vs observed late rate: validation "
                  f"{sel['mean_predicted_validation']:.1%} vs {prevalence_val:.1%}; test "
                  f"{sel['mean_predicted_test']:.1%} vs {float(yte.mean()):.1%}.")
    test_text, test_warnings, degraded = test_window_assessment(sel, models["dummy_prior"])
    rationale += " " + test_text + ("" if not test_warnings else " WARNING: " + " ".join(test_warnings))
    res = score_resolution(val_scores[chosen], floors[chosen])

    # ---- bundle + metrics ------------------------------------------------------------------------------
    parity = derivation_parity(df)
    audit = leakage_audit(numeric, categorical, parts, fitted[chosen]["base"], feature_columns, parity)
    if not audit["passed"]:
        raise AssertionError("leakage audit failed:\n" + "\n".join(audit["checks"]))
    rate_te = float(yte.mean())
    accuracy_txt = (f"Only {rate_te:.1%} of test orders are late, so accuracy is misleading: predicting 'on time' for "
                    f"every order scores {1 - rate_te:.1%} accuracy with zero late orders found. "
                    if rate_te < RARE_CLASS_RATE else
                    f"{rate_te:.1%} of test orders are late; predicting 'on time' for every order would still score "
                    f"{1 - rate_te:.1%} accuracy with zero late orders found, so look beyond accuracy. ")
    metric_guidance = (
        accuracy_txt + f"Prefer PR-AUC (average "
        f"precision; the no-skill baseline equals the late rate), recall at a useful precision (see the risk bands "
        f"and precision / recall in the top 10% of scores), ROC-AUC (ranking quality, 0.5 = no skill) and "
        f"calibration (Brier score and the calibration curve: do predicted probabilities match observed rates?). "
        + ("Performance degraded on the later test window (delay regimes shift over time): " if degraded
           else "Delay regimes shift over time, so later windows can differ; on the test window: ")
        + test_text + (" WARNING: " + " ".join(test_warnings) if test_warnings else ""))
    val_diag, test_diag = band_diagnostics(val_bands), band_diagnostics(test_bands)
    vs = val_diag["shares"]
    band_def = (f"Bands are probability cuts taken from the VALIDATION scores of the selected model: High = approximately "
                f"the top {HIGH_TOP_FRACTION:.0%} of validation orders (probability >= {th_bands['high']:.3f}); Moderate = "
                f"approximately the next {MODERATE_TOP_FRACTION - HIGH_TOP_FRACTION:.0%} (probability >= "
                f"{th_bands['moderate']:.3f}); Low = the rest. Tied scores cannot be split, so the actual validation shares "
                f"are High {_fmt(vs['High'], True)}, Moderate {_fmt(vs['Moderate'], True)}, Low {_fmt(vs['Low'], True)}. "
                f"Observed late rates per band are shown for validation (in-sample for the cuts) and test "
                f"(out-of-sample). Validation: {val_diag['note']} Test: {test_diag['note']} Bands are associations, "
                f"not guarantees. {same_state_note(test_by_state)}")
    caveat = test_window_caveat(sel, models["dummy_prior"], test_bands, _window_summary(te), degraded, test_by_state)
    train_seconds = time.perf_counter() - t0
    decision_thr = thresholds[chosen]
    form_options = build_form_options(tr)
    form_options["dataset_date_span"] = [df["purchase_date"].min().strftime("%Y-%m-%d"),
                                         df["purchase_date"].max().strftime("%Y-%m-%d")]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metrics = _clean({
        "generated_at": now,
        "target": f"{TARGET} (1 = delivered after the estimated delivery calendar date)",
        "features": {"numeric": numeric, "categorical": categorical, "dropped_candidates": dropped,
                     "rare_category_handling": f"one-hot with min_frequency={MIN_FREQUENCY}: rarer levels are grouped"},
        "split": {
            "method": "chronological by order_purchase_timestamp: train earliest, validation middle, test latest "
                      "(start inclusive, end exclusive; orders are the unit so no order is in two partitions)",
            "windows": {"train": [TRAIN_START, VALIDATION_START], "validation": [VALIDATION_START, TEST_START],
                        "test": [TEST_START, None]},
            "train": _window_summary(tr), "validation": _window_summary(va), "test": _window_summary(te),
            "notes": notes, "monthly_late_rate": monthly, "covariate_support": shift,
        },
        "models": models,
        "selected_model": {"name": chosen, "primary_metric": "validation PR-AUC (average precision)",
                           "rationale": rationale, "threshold": decision_thr,
                           "threshold_criterion": "F1-optimal cut on validation scores",
                           "calibrated": bool(sel["calibrated"]), "validation_note": VALIDATION_CAVEAT,
                           "test_window_assessment": test_text, "test_window_warnings": test_warnings,
                           "test_performance_degraded": degraded, "test_window_caveat": caveat,
                           "score_resolution": res},
        "risk_bands": {"definition": band_def, "thresholds": th_bands, "validation": val_bands, "test": test_bands,
                       "test_by_same_state": test_by_state,
                       "validation_diagnostics": val_diag, "test_diagnostics": test_diag,
                       "validation_note": VALIDATION_CAVEAT},
        "permutation_importance": {
            "window": "validation", "scoring": "average_precision (PR-AUC)", "n_repeats": n_repeats,
            "note": "Drop in validation PR-AUC when one ORIGINAL column is shuffled before preprocessing (selected "
                    "model, incl. calibrator). These are predictive associations, not causes. The validation window "
                    "was also used to tune the model, so see test_items for the out-of-sample ranking.",
            "items": pi_items,
            "test_window": "test", "test_items": pi_test_items,
            "comparison_note": importance_comparison(pi_items, pi_test_items)},
        "leakage_audit": audit,
        "metric_guidance": metric_guidance,
        "training_seconds": train_seconds,
        "library_versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                             "pandas": pd.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                             "joblib": joblib.__version__},
    })
    bundle = {
        "pipeline": chosen_est,
        "base_pipeline": fitted[chosen]["base"],
        "feature_columns": feature_columns,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "target": TARGET,
        "model_name": chosen,
        "calibrated": bool(sel["calibrated"]),
        "decision_threshold": decision_thr,
        "probability_floor": floors[chosen],
        "score_resolution": res,
        "test_window_caveat": caveat,
        "risk_thresholds": th_bands,
        "form_options": form_options,
        "trained_at": now,
        "sklearn_version": sklearn.__version__,
        "train_window": {"start": _window_summary(tr)["start"], "end": _window_summary(tr)["end"], "n": int(len(tr))},
        "metrics_summary": {"validation": models[chosen]["validation"], "test": models[chosen]["test"],
                            "primary_metric": "pr_auc", "test_performance_degraded": bool(degraded)},
    }
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, out_dir / config.MODEL_PATH.name)
        with open(out_dir / config.MODEL_METRICS_PATH.name, "w") as fh:
            json.dump(metrics, fh, indent=2, allow_nan=False)
        log(f"saved model + metrics to {out_dir} in {train_seconds:.0f}s")
    return bundle, metrics


def _print_summary(m: dict[str, Any]) -> None:
    print("\nSplit")
    for k in ("train", "validation", "test"):
        w = m["split"][k]
        print(f"  {k:<10} {w['start']} .. {w['end']}  n={w['n']:>6,}  late={w['late_rate']:.2%}")
    print("\nModels (validation | test)")
    for n, r in m["models"].items():
        v, t = r["validation"], r["test"]
        print(f"  {n:<24} PR-AUC {_fmt(v['pr_auc'])}|{_fmt(t['pr_auc'])}  ROC-AUC {_fmt(v['roc_auc'])}|{_fmt(t['roc_auc'])}  "
              f"Brier {_fmt(v['brier'])}|{_fmt(t['brier'])}  P/R/F1(test) {_fmt(t['precision'])}/{_fmt(t['recall'])}/{_fmt(t['f1'])}")
    s = m["selected_model"]
    print(f"\nSelected: {s['name']} (calibrated={s['calibrated']}, threshold={s['threshold']:.3f})\n  {s['rationale']}")
    print("\nRisk bands (validation | test)")
    for a, b in zip(m["risk_bands"]["validation"], m["risk_bands"]["test"]):
        print(f"  {a['band']:<9} n={a['n']:>6,} late={_fmt(a['late_rate'], True)} | n={b['n']:>6,} late={_fmt(b['late_rate'], True)}")
    print("\nTop permutation importances (validation, PR-AUC drop)")
    for it in m["permutation_importance"]["items"][:10]:
        print(f"  {it['feature']:<26} {it['importance_mean']:+.4f} +/- {it['importance_std']:.4f}")
    print(f"\nTraining time: {m['training_seconds']:.0f}s")


def main() -> int:
    """CLI entry point."""
    try:
        _, metrics = train()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    _print_summary(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
