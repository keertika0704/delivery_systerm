"""Descriptive analytics for DeliveryIQ (pure pandas / numpy, no Streamlit).

Every function operates on the **delivered / eligible** order frame produced by
``src.preprocessing`` and is safe on empty, single-row and tiny frames: counts become 0,
undefined means / rates become ``NaN`` and tables come back empty *with their documented
columns*.  Callers detect "no data" via ``kpi_summary(df)["n_orders"] == 0`` or
``DataFrame.empty``.

Conventions
-----------
* Rates are fractions in [0, 1]; intervals are Wilson score intervals.
* A segment is ``reliable`` when ``n_orders >= min_orders``.  Every table function takes
  ``min_orders`` (default 0, i.e. everything is ``reliable`` unless the caller sets a threshold)
  and ``alpha`` (the Wilson-interval level, default 0.05).  ``reliable`` is purely a sample-size
  flag; whether an interval excludes the overall rate is a separate question (see
  :func:`high_delay_segments`).
* Rows whose ``is_late`` is missing are never counted (they cannot be labelled on time).
* Delivery status label (used by the ``statuses`` filter): ``"late"`` if ``is_late == 1``,
  ``"on_time"`` if ``is_late == 0``.
* Everything here is descriptive / associative -- nothing is a causal estimate.
"""
from __future__ import annotations

import warnings
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import DEFAULT_ALPHA, DEFAULT_MIN_ORDERS, RANDOM_STATE, STATE_NAMES

__all__ = [
    "apply_filters", "delivery_status_label", "kpi_summary", "wilson_ci", "wilson_ci_arrays",
    "monthly_performance", "delay_trend", "duration_summary", "state_performance",
    "category_performance", "state_pair_performance", "high_delay_segments", "binned_stat",
    "weight_bins", "freight_bins", "processing_summary", "late_by_processing_bucket",
    "scatter_sample", "category_cost_comparison", "state_matrix", "same_state_comparison",
    "cramers_v_bootstrap_ci", "STATE_NAMES", "DEFAULT_MIN_ORDERS",
]

#: Fixed (domain-chosen, round-number) bin edges used by ``weight_bins`` / ``freight_bins`` /
#: ``late_by_processing_bucket``.  Bins are right-closed ``(a, b]`` (first bin includes ``a``).
WEIGHT_BIN_EDGES_G: tuple[float, ...] = (0, 500, 1000, 2000, 5000, 10000, 30000, np.inf)
FREIGHT_BIN_EDGES: tuple[float, ...] = (0, 10, 15, 20, 30, 50, np.inf)
PROCESSING_BIN_EDGES_DAYS: tuple[float, ...] = (0, 1, 2, 3, 5, 7, 14, np.inf)

PERF_COLS = [
    "n_orders", "n_late", "on_time_rate", "late_rate", "late_ci_low", "late_ci_high",
    "mean_duration", "median_duration", "mean_late_days", "mean_freight", "mean_weight_g",
    "reliable",
]
MONTHLY_COLS = [
    "purchase_year_month", "n_orders", "n_late", "late_rate", "on_time_rate", "mean_duration",
    "late_ci_low", "late_ci_high", "reliable",
]
KPI_KEYS = [
    "n_orders", "on_time_rate", "late_rate", "n_late", "mean_duration", "median_duration",
    "mean_late_days", "total_freight", "n_states",
]
DURATION_KEYS = ["n", "mean", "median", "std", "p90", "p95", "min", "max", "skew"]
BIN_COLS = [
    "bin", "bin_low", "bin_high", "n_orders", "n_late", "late_rate", "late_ci_low",
    "late_ci_high", "mean_duration", "median_duration", "mean_value", "reliable",
]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """Column as float64 (nullable ints / <NA> -> NaN); all-NaN if the column is absent."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return df[col].astype("float64")


def _labeled(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a known ``is_late`` label (a no-op view on the eligible frame)."""
    if "is_late" not in df.columns:
        return df.iloc[0:0]
    mask = df["is_late"].notna()
    return df if bool(mask.all()) else df.loc[mask]


def _empty(cols: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in cols})


def _safe_div(a, b):
    """Elementwise a / b with NaN where b == 0 (works for scalars and arrays)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(a, b, out=np.full(np.shape(a), np.nan, dtype="float64"),
                         where=np.asarray(b) != 0)


def wilson_ci_arrays(k: np.ndarray, n: np.ndarray, alpha: float = DEFAULT_ALPHA
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised Wilson score interval for proportions ``k / n`` (NaN where ``n <= 0``)."""
    k = np.asarray(k, dtype="float64")
    n = np.asarray(n, dtype="float64")
    z = float(norm.ppf(1 - alpha / 2))
    ok = n > 0
    n_s = np.where(ok, n, 1.0)
    p = np.clip(k, 0, n_s) / n_s
    denom = 1 + z * z / n_s
    centre = (p + z * z / (2 * n_s)) / denom
    half = z * np.sqrt(p * (1 - p) / n_s + z * z / (4 * n_s * n_s)) / denom
    lo = np.where(ok, np.clip(centre - half, 0.0, 1.0), np.nan)
    hi = np.where(ok, np.clip(centre + half, 0.0, 1.0), np.nan)
    return lo, hi


def wilson_ci(k: float, n: float, alpha: float = DEFAULT_ALPHA) -> tuple[float, float]:
    """Wilson score interval ``(low, high)`` for ``k`` successes in ``n`` trials.

    Returns ``(nan, nan)`` when ``n <= 0``.  Matches
    ``scipy.stats.binomtest(k, n).proportion_ci(method="wilson")``.
    """
    if n is None or k is None or not np.isfinite(n) or n <= 0:
        return (float("nan"), float("nan"))
    lo, hi = wilson_ci_arrays(np.array([k]), np.array([n]), alpha)
    return float(lo[0]), float(hi[0])


def _add_rates(out: pd.DataFrame, min_orders: int, alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    """Add on_time_rate / late_rate / Wilson CI / reliable to a frame with n_orders, n_late."""
    n = out["n_orders"].to_numpy("float64")
    k = out["n_late"].to_numpy("float64")
    out["late_rate"] = _safe_div(k, n)
    out["on_time_rate"] = _safe_div(n - k, n)
    lo, hi = wilson_ci_arrays(k, n, alpha)
    out["late_ci_low"], out["late_ci_high"] = lo, hi
    out["reliable"] = out["n_orders"] >= min_orders
    return out


def delivery_status_label(df: pd.DataFrame) -> pd.Series:
    """Per-row status label: ``"late"`` / ``"on_time"`` (``<NA>`` when ``is_late`` is missing)."""
    late = _num(df, "is_late")
    label = pd.Series(pd.NA, index=df.index, dtype="string")
    label[late == 1] = "late"
    label[late == 0] = "on_time"
    return label


def _contains(series: pd.Series, needle: str) -> np.ndarray:
    """Case-insensitive literal substring match; evaluated on the categories for category dtype."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        cats = pd.Series(series.cat.categories.astype(str)).str.contains(
            needle, case=False, regex=False).to_numpy(dtype=bool)
        codes = series.cat.codes.to_numpy()
        return np.where(codes >= 0, cats[np.clip(codes, 0, None)] if len(cats) else False, False)
    return series.astype("string").str.contains(needle, case=False, regex=False, na=False).to_numpy(dtype=bool)


STATUS_LABELS = ("late", "on_time")


def _norm_status(s: str) -> str:
    return str(s).strip().lower().replace("-", "_").replace(" ", "_")


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #
def apply_filters(
    df: pd.DataFrame,
    date_range: tuple | None = None,
    states: Iterable[str] | None = None,
    categories: Iterable[str] | None = None,
    statuses: Iterable[str] | None = None,
    search: str | None = None,
) -> pd.DataFrame:
    """Return a new frame with the requested filters applied (input is never mutated).

    * ``date_range`` -- ``(start, end)`` on ``purchase_date``, both bounds inclusive; either
      side may be ``None``.  Rows with a missing purchase date are dropped when a bound is set.
    * ``states`` -- list of ``customer_state`` codes; ``None`` / empty = no filter.
    * ``categories`` -- list of ``product_category`` values; ``None`` / empty = no filter.
    * ``statuses`` -- delivery status labels ``"late"`` / ``"on_time"`` (derived from
      ``is_late``; case-insensitive, ``"on time"`` accepted); ``None`` / empty = no filter.
      These are NOT ``order_status`` values (every row here is already delivered); an unknown
      label raises a ``UserWarning`` and matches no rows.
    * ``search`` -- case-insensitive literal substring matched against ``order_id``,
      ``customer_city``, ``seller_city`` and ``product_category`` (any column may match);
      ``None`` / blank = no filter.
    """
    mask = np.ones(len(df), dtype=bool)

    if date_range is not None:
        start, end = (list(date_range) + [None, None])[:2]
        pdate = df["purchase_date"]
        if start is not None and not pd.isna(start):
            mask &= (pdate >= pd.Timestamp(start).normalize()).to_numpy(dtype=bool)
        if end is not None and not pd.isna(end):
            mask &= (pdate <= pd.Timestamp(end).normalize()).to_numpy(dtype=bool)

    for col, values in (("customer_state", states), ("product_category", categories)):
        if values is not None:
            values = [v for v in (values if not isinstance(values, str) else [values])]
            if values:
                mask &= df[col].isin(values).to_numpy(dtype=bool)

    if statuses is not None:
        wanted = {_norm_status(s) for s in ([statuses] if isinstance(statuses, str) else statuses)}
        if wanted:
            unknown = sorted(wanted - set(STATUS_LABELS))
            if unknown:
                warnings.warn(f"Unknown delivery status label(s) {unknown}; expected 'late' or "
                              "'on_time' (derived from is_late, not order_status).", UserWarning,
                              stacklevel=2)
            mask &= delivery_status_label(df).isin(wanted).to_numpy(dtype=bool)

    if search is not None and str(search).strip():
        needle = str(search).strip()
        hit = np.zeros(len(df), dtype=bool)
        for col in ("order_id", "customer_city", "seller_city", "product_category"):
            if col in df.columns:
                hit |= _contains(df[col], needle)
        mask &= hit

    return df.loc[mask].copy()


# --------------------------------------------------------------------------- #
# Headline numbers
# --------------------------------------------------------------------------- #
def kpi_summary(df: pd.DataFrame) -> dict:
    """Headline KPIs of a (filtered) delivered frame.

    Keys: ``n_orders``, ``on_time_rate``, ``late_rate`` (fractions), ``n_late``,
    ``mean_duration`` / ``median_duration`` (days, purchase -> customer), ``mean_late_days``
    (mean ``delivery_delay_days`` among LATE orders only), ``total_freight``, ``n_states``
    (distinct customer states).  Empty input: counts and ``total_freight`` are 0 and every
    undefined rate / mean is ``NaN`` -- test ``n_orders == 0`` for emptiness.
    ``mean_late_days`` is also NaN when no order is late.
    """
    d = _labeled(df)
    n = int(len(d))
    if n == 0:
        return {"n_orders": 0, "on_time_rate": np.nan, "late_rate": np.nan, "n_late": 0,
                "mean_duration": np.nan, "median_duration": np.nan, "mean_late_days": np.nan,
                "total_freight": 0.0, "n_states": 0}
    late = _num(d, "is_late")
    n_late = int(late.sum())
    delay = _num(d, "delivery_delay_days")
    dur = _num(d, "delivery_duration_days")
    n_states = int(d["customer_state"].nunique()) if "customer_state" in d.columns else 0
    return {
        "n_orders": n,
        "on_time_rate": (n - n_late) / n,
        "late_rate": n_late / n,
        "n_late": n_late,
        "mean_duration": float(dur.mean()),
        "median_duration": float(dur.median()),
        "mean_late_days": float(delay[late == 1].mean()) if n_late else np.nan,
        "total_freight": float(_num(d, "total_freight").sum()),
        "n_states": n_states,
    }


def duration_summary(df: pd.DataFrame) -> dict:
    """Distribution of ``delivery_duration_days``: n, mean, median, std, p90, p95, min, max, skew.

    NaN for every statistic except ``n`` when undefined (empty; ``std`` needs n >= 2,
    ``skew`` needs n >= 3).
    """
    v = _num(_labeled(df), "delivery_duration_days").dropna()
    if v.empty:
        return {"n": 0, **{k: np.nan for k in DURATION_KEYS if k != "n"}}
    return {
        "n": int(len(v)), "mean": float(v.mean()), "median": float(v.median()),
        "std": float(v.std()) if len(v) > 1 else np.nan,
        "p90": float(v.quantile(0.90)), "p95": float(v.quantile(0.95)),
        "min": float(v.min()), "max": float(v.max()),
        "skew": float(v.skew()) if len(v) > 2 else np.nan,
    }


# --------------------------------------------------------------------------- #
# Segment tables
# --------------------------------------------------------------------------- #
def _perf_table(df: pd.DataFrame, keys: list[str], min_orders: int,
                alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    """One row per observed combination of ``keys`` with the PERF_COLS metrics.

    Sorted reliable-first, then by late rate (descending), then by volume.
    """
    d = _labeled(df)
    if d.empty or any(k not in d.columns for k in keys):
        return _empty([*keys, *PERF_COLS])
    late = _num(d, "is_late")
    delay = _num(d, "delivery_delay_days")
    w = pd.DataFrame({k: d[k] for k in keys})
    w["_late"] = late
    w["_dur"] = _num(d, "delivery_duration_days")
    w["_late_delay"] = delay.where(late == 1)
    w["_freight"] = _num(d, "total_freight")
    w["_weight"] = _num(d, "total_weight_g")
    out = w.groupby(keys, observed=True, sort=True).agg(
        n_orders=("_late", "size"), n_late=("_late", "sum"),
        mean_duration=("_dur", "mean"), median_duration=("_dur", "median"),
        mean_late_days=("_late_delay", "mean"), mean_freight=("_freight", "mean"),
        mean_weight_g=("_weight", "mean"),
    ).reset_index()
    for k in keys:
        out[k] = out[k].astype(str)
    out["n_orders"] = out["n_orders"].astype("int64")
    out["n_late"] = out["n_late"].astype("int64")
    out = _add_rates(out, min_orders, alpha)
    out = out.sort_values(["reliable", "late_rate", "n_orders"], ascending=[False, False, False],
                          kind="mergesort").reset_index(drop=True)
    return out[[*keys, *PERF_COLS]]


def state_performance(df: pd.DataFrame, min_orders: int = 0, by: str = "customer_state",
                      alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    """Performance per state (default ``customer_state``; pass ``by="seller_state"`` for sellers).

    Columns: ``<by>, n_orders, n_late, on_time_rate, late_rate, late_ci_low, late_ci_high,
    mean_duration, median_duration, mean_late_days, mean_freight, mean_weight_g, reliable``.
    ``reliable = n_orders >= min_orders``.  Sorted reliable-first, then late rate descending.
    """
    return _perf_table(df, [by], min_orders, alpha)


def category_performance(df: pd.DataFrame, min_orders: int = 0, alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    """Same layout as :func:`state_performance`, keyed by ``product_category``."""
    return _perf_table(df, ["product_category"], min_orders, alpha)


def state_pair_performance(df: pd.DataFrame, min_orders: int = 0, alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    """Performance per seller -> customer state pair.

    Columns: ``state_pair`` (``"SP→RJ"``), ``seller_state``, ``customer_state``, ``same_state``
    (bool) followed by the usual metrics.
    """
    t = _perf_table(df, ["seller_state", "customer_state"], min_orders, alpha)
    t.insert(0, "state_pair", (t["seller_state"] + "→" + t["customer_state"]).astype(str)
             if len(t) else pd.Series(dtype="object"))
    t.insert(3, "same_state", (t["seller_state"] == t["customer_state"]) if len(t)
             else pd.Series(dtype="bool"))
    return t


def high_delay_segments(
    perf_df: pd.DataFrame,
    min_orders: int = DEFAULT_MIN_ORDERS,
    overall_late_rate: float | None = None,
    alpha: float = DEFAULT_ALPHA,
    direction: str = "high",
) -> pd.DataFrame:
    """Segments whose late rate is *reliably* different from the overall rate.

    ``direction="high"``: ``late_rate > overall`` AND Wilson lower bound ``> overall`` AND
    ``n_orders >= min_orders``.  ``direction="low"`` mirrors it (upper bound ``< overall``).
    The Wilson bounds are recomputed from ``n_late`` / ``n_orders`` at ``alpha``.  If
    ``overall_late_rate`` is ``None`` it is the pooled rate of ``perf_df`` itself.
    Returns the flagged rows (all input columns plus ``excess_vs_overall``, in rate points,
    and ``rate_ratio``), most extreme first; an empty frame if none qualify.
    """
    extra = ["excess_vs_overall", "rate_ratio"]
    if perf_df is None or perf_df.empty:
        cols = list(perf_df.columns) if perf_df is not None else []
        return _empty([*cols, *[c for c in extra if c not in cols]])
    n = perf_df["n_orders"].to_numpy("float64")
    k = (perf_df["n_late"].to_numpy("float64") if "n_late" in perf_df.columns
         else np.round(perf_df["late_rate"].to_numpy("float64") * n))
    if overall_late_rate is None:
        overall_late_rate = float(k.sum() / n.sum()) if n.sum() > 0 else np.nan
    lo, hi = wilson_ci_arrays(k, n, alpha)
    rate = _safe_div(k, n)
    if direction == "low":
        flag = (rate < overall_late_rate) & (hi < overall_late_rate)
    else:
        flag = (rate > overall_late_rate) & (lo > overall_late_rate)
    flag &= n >= min_orders
    out = perf_df.loc[flag].copy()
    out["excess_vs_overall"] = rate[flag] - overall_late_rate
    out["rate_ratio"] = _safe_div(rate[flag], np.full(int(flag.sum()), overall_late_rate))
    return out.sort_values("excess_vs_overall", ascending=(direction == "low"), kind="mergesort")\
        .reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Time series
# --------------------------------------------------------------------------- #
def monthly_performance(df: pd.DataFrame, min_orders: int = 0, alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    """Per purchase month (``YYYY-MM``): orders, late count / rate, on-time rate, mean duration.

    Columns: ``purchase_year_month, n_orders, n_late, late_rate, on_time_rate, mean_duration,
    late_ci_low, late_ci_high, reliable``; chronological.  Months with very few orders (e.g.
    2016) have wide intervals -- check ``reliable`` / ``n_orders`` before interpreting.
    """
    d = _labeled(df)
    if d.empty or "purchase_year_month" not in d.columns:
        return _empty(MONTHLY_COLS)
    w = pd.DataFrame({"m": d["purchase_year_month"], "_late": _num(d, "is_late"),
                      "_dur": _num(d, "delivery_duration_days")})
    out = w.groupby("m", observed=True, sort=True).agg(
        n_orders=("_late", "size"), n_late=("_late", "sum"), mean_duration=("_dur", "mean"),
    ).reset_index().rename(columns={"m": "purchase_year_month"})
    out["purchase_year_month"] = out["purchase_year_month"].astype(str)
    out["n_orders"] = out["n_orders"].astype("int64")
    out["n_late"] = out["n_late"].astype("int64")
    out = _add_rates(out, min_orders, alpha)
    return out[MONTHLY_COLS]


def delay_trend(df: pd.DataFrame, freq: str = "M", min_orders: int = 0,
                alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    """Late-rate time series at ``freq`` (``"D"``, ``"W"``, ``"M"``, ``"Q"``) by purchase date.

    Columns: ``period`` (period start timestamp), ``n_orders, n_late, late_rate, on_time_rate,
    mean_duration, late_ci_low, late_ci_high, reliable``.  Periods without orders are omitted.
    """
    cols = ["period", "n_orders", "n_late", "late_rate", "on_time_rate", "mean_duration",
            "late_ci_low", "late_ci_high", "reliable"]
    d = _labeled(df)
    if d.empty or "purchase_date" not in d.columns:
        return _empty(cols)
    alias = {"ME": "M", "MS": "M", "QE": "Q", "QS": "Q", "W-SUN": "W", "YE": "Y", "YS": "Y"}
    period = d["purchase_date"].dt.to_period(alias.get(freq, freq)).dt.start_time
    w = pd.DataFrame({"period": period, "_late": _num(d, "is_late"),
                      "_dur": _num(d, "delivery_duration_days")}).dropna(subset=["period"])
    if w.empty:
        return _empty(cols)
    out = w.groupby("period", sort=True).agg(
        n_orders=("_late", "size"), n_late=("_late", "sum"), mean_duration=("_dur", "mean"),
    ).reset_index()
    out["n_orders"] = out["n_orders"].astype("int64")
    out["n_late"] = out["n_late"].astype("int64")
    out = _add_rates(out, min_orders, alpha)
    return out[cols]


# --------------------------------------------------------------------------- #
# Binned statistics
# --------------------------------------------------------------------------- #
def _edge_label(lo: float, hi: float, first: bool) -> str:
    if np.isinf(hi):
        return f"> {lo:g}"
    if lo == hi:
        return f"{lo:g}"
    return f"[{lo:g}, {hi:g}]" if first else f"({lo:g}, {hi:g}]"


def binned_stat(
    df: pd.DataFrame,
    col: str,
    bins: int | Sequence[float] = 5,
    quantile: bool = True,
    min_orders: int = 0,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """Late rate and duration per bin of a numeric column (``NaN`` values are ignored).

    * ``bins`` an ``int`` with ``quantile=True`` (default): equal-count (quantile) bins.
      Duplicate quantile edges are collapsed, so heavily tied data gives fewer bins.
    * ``bins`` an ``int`` with ``quantile=False``: equal-width bins between min and max.
    * ``bins`` a sequence: fixed user edges (``np.inf`` allowed as the last edge).
    Bins are right-closed ``(a, b]`` with the first bin also including ``a``; values outside
    fixed edges are dropped and empty bins omitted.  A constant column yields one bin.

    Columns: ``bin`` (label), ``bin_low``, ``bin_high``, ``n_orders, n_late, late_rate,
    late_ci_low, late_ci_high, mean_duration, median_duration, mean_value`` (mean of ``col``
    in the bin), ``reliable``.
    """
    d = _labeled(df)
    if d.empty or col not in d.columns:
        return _empty(BIN_COLS)
    v = _num(d, col).replace([np.inf, -np.inf], np.nan)
    ok = v.notna().to_numpy()
    if not ok.any():
        return _empty(BIN_COLS)
    vals = v.to_numpy()[ok]

    if isinstance(bins, (int, np.integer)):
        nb = max(int(bins), 1)
        if quantile:
            edges = np.unique(np.quantile(vals, np.linspace(0, 1, nb + 1)))
        else:
            edges = np.unique(np.linspace(vals.min(), vals.max(), nb + 1))
    else:
        edges = np.unique(np.asarray(list(bins), dtype="float64"))
    if len(edges) == 1:
        edges = np.array([edges[0], edges[0]])

    idx = np.searchsorted(edges, vals, side="left")
    idx = np.where(vals == edges[0], 1, idx)
    valid = (idx >= 1) & (idx <= len(edges) - 1)

    sub = d.loc[ok]
    w = pd.DataFrame({
        "bin_idx": idx, "_late": _num(sub, "is_late").to_numpy(),
        "_dur": _num(sub, "delivery_duration_days").to_numpy(), "_val": vals,
    })[valid]
    if w.empty:
        return _empty(BIN_COLS)
    out = w.groupby("bin_idx", sort=True).agg(
        n_orders=("_late", "size"), n_late=("_late", "sum"), mean_duration=("_dur", "mean"),
        median_duration=("_dur", "median"), mean_value=("_val", "mean"),
    ).reset_index()
    out["bin_low"] = edges[out["bin_idx"].to_numpy() - 1]
    out["bin_high"] = edges[out["bin_idx"].to_numpy()]
    out["bin"] = [_edge_label(lo, hi, i == 1) for lo, hi, i in
                  zip(out["bin_low"], out["bin_high"], out["bin_idx"])]
    out["n_orders"] = out["n_orders"].astype("int64")
    out["n_late"] = out["n_late"].astype("int64")
    out = _add_rates(out, min_orders, alpha)
    return out[BIN_COLS]


def weight_bins(df: pd.DataFrame, quantile: bool = False, bins: int = 5, min_orders: int = 0,
                alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    """:func:`binned_stat` on ``total_weight_g`` -- fixed gram edges (default) or quantile bins."""
    return binned_stat(df, "total_weight_g", bins if quantile else WEIGHT_BIN_EDGES_G,
                       quantile=quantile, min_orders=min_orders, alpha=alpha)


def freight_bins(df: pd.DataFrame, quantile: bool = False, bins: int = 5, min_orders: int = 0,
                 alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    """:func:`binned_stat` on ``total_freight`` -- fixed currency edges (default) or quantile bins."""
    return binned_stat(df, "total_freight", bins if quantile else FREIGHT_BIN_EDGES,
                       quantile=quantile, min_orders=min_orders, alpha=alpha)


# --------------------------------------------------------------------------- #
# Retrospective: seller processing time
# --------------------------------------------------------------------------- #
def processing_summary(df: pd.DataFrame) -> dict:
    """RETROSPECTIVE summary of ``processing_days`` (purchase -> handed to carrier).

    The carrier hand-over date is only known after the fact, so this is a diagnostic of
    *what happened*, not a purchase-time predictor.  Keys: ``n`` (valid values),
    ``n_missing``, ``mean``, ``median``, ``p90``, ``max``, ``share_of_duration`` (sum of
    processing days / sum of delivery days over orders with both), ``mean_carrier_to_customer_days``,
    ``retrospective`` (always True).  NaN when undefined.
    """
    d = _labeled(df)
    proc = _num(d, "processing_days")
    dur = _num(d, "delivery_duration_days")
    v = proc.dropna()
    both = proc.notna() & dur.notna()
    res = {"n": int(len(v)), "n_missing": int(proc.isna().sum()), "retrospective": True}
    if v.empty:
        res.update({k: np.nan for k in ("mean", "median", "p90", "max", "share_of_duration")})
    else:
        denom = float(dur[both].sum())
        res.update({
            "mean": float(v.mean()), "median": float(v.median()), "p90": float(v.quantile(0.9)),
            "max": float(v.max()),
            "share_of_duration": float(proc[both].sum() / denom) if denom > 0 else np.nan,
        })
    c2c = _num(d, "carrier_to_customer_days").dropna()
    res["mean_carrier_to_customer_days"] = float(c2c.mean()) if len(c2c) else np.nan
    return res


def late_by_processing_bucket(df: pd.DataFrame, edges: Sequence[float] = PROCESSING_BIN_EDGES_DAYS,
                              min_orders: int = 0, alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    """RETROSPECTIVE late rate per bucket of ``processing_days`` (fixed day edges).

    Not usable for purchase-time prediction (the hand-over date is unknown at checkout).
    Same columns as :func:`binned_stat`.
    """
    return binned_stat(df, "processing_days", list(edges), min_orders=min_orders, alpha=alpha)


# --------------------------------------------------------------------------- #
# Delay-analysis page helpers
# --------------------------------------------------------------------------- #
def scatter_sample(df: pd.DataFrame, x: str, y: str, n: int = 5000, seed: int = RANDOM_STATE,
                   extra: Sequence[str] = ("is_late",)) -> pd.DataFrame:
    """Deterministic sample (fixed ``seed``) of rows with finite ``x`` and ``y`` for scatter plots.

    Returns columns ``[x, y, *extra]`` (extras that do not exist are skipped), in original row
    order.  ``len(result) == min(n, rows with finite x and y)``.
    """
    cols = [c for c in dict.fromkeys([x, y, *extra]) if c in df.columns]
    if x not in df.columns or y not in df.columns:
        return _empty([x, y, *extra])
    sub = df[cols].copy()
    sub[x] = _num(sub, x)
    sub[y] = _num(sub, y)
    sub = sub.replace({x: {np.inf: np.nan, -np.inf: np.nan}, y: {np.inf: np.nan, -np.inf: np.nan}})
    sub = sub.dropna(subset=[x, y])
    if len(sub) > n:
        sub = sub.sample(n=n, random_state=seed).sort_index()
    return sub.reset_index(drop=True)


def category_cost_comparison(df: pd.DataFrame, min_orders: int = 0) -> pd.DataFrame:
    """Freight cost profile per product category (descriptive).

    Columns: ``product_category, n_orders, mean_freight, median_freight, mean_freight_ratio,
    median_freight_ratio, mean_price, mean_weight_g, total_freight, late_rate, mean_duration,
    reliable``; sorted by mean freight (descending).  ``freight_ratio`` = freight / item price.
    """
    cols = ["product_category", "n_orders", "mean_freight", "median_freight", "mean_freight_ratio",
            "median_freight_ratio", "mean_price", "mean_weight_g", "total_freight", "late_rate",
            "mean_duration", "reliable"]
    d = _labeled(df)
    if d.empty or "product_category" not in d.columns:
        return _empty(cols)
    w = pd.DataFrame({
        "k": d["product_category"], "_late": _num(d, "is_late"), "_dur": _num(d, "delivery_duration_days"),
        "_f": _num(d, "total_freight"), "_r": _num(d, "freight_ratio"),
        "_p": _num(d, "total_price"), "_w": _num(d, "total_weight_g"),
    })
    out = w.groupby("k", observed=True, sort=True).agg(
        n_orders=("_late", "size"), mean_freight=("_f", "mean"), median_freight=("_f", "median"),
        mean_freight_ratio=("_r", "mean"), median_freight_ratio=("_r", "median"),
        mean_price=("_p", "mean"), mean_weight_g=("_w", "mean"), total_freight=("_f", "sum"),
        late_rate=("_late", "mean"), mean_duration=("_dur", "mean"),
    ).reset_index().rename(columns={"k": "product_category"})
    out["product_category"] = out["product_category"].astype(str)
    out["n_orders"] = out["n_orders"].astype("int64")
    out["reliable"] = out["n_orders"] >= min_orders
    return out.sort_values("mean_freight", ascending=False, kind="mergesort").reset_index(drop=True)[cols]


def state_matrix(df: pd.DataFrame, min_orders: int = 0, top_n: int | None = None) -> dict[str, pd.DataFrame]:
    """Seller state x customer state matrices for the heat map.

    Returns ``{"n_orders", "late_rate", "mean_duration"}``: wide frames with ``seller_state``
    as the index and ``customer_state`` as the columns (rows / columns sorted by volume).
    ``late_rate`` and ``mean_duration`` are NaN where a pair has fewer than ``min_orders``
    orders or no orders; ``n_orders`` is 0 for unobserved pairs.  ``top_n`` keeps only the
    ``top_n`` busiest states on each axis.
    """
    pairs = _perf_table(df, ["seller_state", "customer_state"], min_orders)
    names = ("n_orders", "late_rate", "mean_duration")
    if pairs.empty:
        e = pd.DataFrame()
        return {k: e.copy() for k in names}
    rows = pairs.groupby("seller_state")["n_orders"].sum().sort_values(ascending=False, kind="mergesort")
    cols = pairs.groupby("customer_state")["n_orders"].sum().sort_values(ascending=False, kind="mergesort")
    if top_n:
        rows, cols = rows.head(top_n), cols.head(top_n)
    pairs = pairs[pairs["seller_state"].isin(rows.index) & pairs["customer_state"].isin(cols.index)]
    out = {}
    for name in names:
        p = pairs if name == "n_orders" else pairs[pairs["reliable"]]
        m = p.pivot(index="seller_state", columns="customer_state", values=name)
        m = m.reindex(index=rows.index, columns=cols.index)
        out[name] = m.fillna(0).astype("int64") if name == "n_orders" else m
    return out


def same_state_comparison(df: pd.DataFrame, min_orders: int = 0, alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    """Same-state (seller and customer in one state) vs cross-state orders.

    Columns: ``group`` (``"same_state"`` / ``"cross_state"``), ``n_orders, n_late, late_rate,
    late_ci_low, late_ci_high, on_time_rate, mean_duration, median_duration, reliable``
    (``reliable = n_orders >= min_orders``).
    """
    d = _labeled(df)
    cols = ["group", "n_orders", "n_late", "late_rate", "late_ci_low", "late_ci_high",
            "on_time_rate", "mean_duration", "median_duration", "reliable"]
    if d.empty or "same_state" not in d.columns:
        return _empty(cols)
    w = pd.DataFrame({
        "group": np.where(_num(d, "same_state") == 1, "same_state", "cross_state"),
        "_late": _num(d, "is_late").to_numpy(), "_dur": _num(d, "delivery_duration_days").to_numpy(),
    })
    out = w.groupby("group", sort=True).agg(
        n_orders=("_late", "size"), n_late=("_late", "sum"),
        mean_duration=("_dur", "mean"), median_duration=("_dur", "median"),
    ).reset_index()
    out["n_orders"] = out["n_orders"].astype("int64")
    out["n_late"] = out["n_late"].astype("int64")
    out = _add_rates(out, min_orders, alpha)
    return out[cols]


def _cramers_v_from_counts(table: np.ndarray, corrected: bool = False) -> float:
    """Cramer's V of a 2-D contingency table of counts (``corrected`` = Bergsma bias correction); NaN if undefined."""
    t = np.asarray(table, dtype=float)
    if t.ndim != 2:
        return float("nan")
    t = t[t.sum(axis=1) > 0][:, t.sum(axis=0) > 0]
    n = t.sum()
    if min(t.shape) < 2 or n <= 0:
        return float("nan")
    expected = np.outer(t.sum(axis=1), t.sum(axis=0)) / n
    phi2 = float(((t - expected) ** 2 / expected).sum() / n)
    r, k = t.shape
    if not corrected:
        return float(np.sqrt(phi2 / (min(r, k) - 1)))
    if n <= 1:
        return float("nan")
    phi2c = max(0.0, phi2 - (r - 1) * (k - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denom = min(rc, kc) - 1
    return float(np.sqrt(phi2c / denom)) if denom > 0 else float("nan")


def cramers_v_bootstrap_ci(table: "pd.DataFrame | np.ndarray", alpha: float = DEFAULT_ALPHA,
                           n_boot: int = 1000, seed: int = RANDOM_STATE) -> tuple[float, float, float]:
    """Bias-corrected Cramer's V of a contingency table with a parametric-bootstrap percentile interval.

    ``table`` holds observed counts (rows = levels, columns = outcome), e.g. the ``table`` of
    ``statistical_analysis.chi_square_test``.  The point estimate is Bergsma's bias-corrected V (plain
    V is biased upward in large tables with small effects).  The interval resamples the table from a
    multinomial with the observed cell proportions (``n_boot`` draws, fixed ``seed``), recomputes the
    corrected V each time and takes the ``alpha / 2`` and ``1 - alpha / 2`` percentiles, shifted so that the bootstrap median
    equals the point estimate (resampled V is inflated when the true effect is near zero) and floored at
    0.  It is descriptive: it ignores uncertainty from the pooling of rare levels.  Returns
    ``(v_corrected, ci_low, ci_high)``; all NaN when V is undefined (fewer than two non-empty rows or
    columns, or no observations).
    """
    t = np.asarray(table, dtype=float)
    v = _cramers_v_from_counts(t, corrected=True)
    if not np.isfinite(v):
        return float("nan"), float("nan"), float("nan")
    n = int(round(t.sum()))
    rng = np.random.default_rng(seed)
    probs = (t / t.sum()).ravel()
    draws = rng.multinomial(n, probs, size=max(int(n_boot), 1)).reshape((-1,) + t.shape)
    vs = np.array([_cramers_v_from_counts(d, corrected=True) for d in draws])
    vs = vs[np.isfinite(vs)]
    if vs.size == 0:
        return v, float("nan"), float("nan")
    lo, hi = np.percentile(vs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    shift = v - float(np.median(vs))          # median-bias correction: resampled V is inflated near zero
    return v, float(max(lo + shift, 0.0)), float(max(hi + shift, 0.0))
