"""Hypothesis tests and effect sizes for DeliveryIQ (scipy; no Streamlit).

All tests are *associational*.  Every result dict carries ``n`` (the rows actually used), a
``test`` key naming the procedure and enough numbers for :func:`plain_language` to explain it.
Functions never raise on empty / degenerate input; they return a result with NaN statistics
and a ``note`` explaining why the test could not be run.

Documented conventions
----------------------
* Chi-square: Pearson statistic without continuity correction; levels that would give an
  expected count below ``min_expected`` are pooled into one ``"Other (merged)"`` level.
  Effect size = Cramer's V; the label thresholds are Cohen's (0.1 / 0.3 / 0.5 for ``w``)
  scaled by ``1 / sqrt(min(r-1, c-1))``: below the small threshold = ``negligible``.
* Spearman: rank correlation with ``scipy.stats.spearmanr``; the confidence interval is the
  *approximate* Fisher-z interval with the Bonett-Wright standard error
  ``sqrt((1 + rho^2 / 2) / (n - 3))``; it is not reported when either variable is binary.  Strength by ``|rho|``: < 0.1 negligible, < 0.3 weak,
  < 0.5 moderate, < 0.7 strong, otherwise very strong.
* Difference of two proportions: Newcombe's hybrid score interval (built from Wilson
  intervals; method 10 in Newcombe 1998) -- better behaved than Wald for small n / extreme p.
* Kruskal-Wallis effect size: epsilon-squared = H / (n - 1) (Tomczak & Tomczak 2014).  The
  0.01 / 0.08 / 0.26 labels are the rule-of-thumb benchmarks for it popularised by Field (2013);
  they are conventions, not statistical thresholds.
* p-values below the smallest positive float (about 2.2e-308) underflow to 0 in scipy; they are
  reported as that floor (``P_FLOOR``) instead, so a stored p of ``P_FLOOR`` means "smaller than
  can be represented", never exactly zero.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .analytics import wilson_ci
from .config import DEFAULT_ALPHA

__all__ = [
    "chi_square_test", "spearman_test", "proportion_diff_ci", "two_proportion_ztest",
    "kruskal_or_mannwhitney", "run_all_tests", "plain_language", "holm_adjust",
    "cramers_v_label", "spearman_strength", "CAVEAT", "MERGED_LABEL", "HUGE_N", "P_FLOOR",
    "POOLED_MONTH_CAVEAT",
]

MERGED_LABEL = "Other (merged)"
HUGE_N = 10_000  # above this, tiny effects routinely reach p < 0.05
P_FLOOR = float(np.finfo("float64").tiny)  # smallest positive normal float; stands for "underflowed"
POOLED_MONTH_CAVEAT = (
    "Calendar months are pooled across all years in the data (and not every month appears in "
    "every year), so this compares month-of-year groups that are also mixed with year and with "
    "one-off events; it does not isolate seasonality.")
CAVEAT = (
    "Statistical significance is not the same as practical importance, and neither shows "
    "causality: these are associations in observational data, so an association does not mean "
    "that one factor causes the other."
)


# --------------------------------------------------------------------------- #
# Labels and small helpers
# --------------------------------------------------------------------------- #
def cramers_v_label(v: float, min_dim: int) -> str:
    """Effect-size label for Cramer's V given ``min_dim = min(rows-1, cols-1)``."""
    if v is None or not np.isfinite(v) or min_dim < 1:
        return "undefined"
    scale = 1.0 / math.sqrt(min_dim)
    if v < 0.1 * scale:
        return "negligible"
    if v < 0.3 * scale:
        return "small"
    if v < 0.5 * scale:
        return "medium"
    return "large"


def spearman_strength(rho: float) -> str:
    """Strength label from ``|rho|`` (thresholds in the module docstring)."""
    if rho is None or not np.isfinite(rho):
        return "undefined"
    a = abs(rho)
    for cut, label in ((0.1, "negligible"), (0.3, "weak"), (0.5, "moderate"), (0.7, "strong")):
        if a < cut:
            return label
    return "very strong"


def _rank_biserial_label(r: float) -> str:
    """Label for ``|r|`` of the rank-biserial correlation (0.1 / 0.3 / 0.5 cut-offs)."""
    a = abs(r)
    return "negligible" if a < 0.1 else "small" if a < 0.3 else "medium" if a < 0.5 else "large"


def _floor_p(p: float) -> float:
    """Clip an underflowed p-value (0.0) up to ``P_FLOOR``; NaN stays NaN."""
    p = float(p)
    return max(p, P_FLOOR) if np.isfinite(p) else p


def _epsilon_squared_label(eps2: float) -> str:
    """Label for Kruskal-Wallis epsilon-squared (0.01 / 0.08 / 0.26 cut-offs)."""
    if eps2 is None or not np.isfinite(eps2):
        return "undefined"
    return ("negligible" if eps2 < 0.01 else "small" if eps2 < 0.08 else
            "medium" if eps2 < 0.26 else "large")


def _f(x) -> float:
    return float(x) if x is not None and np.isfinite(x) else float("nan")


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (NaN stays NaN)."""
    p = np.asarray(p_values, dtype="float64")
    out = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    if len(ok) == 0:
        return out
    order = ok[np.argsort(p[ok])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        out[i] = min(1.0, running)
    return out


# --------------------------------------------------------------------------- #
# Chi-square test of independence
# --------------------------------------------------------------------------- #
def _chi_result(col: str, target: str, n: int, note: str) -> dict:
    return {
        "test": "chi_square", "col": col, "target": target, "chi2": np.nan, "dof": 0,
        "p_value": np.nan, "n": n, "cramers_v": np.nan, "cramers_v_corrected": np.nan,
        "effect_label": "undefined", "table": pd.DataFrame(), "residuals": pd.DataFrame(),
        "level_summary": pd.DataFrame(), "merged_rare_levels": [], "n_levels_original": 0,
        "n_levels_tested": 0, "min_expected_after": np.nan, "significant": False,
        "alpha": DEFAULT_ALPHA, "note": note,
    }


def chi_square_test(
    df: pd.DataFrame,
    col: str,
    target: str = "is_late",
    min_expected: float = 5,
    top_n: int | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    """Chi-square test of independence between categorical ``col`` and ``target`` (``is_late``).

    Rare levels are merged: every level whose smallest expected count would be below
    ``min_expected`` is pooled into ``"Other (merged)"`` (the next-smallest level is added
    when the pool is a single level or is itself still too small).  ``top_n`` additionally
    keeps only the ``top_n`` most frequent levels and pools the rest.

    Returns ``chi2, dof, p_value, n, cramers_v`` (uncorrected), ``cramers_v_corrected``
    (Bergsma bias correction), ``effect_label``, ``table`` (observed counts after merging;
    rows = levels, columns = target values), ``residuals`` (adjusted standardized residuals,
    same shape), ``level_summary`` (for a 0/1 target: level, n, n_late, late_rate,
    adj_residual -- sorted by |adj_residual|, i.e. the top contributing levels),
    ``merged_rare_levels`` (original level names pooled; a ``top_n`` pool may be one level), ``n_levels_original``,
    ``n_levels_tested``, ``min_expected_after``, ``significant``, ``alpha``, ``note``.
    NaN statistics (and an explanatory ``note``) if fewer than 2 levels / target values remain.
    """
    if col not in df.columns or target not in df.columns:
        return _chi_result(col, target, 0, "column not found")
    x = df[col].astype("object")
    y = df[target].astype("float64") if str(df[target].dtype) not in ("object", "str", "category") \
        else df[target].astype("object")
    ok = x.notna() & y.notna()
    x, y = x[ok].astype(str), y[ok]
    n = int(len(x))
    if n == 0:
        return _chi_result(col, target, 0, "no complete rows")
    table = pd.crosstab(x, y)
    n_orig = int(table.shape[0])
    if table.shape[1] < 2:
        res = _chi_result(col, target, n, "target has a single value in this data")
        res["n_levels_original"] = n_orig
        return res

    col_tot = table.sum(axis=0).to_numpy("float64")
    thr = min_expected * n / col_tot.min()  # row total needed for min expected count
    row_tot = table.sum(axis=1)
    order = row_tot.sort_values(kind="mergesort").index.tolist()  # ascending size
    merged: list[str] = []
    if top_n is not None and n_orig > top_n:
        merged = order[: n_orig - top_n]
    rare = [lvl for lvl in order if row_tot[lvl] < thr]
    for lvl in rare:
        if lvl not in merged:
            merged.append(lvl)
    if merged:
        rest = [lvl for lvl in order if lvl not in merged]
        # A pool built because levels are rare must be >= 2 levels and itself large enough; a
        # pool made only by ``top_n`` (no rare level) may be a single level.
        while rare and rest and (len(merged) < 2 or row_tot[merged].sum() < thr):
            merged.append(rest.pop(0))
        table = pd.concat([table.drop(index=merged),
                           table.loc[merged].sum().to_frame(MERGED_LABEL).T])
    table = table.loc[table.sum(axis=1) > 0]
    if table.shape[0] < 2:
        res = _chi_result(col, target, n, "fewer than 2 levels remain after merging rare levels")
        res.update(n_levels_original=n_orig, merged_rare_levels=sorted(merged),
                   table=table)
        return res

    chi2, p, dof, expected = stats.chi2_contingency(table.to_numpy(), correction=False)
    r, c = table.shape
    min_dim = min(r - 1, c - 1)
    phi2 = chi2 / n
    v = math.sqrt(phi2 / min_dim)
    phi2c = max(0.0, phi2 - (r - 1) * (c - 1) / (n - 1)) if n > 1 else float("nan")
    rc = r - (r - 1) ** 2 / (n - 1) if n > 1 else float("nan")
    cc = c - (c - 1) ** 2 / (n - 1) if n > 1 else float("nan")
    denom = min(rc - 1, cc - 1) if n > 1 else float("nan")
    v_corr = math.sqrt(phi2c / denom) if np.isfinite(denom) and denom > 0 else float("nan")

    exp = pd.DataFrame(expected, index=table.index, columns=table.columns)
    rt = table.sum(axis=1).to_numpy("float64")[:, None]
    ct = table.sum(axis=0).to_numpy("float64")[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        resid = (table - exp) / np.sqrt(exp * (1 - rt / n) * (1 - ct / n))
    resid = resid.replace([np.inf, -np.inf], np.nan)

    level_summary = pd.DataFrame()
    cols_set = {float(cn) if isinstance(cn, (int, float, np.number)) else cn for cn in table.columns}
    if cols_set == {0.0, 1.0}:
        col1 = [cn for cn in table.columns if float(cn) == 1.0][0]
        col0 = [cn for cn in table.columns if float(cn) == 0.0][0]
        n_lvl = table[col0] + table[col1]
        level_summary = pd.DataFrame({
            "level": table.index, "n": n_lvl.to_numpy(), "n_late": table[col1].to_numpy(),
            "late_rate": (table[col1] / n_lvl).to_numpy(), "adj_residual": resid[col1].to_numpy(),
        })
        level_summary = level_summary.reindex(
            level_summary["adj_residual"].abs().sort_values(ascending=False, kind="mergesort").index
        ).reset_index(drop=True)

    return {
        "test": "chi_square", "col": col, "target": target, "chi2": float(chi2), "dof": int(dof),
        "p_value": _floor_p(p), "n": n, "cramers_v": float(v), "cramers_v_corrected": float(v_corr),
        "effect_label": cramers_v_label(v, min_dim), "table": table, "residuals": resid,
        "level_summary": level_summary, "merged_rare_levels": sorted(merged),
        "n_levels_original": n_orig, "n_levels_tested": int(r),
        "min_expected_after": float(expected.min()), "significant": bool(p < alpha),
        "alpha": alpha, "note": "",
    }


# --------------------------------------------------------------------------- #
# Spearman rank correlation
# --------------------------------------------------------------------------- #
def spearman_test(df: pd.DataFrame, x: str, y: str, alpha: float = DEFAULT_ALPHA) -> dict:
    """Spearman rank correlation of ``x`` and ``y`` (NaN pairs dropped; ``n`` = pairs used).

    Returns ``rho, p_value, n, ci_low, ci_high`` (approximate Fisher-z CI at ``1 - alpha``),
    ``strength_label``, ``direction``, ``significant``.  With a binary 0/1 variable (e.g.
    ``is_late``) this is the point-biserial-style rank association and the CI is NaN (with a
    ``note``) because the normal-theory interval is unreliable there.  NaN results (plus a
    ``note``) when n < 4 or either variable is constant.
    """
    res = {"test": "spearman", "x": x, "y": y, "rho": np.nan, "p_value": np.nan, "n": 0,
           "ci_low": np.nan, "ci_high": np.nan, "strength_label": "undefined",
           "direction": "none", "significant": False, "alpha": alpha, "note": ""}
    if x not in df.columns or y not in df.columns:
        res["note"] = "column not found"
        return res
    pair = pd.DataFrame({"x": df[x].astype("float64"), "y": df[y].astype("float64")})
    pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(pair))
    res["n"] = n
    if n < 4:
        res["note"] = "fewer than 4 complete pairs"
        return res
    if pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        res["note"] = "one variable is constant"
        return res
    rho, p = stats.spearmanr(pair["x"].to_numpy(), pair["y"].to_numpy())
    rho, p = float(rho), _floor_p(p)
    if pair["x"].nunique() == 2 or pair["y"].nunique() == 2:
        # The Fisher-z / Bonett-Wright interval assumes a roughly continuous bivariate normal
        # pair; with a binary variable it is unreliable, so it is not reported.
        lo = hi = float("nan")
        res["note"] = ("confidence interval not reported: one variable is binary, where the "
                       "normal-theory interval is unreliable")
    elif abs(rho) >= 1 - 1e-12:  # perfect monotone relation: the Fisher-z interval degenerates
        lo = hi = rho
    else:
        se = math.sqrt((1 + rho * rho / 2) / (n - 3))
        zc = float(stats.norm.ppf(1 - alpha / 2))
        z = math.atanh(rho)
        lo, hi = math.tanh(z - zc * se), math.tanh(z + zc * se)
    res.update(rho=rho, p_value=p, ci_low=lo, ci_high=hi, strength_label=spearman_strength(rho),
               direction="positive" if rho > 0 else "negative" if rho < 0 else "none",
               significant=bool(p < alpha))
    return res


# --------------------------------------------------------------------------- #
# Two proportions
# --------------------------------------------------------------------------- #
def proportion_diff_ci(k1: float, n1: float, k2: float, n2: float,
                       alpha: float = DEFAULT_ALPHA) -> dict:
    """Difference ``p1 - p2`` with Newcombe's hybrid score CI (method 10).

    Returns ``p1, p2, diff, ci_low, ci_high, n1, n2, method``; all NaN if either n <= 0.
    """
    out = {"test": "proportion_diff", "p1": np.nan, "p2": np.nan, "diff": np.nan,
           "ci_low": np.nan, "ci_high": np.nan, "n1": int(n1) if n1 else 0,
           "n2": int(n2) if n2 else 0, "method": "newcombe_hybrid_score", "alpha": alpha}
    if not n1 or not n2 or n1 <= 0 or n2 <= 0:
        return out
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson_ci(k1, n1, alpha)
    l2, u2 = wilson_ci(k2, n2, alpha)
    d = p1 - p2
    out.update(p1=p1, p2=p2, diff=d,
               ci_low=d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
               ci_high=d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))
    return out


def two_proportion_ztest(k1: float, n1: float, k2: float, n2: float) -> dict:
    """Two-sided pooled z-test of ``p1 == p2``.  Returns ``z, p_value`` (NaN if undefined)."""
    res = {"z": np.nan, "p_value": np.nan}
    if not n1 or not n2 or n1 <= 0 or n2 <= 0:
        return res
    pooled = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return res
    z = (k1 / n1 - k2 / n2) / se
    return {"z": float(z), "p_value": _floor_p(2 * stats.norm.sf(abs(z)))}


# --------------------------------------------------------------------------- #
# Mann-Whitney / Kruskal-Wallis
# --------------------------------------------------------------------------- #
def kruskal_or_mannwhitney(df: pd.DataFrame, value: str, group: str, min_group_size: int = 2,
                           alpha: float = DEFAULT_ALPHA) -> dict:
    """Compare the distribution of numeric ``value`` across the levels of ``group``.

    Two groups -> Mann-Whitney U (effect size = rank-biserial correlation of the first vs
    second level in sorted order); three or more -> Kruskal-Wallis H (effect size = epsilon^2 = H / (n - 1)).
    Groups smaller than ``min_group_size`` are dropped.  Returns ``test`` (``"mann_whitney"`` /
    ``"kruskal"``), ``statistic, p_value, n, n_groups, effect_size, effect_label,
    group_summary`` (group, n, median, mean), ``significant``, ``note``.
    """
    res = {"test": "kruskal_or_mannwhitney", "value": value, "group": group, "statistic": np.nan,
           "p_value": np.nan, "n": 0, "n_groups": 0, "effect_size": np.nan,
           "effect_label": "undefined", "group_summary": pd.DataFrame(), "significant": False,
           "alpha": alpha, "note": ""}
    if value not in df.columns or group not in df.columns:
        res["note"] = "column not found"
        return res
    d = pd.DataFrame({"g": df[group].astype("object"), "v": df[value].astype("float64")})
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    d["g"] = d["g"].astype(str)
    sizes = d.groupby("g", sort=True)["v"].size()
    keep = sizes[sizes >= min_group_size].index
    d = d[d["g"].isin(keep)]
    res["n"] = int(len(d))
    summ = d.groupby("g", sort=True)["v"].agg(n="size", median="median", mean="mean").reset_index()
    res["group_summary"] = summ.rename(columns={"g": "group"})
    res["n_groups"] = int(len(summ))
    if len(summ) < 2:
        res["note"] = "fewer than 2 groups with enough observations"
        return res
    samples = [grp["v"].to_numpy() for _, grp in d.groupby("g", sort=True)]
    if len(samples) == 2:
        u, p = stats.mannwhitneyu(samples[0], samples[1], alternative="two-sided")
        r = 2 * u / (len(samples[0]) * len(samples[1])) - 1
        res.update(test="mann_whitney", statistic=float(u), p_value=_floor_p(p),
                   effect_size=float(r), effect_label=_rank_biserial_label(r))
    else:
        if d["v"].nunique() < 2:
            res["note"] = "all values identical"
            return res
        h, p = stats.kruskal(*samples)
        k, n = len(samples), len(d)
        eps2 = min(1.0, max(0.0, h / (n - 1))) if n > 1 else float("nan")
        res.update(test="kruskal", statistic=float(h), p_value=_floor_p(p),
                   effect_size=float(eps2), effect_label=_epsilon_squared_label(eps2))
    res["significant"] = bool(res["p_value"] < alpha)
    return res


# --------------------------------------------------------------------------- #
# Orchestration and plain language
# --------------------------------------------------------------------------- #
CHI_SQUARE_SPECS = [
    ("customer_state", "customer state"),
    ("product_category", "product category"),
    ("same_state", "seller and customer in the same state"),
    ("purchase_month", "purchase month of the year"),
]
SPEARMAN_SPECS = [
    ("total_weight_g", "delivery_duration_days"),
    ("total_freight", "delivery_duration_days"),
    ("total_weight_g", "total_freight"),
    ("freight_ratio", "is_late"),
    ("total_weight_g", "is_late"),
    ("estimated_delivery_days", "is_late"),
]


def run_all_tests(df: pd.DataFrame, alpha: float = DEFAULT_ALPHA, min_expected: float = 5) -> dict:
    """Run the standard battery on a delivered frame.

    * chi-square vs ``is_late``: ``customer_state``, ``product_category``, ``same_state``,
      ``purchase_month`` (months pooled across years; its result carries a ``caveat``);
    * Spearman: weight~duration, freight~duration, weight~freight, freight_ratio~is_late,
      weight~is_late, estimated_delivery_days~is_late (the last two are point-biserial-style);
    * group comparisons: Mann-Whitney of ``delivery_duration_days`` by ``same_state`` and
      Kruskal-Wallis of ``delivery_duration_days`` by ``customer_state`` (groups with >= 30
      orders).

    Returns ``{"n_orders", "alpha", "chi_square": {col: result}, "spearman": {"x~y": result},
    "group_comparison": {name: result}, "summary": DataFrame}``.  ``summary`` has one row per
    test (``test, variables, statistic, p_value, p_holm, effect_size, effect_label, n,
    significant_holm``) with Holm-adjusted p-values across the whole battery, because many
    tests are run at once.  Each result's ``n_tests`` is the number of tests that produced a
    p-value (undefined tests are not part of the adjustment).
    """
    chi = {col: chi_square_test(df, col, "is_late", min_expected=min_expected, alpha=alpha)
           for col, _ in CHI_SQUARE_SPECS}
    sp = {f"{x}~{y}": spearman_test(df, x, y, alpha) for x, y in SPEARMAN_SPECS}
    grp = {
        "delivery_duration_days~same_state": kruskal_or_mannwhitney(
            df, "delivery_duration_days", "same_state", alpha=alpha),
        "delivery_duration_days~customer_state": kruskal_or_mannwhitney(
            df, "delivery_duration_days", "customer_state", min_group_size=30, alpha=alpha),
    }
    rows = []
    for col, r in chi.items():
        rows.append(("chi-square", f"{col} x is_late", r["chi2"], r["p_value"], r["cramers_v"],
                     r["effect_label"], r["n"]))
    for key, r in sp.items():
        rows.append(("spearman", key.replace("~", " ~ "), r["rho"], r["p_value"], r["rho"],
                     r["strength_label"], r["n"]))
    for key, r in grp.items():
        rows.append((r["test"], key.replace("~", " by "), r["statistic"], r["p_value"],
                     r["effect_size"], r["effect_label"], r["n"]))
    summary = pd.DataFrame(rows, columns=["test", "variables", "statistic", "p_value",
                                          "effect_size", "effect_label", "n"])
    summary["p_holm"] = holm_adjust(summary["p_value"].to_numpy())
    summary["significant_holm"] = summary["p_holm"] < alpha
    if "purchase_month" in chi:
        chi["purchase_month"]["caveat"] = POOLED_MONTH_CAVEAT
    # let plain_language() quote the battery-adjusted p-value next to the per-test one; the
    # count is the number of p-values Holm actually ran over (tests that could not run are NaN)
    n_run = int(summary["p_value"].notna().sum())
    for r, p_adj in zip([*chi.values(), *sp.values(), *grp.values()], summary["p_holm"]):
        r["p_holm"], r["n_tests"] = float(p_adj), n_run
    return {"n_orders": int(len(df)), "alpha": alpha, "chi_square": chi, "spearman": sp,
            "group_comparison": grp, "summary": summary}


def _level(result: dict) -> str:
    """Confidence level text (e.g. ``95%``) from the ``alpha`` stored in a result dict."""
    return f"{100 * (1 - _alpha(result)):g}%"


def _fmt_p(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "p = n/a"
    if p < 1e-300:  # underflowed (stored as P_FLOOR) or a Holm multiple of it
        return "p < 0.001 (below floating-point precision)"
    return "p < 0.001 (p = %.2g)" % p if p < 0.001 else "p = %.3f" % p


def _pct(x: float) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.1f}%"


def _alpha(result: dict) -> float:
    a = result.get("alpha", DEFAULT_ALPHA)
    return DEFAULT_ALPHA if a is None or not np.isfinite(a) else float(a)


def _adjustment_note(result: dict) -> str:
    """Say whether the per-test significance statement is adjusted for the test battery."""
    p_adj, m = result.get("p_holm"), result.get("n_tests")
    if p_adj is None or not np.isfinite(p_adj) or not m:
        return ("This significance statement is for this test alone (unadjusted for multiple "
                "testing).")
    verdict = "still statistically detectable" if p_adj < _alpha(result) else "not statistically detectable"
    return (f"This statement is unadjusted; after Holm adjustment across the {m} tests run together "
            f"({_fmt_p(p_adj)}) it is {verdict}.")


def _closing(n: int, alpha: float = DEFAULT_ALPHA, adjustment: str = "") -> str:
    parts = [adjustment] if adjustment else []
    if n >= HUGE_N:
        parts.append(
            f"With n = {n:,} observations, even tiny effects can be statistically 'significant' "
            f"(p < {alpha:g}); look at the effect size, not only the p-value.")
    parts.append(CAVEAT)
    return " ".join(parts)


def plain_language(result: dict) -> str:
    """Plain-English reading of one test result (chi-square, Spearman, group comparison,
    proportion difference).  Always ends with the significance / practical importance /
    causality caveat, plus a large-sample warning when n is huge."""
    if not result:
        return CAVEAT
    test = result.get("test", "")
    n = int(result.get("n", 0) or 0)
    if test == "chi_square":
        if not np.isfinite(result.get("chi2", np.nan)):
            return (f"The chi-square test for {result.get('col')} could not be run "
                    f"(n = {n:,}): {result.get('note') or 'insufficient data'}. " + CAVEAT)
        sig = "a statistically detectable" if result["significant"] else "no statistically detectable"
        txt = (f"Chi-square test of {result['col']} vs late delivery (n = {n:,}, "
               f"chi2 = {result['chi2']:.1f}, df = {result['dof']}, {_fmt_p(result['p_value'])}): "
               f"{sig} association. Effect size (Cramer's V) = {result['cramers_v']:.3f} "
               f"({result['effect_label']}).")
        ls = result.get("level_summary")
        if ls is not None and len(ls):
            top = ls.iloc[0]
            txt += (f" The level that departs most from independence is '{top['level']}' "
                    f"(late rate {_pct(top['late_rate'])}, n = {int(top['n']):,}).")
        if result.get("merged_rare_levels"):
            txt += (f" {len(result['merged_rare_levels'])} rare level(s) were pooled so that "
                    "expected counts stay valid.")
        if result.get("caveat"):
            txt += " Note: " + result["caveat"]
        return txt + " " + _closing(n, _alpha(result), _adjustment_note(result))
    if test == "spearman":
        if not np.isfinite(result.get("rho", np.nan)):
            return (f"The Spearman correlation of {result.get('x')} and {result.get('y')} could not "
                    f"be computed (n = {n:,}): {result.get('note') or 'insufficient data'}. " + CAVEAT)
        sig = "statistically detectable" if result["significant"] else "not statistically detectable"
        ci = (f"approximate {_level(result)} CI {result['ci_low']:.3f} to {result['ci_high']:.3f}"
              if np.isfinite(result.get("ci_low", np.nan)) and np.isfinite(result.get("ci_high", np.nan))
              else "no confidence interval shown (one variable is binary)")
        return (f"Spearman rank correlation between {result['x']} and {result['y']} "
                f"(n = {n:,} pairs): rho = {result['rho']:.3f} ({result['strength_label']}, "
                f"{result['direction']}); {ci}; {_fmt_p(result['p_value'])}, i.e. {sig}. "
                + _closing(n, _alpha(result), _adjustment_note(result)))
    if test in ("mann_whitney", "kruskal", "kruskal_or_mannwhitney"):
        if not np.isfinite(result.get("p_value", np.nan)):
            return (f"The group comparison of {result.get('value')} by {result.get('group')} could not "
                    f"be run (n = {n:,}): {result.get('note') or 'insufficient data'}. " + CAVEAT)
        name = "Mann-Whitney U" if test == "mann_whitney" else "Kruskal-Wallis"
        sig = "differ detectably" if result["significant"] else "do not differ detectably"
        return (f"{name} test of {result['value']} across {result['n_groups']} groups of "
                f"{result['group']} (n = {n:,}): the distributions {sig} "
                f"({_fmt_p(result['p_value'])}); effect size {result['effect_size']:.3f} "
                f"({result['effect_label']}). "
                + _closing(n, _alpha(result), _adjustment_note(result)))
    if test == "proportion_diff":
        if not np.isfinite(result.get("diff", np.nan)):
            return "The difference in proportions could not be computed (an empty group). " + CAVEAT
        n_tot = int(result["n1"] + result["n2"])
        return (f"Rate in group 1 = {_pct(result['p1'])} (n = {result['n1']:,}) vs group 2 = "
                f"{_pct(result['p2'])} (n = {result['n2']:,}); difference "
                f"{100 * result['diff']:+.1f} percentage points, {_level(result)} CI "
                f"{100 * result['ci_low']:+.1f} to {100 * result['ci_high']:+.1f} (Newcombe). "
                + _closing(n_tot, _alpha(result)))
    return CAVEAT
