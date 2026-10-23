"""Generate, execute and verify notebooks/02_statistical_analysis.ipynb.

Usage (from anywhere):  .venv/bin/python notebooks/_build_nb02.py

* Code cells live in this file; narrative text lives in notebooks/_nb02_text.py.
* Every number quoted in the narrative is filled in from ``compute_values()`` (computed here,
  independently of the notebook, from the processed parquet) via ``«name»`` placeholders.
* After execution the notebook prints a ``KEY_RESULTS`` JSON; ``verify_against_notebook`` compares
  it with ``compute_values()`` so the text cannot drift from the executed outputs.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import _nb02_text as TXT  # noqa: E402  # same directory

NB_PATH = HERE / "02_statistical_analysis.ipynb"
cells: list = []


def md(key: str) -> None:
    cells.append(("md", key))


def code(src: str) -> None:
    cells.append(("code", src.strip("\n")))


# =============================================================================
# 0. Title + setup
# =============================================================================
md("title")

code('''
import sys, json, math, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator
from matplotlib.patches import Patch
from scipy import stats

# Resolve the project root whether the notebook is run from notebooks/ or from the root.
_here = Path.cwd().resolve()
ROOT = next((p for p in [_here, *_here.parents] if (p / "src" / "config.py").is_file()), None)
if ROOT is None:
    raise RuntimeError("Run this notebook from inside the DeliveryIQ project folder.")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src import analytics as an
from src import statistical_analysis as sa
from src.preprocessing import load_processed

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 80)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

P = config.PALETTE
plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 110, "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.edgecolor": P["slate_light"], "axes.labelcolor": P["slate"],
    "axes.titlecolor": P["navy"], "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": P["border"], "grid.linewidth": 0.7,
    "axes.axisbelow": True, "xtick.color": P["slate"], "ytick.color": P["slate"],
    "font.size": 9.5, "legend.frameon": False,
})
FIG_DIR = config.FIGURES_DIR
FIG_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = config.DEFAULT_ALPHA          # 0.05, two-sided everywhere
MIN_N = config.DEFAULT_MIN_ORDERS     # segments below this are flagged as small samples
SEED = config.RANDOM_STATE
ZC = float(stats.norm.ppf(1 - ALPHA / 2))   # 1.959964: the same critical value src.statistical_analysis uses
P_FLOOR = sa.P_FLOOR                  # scipy underflows p-values below ~2.2e-308 to 0; we report this floor instead


def save_show(fig, name: str) -> None:
    """Save a figure as PNG under reports/figures/ and display it inline."""
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    plt.show()


def pct_axis(ax, axis: str = "x") -> None:
    """Percent axis whose ticks fall on whole numbers (values are already in percent units)."""
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(MaxNLocator(integer=True, nbins=8))
    a.set_major_formatter(PercentFormatter(decimals=0))


def fmt_p(p: float) -> str:
    """Readable p-value; values at the floating-point floor are shown as an explicit bound."""
    if p is None or not np.isfinite(p):
        return "n/a"
    if p <= P_FLOOR * 1.0001:
        return "< 1e-300"
    return f"{p:.2e}" if p < 0.001 else f"{p:.4f}"


def pct(x: float, nd: int = 2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.{nd}f}%"


# The notebook-wide ledger of tests: filled as the analysis proceeds, adjusted with Holm in section 9.
LEDGER: list[dict] = []


def log_test(rq, name, test, n, p, effect_name, effect, label, family="primary"):
    LEDGER.append(dict(rq=rq, test=name, kind=test, n=int(n), p_value=float(p),
                       effect_name=effect_name, effect=float(effect), effect_label=label, family=family))


print(f"Project root : {ROOT}")
print(f"Figures dir  : {FIG_DIR.relative_to(ROOT)}   (this notebook writes 21_*.png ... 29_*.png)")
print(f"alpha = {ALPHA}  |  small-sample flag: n < {MIN_N}  |  random seed = {SEED}")
''')

# =============================================================================
# 1. Data + framework
# =============================================================================
md("s1_intro")

code('''
d = load_processed()                         # data/processed/orders_delivered.parquet (analysis set)
for c in d.select_dtypes("category").columns:
    d[c] = d[c].cat.remove_unused_categories()
d["late"] = d["is_late"].astype(int)         # plain int copy of the target (0 = on time / early, 1 = late)
d["ym"] = d["purchase_year_month"].astype(str)

N = len(d)
K_LATE = int(d["late"].sum())
BASE = K_LATE / N
lo, hi = an.wilson_ci(K_LATE, N)
print(f"Analysis set : {N:,} delivered orders, purchases {d.order_purchase_timestamp.min():%Y-%m-%d} .. {d.order_purchase_timestamp.max():%Y-%m-%d}")
print(f"Late orders  : {K_LATE:,}   overall late rate = {pct(BASE)}   (Wilson 95% CI {pct(lo)} to {pct(hi)})")
print(f"Late = delivered on a calendar date AFTER the estimated delivery date (see ARCHITECTURE.md section 1).")

# Orders whose product has no category in the source data carry the placeholder level "unknown" (category_missing = 1).
# That is a missing-data bucket, not a product category, so RQ2 (chi-square, ranking, category-vs-rest) uses d_cat only.
UNKNOWN_CAT = "unknown"
d_cat = d[d["product_category"] != UNKNOWN_CAT].copy()
if str(d_cat["product_category"].dtype) == "category":
    d_cat["product_category"] = d_cat["product_category"].cat.remove_unused_categories()
N_UNKNOWN = N - len(d_cat)
N_CAT, K_CAT = len(d_cat), int(d_cat["late"].sum())
BASE_CAT = K_CAT / N_CAT
assert N_UNKNOWN == int(d["category_missing"].sum()), "'unknown' bucket and category_missing flag disagree"
print(f"Missing-category bucket '{UNKNOWN_CAT}': {N_UNKNOWN:,} orders ({N_UNKNOWN / N:.1%}); excluded from the product-category analysis (RQ2) only, "
      f"which therefore rests on {N_CAT:,} orders (late rate {pct(BASE_CAT)}).")

used = ["customer_state", "product_category", "same_state", "purchase_month", "purchase_dow",
        "total_weight_g", "total_freight", "freight_ratio", "estimated_delivery_days",
        "delivery_duration_days", "processing_days", "carrier_to_customer_days"]
avail = pd.DataFrame({"variable": used,
                      "type": [str(d[c].dtype) for c in used],
                      "n_available": [int(d[c].notna().sum()) for c in used],
                      "n_missing": [int(d[c].isna().sum()) for c in used]})
avail["pct_missing"] = (100 * avail["n_missing"] / N).round(3)
avail
''')

md("s1_plan")

code('''
plan = pd.DataFrame([
    ("RQ1", "customer_state x late", "Chi-square independence; Cramer's V; adjusted std. residuals; state-vs-rest z-tests (Holm)",
     "late rate identical in every customer state"),
    ("RQ2", "product_category x late", "Chi-square (rare levels pooled); Cramer's V; category-vs-rest z-tests (Holm)",
     "late rate identical in every product category"),
    ("RQ3", "same_state x late; same_state vs duration", "Chi-square 2x2 / Newcombe CI / odds ratio; Mann-Whitney; Mantel-Haenszel stratified by customer state",
     "late rate (and duration) identical for same-state and cross-state orders"),
    ("RQ4", "weight, freight, freight ratio, promise vs duration", "Spearman rho, Fisher-z CI, bootstrap CI, partial rho",
     "rho = 0 (no monotone association)"),
    ("RQ5", "purchase month / year-month / weekday x late", "Chi-square; spike-month vs other-month proportion difference",
     "late rate identical across months"),
    ("RQ6", "processing_days (RETROSPECTIVE) vs late / duration", "Spearman; Mann-Whitney; chi-square on processing buckets",
     "rho = 0 / identical distributions"),
], columns=["RQ", "variables", "procedure", "H0 (null hypothesis)"])
plan
''')

# =============================================================================
# Helpers: independent re-implementations (scipy / numpy only)
# =============================================================================
md("s1_helpers")

code('''
# ---- Independent re-implementations (plain numpy / scipy, no src.statistical_analysis) --------------
def manual_chi2(frame: pd.DataFrame, col: str, target: str = "late", min_exp: float = 5, top_n=None):
    """Chi-square test of independence built directly from a pandas crosstab + scipy.

    Rare levels (smallest expected count < min_exp) are pooled into 'Other (merged)'; the pool must hold
    at least two levels and itself satisfy the expected-count rule. Returns a dict with the table.
    """
    t = pd.crosstab(frame[col].astype(str), frame[target])
    n = t.to_numpy().sum()
    exp_raw = np.outer(t.sum(axis=1), t.sum(axis=0)) / n
    row_need = min_exp * n / t.sum(axis=0).min()                    # row total needed so that min expected >= min_exp
    asc = t.sum(axis=1).sort_values(kind="mergesort").index.tolist()
    pool = [lvl for lvl in asc if t.loc[lvl].sum() < row_need]
    rest = [lvl for lvl in asc if lvl not in pool]
    while pool and rest and (len(pool) < 2 or t.loc[pool].sum().sum() < row_need):
        pool.append(rest.pop(0))
    tt = t if not pool else pd.concat([t.drop(index=pool), t.loc[pool].sum().to_frame("Other (merged)").T])
    chi2, p, dof, exp = stats.chi2_contingency(tt.to_numpy(), correction=False)
    r, c = tt.shape
    v = math.sqrt(chi2 / n / min(r - 1, c - 1))
    return dict(chi2=float(chi2), p=float(p), dof=int(dof), v=v, n=int(n), table=tt, levels_before=int(t.shape[0]),
                levels_after=int(r), pooled=sorted(pool), min_exp_before=float(exp_raw.min()),
                cells_below5_before=int((exp_raw < 5).sum()), cells_before=int(exp_raw.size),
                min_exp_after=float(exp.min()), cells_below5_after=int((exp < 5).sum()))


def v_corrected(obs: np.ndarray) -> float:
    """Bergsma (2013) bias-corrected Cramer's V of a table of counts (0 when the corrected phi^2 is negative)."""
    obs = np.asarray(obs, dtype=float)
    n = obs.sum(); r, c = obs.shape
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0)) / n
    phi2c = max(0.0, float(((obs - exp) ** 2 / exp).sum()) / n - (r - 1) * (c - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1); cc = c - (c - 1) ** 2 / (n - 1)
    return math.sqrt(phi2c / min(rc - 1, cc - 1))


def v_bootstrap(table: pd.DataFrame, B: int = 2000, seed: int = SEED):
    """Median-bias-corrected parametric (multinomial) bootstrap 95% interval for the BIAS-CORRECTED Cramer's V.

    Resampling orders from the observed table (a multinomial over its cells) re-creates the table's own small-sample
    upward bias, and applying the Bergsma correction again does not remove it: the raw percentile interval for a
    table with many cells and a tiny effect lies ABOVE its own point estimate (this happened for product category).
    The interval is therefore shifted so that the bootstrap median equals the point estimate, and floored at 0.
    """
    rng = np.random.default_rng(seed)
    obs = table.to_numpy().astype(float)
    n = int(obs.sum()); r, c = obs.shape; k = min(r - 1, c - 1)
    draws = rng.multinomial(n, (obs / n).ravel(), size=B).reshape(B, r, c).astype(float)
    rows = draws.sum(axis=2, keepdims=True); cols = draws.sum(axis=1, keepdims=True)
    exp = rows * cols / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = np.nansum(np.where(exp > 0, (draws - exp) ** 2 / exp, 0.0), axis=(1, 2))
    phi2c = np.maximum(0.0, chi / n - (r - 1) * (c - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1); cc = c - (c - 1) ** 2 / (n - 1)
    v = np.sqrt(phi2c / min(rc - 1, cc - 1))
    shift = v_corrected(obs) - float(np.median(v))
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(max(lo + shift, 0.0)), float(max(hi + shift, 0.0))


def spearman_full(frame: pd.DataFrame, x: str, y: str, B: int = 0, seed: int = SEED):
    """Spearman rho via scipy, via Pearson-on-ranks, Fisher-z CIs (two standard errors) and optional bootstrap CI."""
    pair = frame[[x, y]].astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
    xv, yv = pair[x].to_numpy(), pair[y].to_numpy()
    n = len(pair)
    rho, p = stats.spearmanr(xv, yv)
    rho_ranks = float(np.corrcoef(stats.rankdata(xv), stats.rankdata(yv))[0, 1])
    z = math.atanh(rho)
    out = dict(x=x, y=y, n=n, rho=float(rho), p=max(float(p), P_FLOOR), rho_pearson_on_ranks=rho_ranks)
    for tag, se in (("bw", math.sqrt((1 + rho ** 2 / 2) / (n - 3))), ("classic", 1 / math.sqrt(n - 3))):
        out[f"ci_{tag}"] = (math.tanh(z - ZC * se), math.tanh(z + ZC * se))
    if B:
        rng = np.random.default_rng(seed)
        bs = np.empty(B)
        for b in range(B):
            i = rng.integers(0, n, n)
            bs[b] = np.corrcoef(stats.rankdata(xv[i]), stats.rankdata(yv[i]))[0, 1]
        out["ci_boot"] = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    return out


def strength(r: float) -> str:
    return sa.spearman_strength(r)


def forest(ax, tab, label_col, overall, rate="late_rate", lo_c="late_ci_low", hi_c="late_ci_high", n_col="n_orders",
           label_fmt=lambda s: s):
    """Horizontal Wilson-interval 'forest': first row at the top. Red = interval above overall, green = below, grey = includes it.
    Hollow markers = small sample (n < MIN_N)."""
    t = tab.reset_index(drop=True)
    y = np.arange(len(t))[::-1]
    r, lo_, hi_ = t[rate].to_numpy() * 100, t[lo_c].to_numpy() * 100, t[hi_c].to_numpy() * 100
    colr = np.where(lo_ > overall * 100, P["red"], np.where(hi_ < overall * 100, P["green"], P["slate_light"]))
    for yi, l, h, c in zip(y, lo_, hi_, colr):
        ax.hlines(yi, l, h, color=c, lw=1.8, zorder=2)
    small = (t[n_col] < MIN_N).to_numpy()
    ax.scatter(r[~small], y[~small], c=colr[~small], s=26, zorder=3)
    ax.scatter(r[small], y[small], facecolors="white", edgecolors=colr[small], s=26, zorder=3, linewidths=1.3)
    ax.axvline(overall * 100, color=P["navy"], ls=":", lw=1.2, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{label_fmt(str(a))}  (n={int(n):,})" for a, n in zip(t[label_col], t[n_col])], fontsize=8)
    ax.set_ylim(-0.8, len(t) - 0.2)
    pct_axis(ax, "x")
    ax.grid(axis="y", visible=False)


def or_table(a, b, c, dd):
    """Odds ratio (late) of group 1 vs group 0 with Woolf 95% CI. a,b = group1 late/on-time; c,dd = group0 late/on-time."""
    if min(a, b, c, dd) == 0:
        return np.nan, np.nan, np.nan
    o = (a * dd) / (b * c)
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / dd)
    return o, math.exp(math.log(o) - ZC * se), math.exp(math.log(o) + ZC * se)


print("helpers defined")
''')

# =============================================================================
# 2. RQ1 regional
# =============================================================================
md("s2_intro")

code('''
st = an.state_performance(d, min_orders=MIN_N)                         # n, n_late, late_rate, Wilson CI, durations
st["state_name"] = st["customer_state"].map(config.STATE_NAMES)
st = st.sort_values("late_rate", ascending=False).reset_index(drop=True)
view = st[["customer_state", "state_name", "n_orders", "n_late", "late_rate", "late_ci_low", "late_ci_high",
           "median_duration", "reliable"]].copy()
for c in ["late_rate", "late_ci_low", "late_ci_high"]:
    view[c] = (100 * view[c]).round(2)
view = view.rename(columns={"late_rate": "late_%", "late_ci_low": "CI_low_%", "late_ci_high": "CI_high_%",
                            "median_duration": "median_days", "reliable": f"n>={MIN_N}"})
print(f"{len(st)} customer states; {int((~st.reliable).sum())} have n < {MIN_N} (small sample - wide intervals, interpret with care).")
view
''')

code('''
# ---- Chi-square: customer_state x late.  src module AND an independent scipy/pandas re-computation ----
def compare_chi(col: str, frame: pd.DataFrame = None, **kw):
    frame = d if frame is None else frame
    s = sa.chi_square_test(frame, col, "late", **kw)
    m = manual_chi2(frame, col, "late", **{k: v for k, v in kw.items() if k == "min_expected" and False})
    return s, m

CHI, MAN = {}, {}
for col in ["customer_state", "product_category", "same_state", "purchase_month"]:
    CHI[col], MAN[col] = compare_chi(col, d_cat if col == "product_category" else None)   # RQ2 excludes the 'unknown' bucket

def agreement_row(col):
    s, m = CHI[col], MAN[col]
    return {"variable": col, "chi2 (src)": s["chi2"], "chi2 (scipy by hand)": m["chi2"],
            "|diff|": abs(s["chi2"] - m["chi2"]), "dof (src / hand)": f'{s["dof"]} / {m["dof"]}',
            "Cramer V (src)": s["cramers_v"], "Cramer V (hand)": m["v"],
            "p (src)": fmt_p(s["p_value"]), "p (hand)": fmt_p(max(m["p"], P_FLOOR)),
            "same pooled levels": s["merged_rare_levels"] == m["pooled"]}

agree = pd.DataFrame([agreement_row(c) for c in CHI])
display(agree)
assert (agree["|diff|"] < 1e-6).all() and agree["same pooled levels"].all(), "src and scipy-by-hand disagree"
print("src.statistical_analysis and the independent scipy computation agree on every chi-square statistic.")
''')

md("s2_assump")

code('''
def assumption_table(cols):
    rows = []
    for col in cols:
        m = MAN[col]
        rows.append({"variable": col, "levels observed": m["levels_before"],
                     "cells with expected < 5 (before pooling)": f'{m["cells_below5_before"]} of {m["cells_before"]}',
                     "min expected before": m["min_exp_before"], "levels tested": m["levels_after"],
                     "levels pooled into 'Other'": len(m["pooled"]), "min expected after": m["min_exp_after"],
                     "cells with expected < 5 (after)": m["cells_below5_after"], "n": m["n"]})
    return pd.DataFrame(rows)

display(assumption_table(["customer_state", "product_category", "same_state", "purchase_month"]))
print("Pooled customer states :", MAN["customer_state"]["pooled"])
print("Pooled product categories (%d): %s" % (len(MAN["product_category"]["pooled"]), ", ".join(MAN["product_category"]["pooled"])))
''')

code('''
# ---- effect sizes with uncertainty: Cramer's V (uncorrected, bias-corrected, median-bias-corrected multinomial-bootstrap 95% CI) ----
VCI = {}
rows = []
for col, s in CHI.items():
    VCI[col] = v_bootstrap(s["table"])
    # the interval must contain its own point estimate, and must match the interval the app's src module reports
    assert VCI[col][0] <= s["cramers_v_corrected"] <= VCI[col][1], f"bootstrap interval for {col} excludes its point estimate"
    _vs, _lo_s, _hi_s = an.cramers_v_bootstrap_ci(s["table"], n_boot=2000, seed=SEED)
    assert abs(_vs - s["cramers_v_corrected"]) < 1e-9 and abs(_lo_s - VCI[col][0]) < 1e-9 and abs(_hi_s - VCI[col][1]) < 1e-9, f"src bootstrap CI differs for {col}"
    rows.append({"variable": col, "n": s["n"], "chi2": s["chi2"], "dof": s["dof"], "p": fmt_p(s["p_value"]),
                 "Cramer V": s["cramers_v"], "V (bias-corrected)": s["cramers_v_corrected"],
                 "corrected V, bootstrap 95% CI": f"{VCI[col][0]:.3f} to {VCI[col][1]:.3f}", "label (Cohen)": s["effect_label"]})
chi_summary = pd.DataFrame(rows)
for col, s in CHI.items():
    log_test({"customer_state": "RQ1", "product_category": "RQ2", "same_state": "RQ3", "purchase_month": "RQ5"}[col],
             f"chi-square {col} x late", "chi-square", s["n"], s["p_value"], "Cramer's V", s["cramers_v"], s["effect_label"])
chi_summary
''')

code('''
# ---- Figure 21: late rate by customer state (Wilson CI) + adjusted standardized residuals --------------
res_map = dict(zip(CHI["customer_state"]["level_summary"]["level"], CHI["customer_state"]["level_summary"]["adj_residual"]))
order = st.copy()
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 8.2), gridspec_kw={"width_ratios": [1.55, 1]}, sharey=False)
forest(ax, order, "customer_state", BASE)
ax.set_xlabel("Late-delivery rate (%), with Wilson 95% CI")
ax.set_title(f"Late rate by customer state (overall {pct(BASE)})")
ax.plot([], [], "o", color=P["red"], label="interval above overall")
ax.plot([], [], "o", color=P["green"], label="interval below overall")
ax.plot([], [], "o", color=P["slate_light"], label="interval includes overall")
ax.plot([], [], "o", mfc="white", mec=P["slate"], label=f"small sample (n<{MIN_N})")
ax.legend(loc="lower right", fontsize=8)
y = np.arange(len(order))[::-1]
vals = order["customer_state"].map(res_map).to_numpy(dtype=float)
cols_r = np.where(vals > 0, P["red"], P["green"])
ax2.barh(y[np.isfinite(vals)], vals[np.isfinite(vals)], color=cols_r[np.isfinite(vals)], height=0.6)
for yi, v, s_ in zip(y, vals, order["customer_state"]):
    if not np.isfinite(v):
        ax2.text(0.3, yi, "pooled into 'Other' (expected count < 5)", va="center", fontsize=7.5, color=P["slate_light"])
ax2.axvline(0, color=P["navy"], lw=1); ax2.axvline(1.96, color=P["slate_light"], ls=":"); ax2.axvline(-1.96, color=P["slate_light"], ls=":")
ax2.set_yticks(y); ax2.set_yticklabels(order["customer_state"], fontsize=8); ax2.set_ylim(-0.8, len(order) - 0.2)
ax2.set_xlabel("Adjusted standardized residual (late cell)\\n> 1.96 = more late orders than independence predicts")
ax2.set_title("Which states drive the chi-square?"); ax2.grid(axis="y", visible=False)
save_show(fig, "21_regional_late_rate_and_residuals")
''')

code('''
# ---- state-vs-rest proportion tests (Newcombe CI + pooled z) with Holm correction, reliable states only -------
def vs_rest(tab: pd.DataFrame, key: str, total_k: int = K_LATE, total_n: int = N, only_reliable: bool = True) -> pd.DataFrame:
    t = tab[tab["reliable"]] if only_reliable else tab
    rows = []
    for _, r in t.iterrows():
        k, n = int(r["n_late"]), int(r["n_orders"])
        diff = sa.proportion_diff_ci(k, n, total_k - k, total_n - n)
        z = sa.two_proportion_ztest(k, n, total_k - k, total_n - n)
        rows.append({key: r[key], "n": n, "late_%": 100 * k / n, "rest_%": 100 * (total_k - k) / (total_n - n),
                     "diff_pp": 100 * diff["diff"], "CI_low_pp": 100 * diff["ci_low"], "CI_high_pp": 100 * diff["ci_high"],
                     "p_raw": z["p_value"]})
    out = pd.DataFrame(rows)
    out["p_holm"] = sa.holm_adjust(out["p_raw"].to_numpy())
    out["sig_raw"] = out["p_raw"] < ALPHA
    out["sig_holm"] = out["p_holm"] < ALPHA
    return out.sort_values("diff_pp", ascending=False).reset_index(drop=True)

state_vs_rest = vs_rest(st, "customer_state")
m = len(state_vs_rest)
print(f"{m} states with n >= {MIN_N} were each compared with all other orders: "
      f"{int(state_vs_rest.sig_raw.sum())} differ at unadjusted p < {ALPHA}; "
      f"{int(state_vs_rest.sig_holm.sum())} survive Holm adjustment for {m} comparisons "
      f"({int((state_vs_rest.sig_raw & ~state_vs_rest.sig_holm).sum())} do not).")
tbl = state_vs_rest.copy()
tbl["p_raw"] = tbl["p_raw"].map(fmt_p); tbl["p_holm"] = tbl["p_holm"].map(fmt_p)
display(tbl.round(2))
''')

code('''
# Two named contrasts that dominate the volume: Rio de Janeiro vs Sao Paulo (both large, both well measured)
def named_contrast(a, b, col="customer_state"):
    ka, na = int(d.loc[d[col] == a, "late"].sum()), int((d[col] == a).sum())
    kb, nb = int(d.loc[d[col] == b, "late"].sum()), int((d[col] == b).sum())
    diff = sa.proportion_diff_ci(ka, na, kb, nb); z = sa.two_proportion_ztest(ka, na, kb, nb)
    return dict(a=a, b=b, ka=ka, na=na, kb=kb, nb=nb, **{k: diff[k] for k in ("p1", "p2", "diff", "ci_low", "ci_high")}, p=z["p_value"])

RJ_SP = named_contrast("RJ", "SP")
print(f"RJ: {RJ_SP['ka']:,}/{RJ_SP['na']:,} late = {pct(RJ_SP['p1'])}   SP: {RJ_SP['kb']:,}/{RJ_SP['nb']:,} late = {pct(RJ_SP['p2'])}")
print(f"difference RJ - SP = {100 * RJ_SP['diff']:+.2f} pp, Newcombe 95% CI {100 * RJ_SP['ci_low']:+.2f} to {100 * RJ_SP['ci_high']:+.2f} pp, p {fmt_p(RJ_SP['p'])}")
''')

code('''
# ---- Is a state's late rate just a long-distance story?  (descriptive, state level, ecological) -----------------
sr = st[st["reliable"]].copy()
sr["mean_promise_days"] = sr["customer_state"].map(d.groupby("customer_state", observed=True)["estimated_delivery_days"].mean())
sr["same_state_share"] = sr["customer_state"].map(d.groupby("customer_state", observed=True)["same_state"].mean())
rho_dur = stats.spearmanr(sr["late_rate"], sr["median_duration"])
rho_same = stats.spearmanr(sr["late_rate"], sr["same_state_share"])
rho_prom = stats.spearmanr(sr["late_rate"], sr["mean_promise_days"])
print(f"Across the {len(sr)} reliable states (state-level, n = {len(sr)} points):")
print(f"  late rate vs median delivery duration : Spearman rho = {rho_dur.statistic:+.3f} (p = {rho_dur.pvalue:.3g})")
print(f"  late rate vs share of same-state orders: Spearman rho = {rho_same.statistic:+.3f} (p = {rho_same.pvalue:.3g})")
print(f"  late rate vs mean promised delivery days: Spearman rho = {rho_prom.statistic:+.3f} (p = {rho_prom.pvalue:.3g})")
sr[["customer_state", "n_orders", "late_rate", "median_duration", "mean_promise_days", "same_state_share"]].head(8).round(3)
''')

md("s2_interp")

# =============================================================================
# 3. RQ2 category
# =============================================================================
md("s3_intro")

code('''
cp = an.category_performance(d_cat, min_orders=MIN_N).sort_values("late_rate", ascending=False).reset_index(drop=True)
rel_cat = cp[cp["reliable"]].reset_index(drop=True)
print(f"{len(cp)} product categories observed (the {N_UNKNOWN:,} orders with a missing category are not a category and are left out); "
      f"{len(rel_cat)} have n >= {MIN_N}; {len(cp) - len(rel_cat)} are small samples and excluded from rankings.")
ext = pd.concat([rel_cat.head(10), rel_cat.tail(10)]).reset_index(drop=True)
view = ext[["product_category", "n_orders", "n_late", "late_rate", "late_ci_low", "late_ci_high", "median_duration", "mean_freight"]].copy()
for c in ["late_rate", "late_ci_low", "late_ci_high"]:
    view[c] = (100 * view[c]).round(2)
view.rename(columns={"late_rate": "late_%", "late_ci_low": "CI_low_%", "late_ci_high": "CI_high_%"})
''')

code('''
# ---- Figure 22: extremes of the category ranking (reliable categories only) -------------------------------------
fig, ax = plt.subplots(figsize=(8.8, 7.2))
forest(ax, ext, "product_category", BASE_CAT, label_fmt=lambda s: s.replace("_", " "))
ax.axhline(9.5, color=P["border"], lw=1)
ax.set_xlabel("Late-delivery rate (%), with Wilson 95% CI")
ax.set_title(f"Product categories: 10 highest and 10 lowest late rates (of {len(rel_cat)} with n >= {MIN_N})")
ax.plot([], [], "o", color=P["red"], label="interval above overall"); ax.plot([], [], "o", color=P["green"], label="interval below overall")
ax.plot([], [], "o", color=P["slate_light"], label=f"interval includes overall ({pct(BASE_CAT)}, dotted line)")
ax.legend(loc="lower right", fontsize=8)
save_show(fig, "22_category_late_rate_extremes")
''')

code('''
# ---- chi-square result, residual leaders, and category-vs-rest with Holm ----------------------------------------
s = CHI["product_category"]
print(f"chi2 = {s['chi2']:.1f}, dof = {s['dof']}, p = {fmt_p(s['p_value'])}, n = {s['n']:,}, levels tested = {s['n_levels_tested']} "
      f"(of {s['n_levels_original']}), Cramer's V = {s['cramers_v']:.4f} [{s['effect_label']}], bias-corrected V = {s['cramers_v_corrected']:.4f} "
      f"(bootstrap 95% CI {VCI['product_category'][0]:.3f} to {VCI['product_category'][1]:.3f})")
display(s["level_summary"].head(8).assign(late_rate=lambda t: (100 * t.late_rate).round(2)).rename(columns={"late_rate": "late_%"}).round(2))

cat_vs_rest = vs_rest(cp, "product_category", total_k=K_CAT, total_n=N_CAT)
mc = len(cat_vs_rest)
print(f"\\n{mc} categories with n >= {MIN_N} compared with all other orders of known category: {int(cat_vs_rest.sig_raw.sum())} differ at unadjusted p < {ALPHA}; "
      f"{int(cat_vs_rest.sig_holm.sum())} survive Holm adjustment for {mc} comparisons.")
tbl = cat_vs_rest[cat_vs_rest.sig_holm].copy()
tbl["p_raw"] = tbl["p_raw"].map(fmt_p); tbl["p_holm"] = tbl["p_holm"].map(fmt_p)
display(tbl.round(2))
''')

code('''
# ---- confounding probe: does the category association survive inside the biggest state (Sao Paulo customers)? ---
sp_only = d[d["customer_state"] == "SP"]
s_sp = sa.chi_square_test(sp_only[sp_only["product_category"] != UNKNOWN_CAT], "product_category", "late")
print(f"SP customers only (n = {s_sp['n']:,}): chi2 = {s_sp['chi2']:.1f}, dof = {s_sp['dof']}, p = {fmt_p(s_sp['p_value'])}, "
      f"Cramer's V = {s_sp['cramers_v']:.4f} [{s_sp['effect_label']}]   (all states: V = {CHI['product_category']['cramers_v']:.4f})")
cat_state_mix = (d_cat.groupby("product_category", observed=True)["same_state"].mean().rename("same_state_share").to_frame()
                 .join(cp.set_index("product_category")["n_orders"]).query("n_orders >= @MIN_N"))
print(f"Same-state share varies across categories from {pct(cat_state_mix.same_state_share.min())} to {pct(cat_state_mix.same_state_share.max())} "
      f"- categories differ in seller/customer geography, which is a confounder for any category comparison.")
''')

md("s3_interp")

# =============================================================================
# 4. RQ3 same vs cross state
# =============================================================================
md("s4_intro")

code('''
ss = an.same_state_comparison(d, min_orders=MIN_N).set_index("group")
same, cross = ss.loc["same_state"], ss.loc["cross_state"]
k1, n1, k0, n0 = int(same.n_late), int(same.n_orders), int(cross.n_late), int(cross.n_orders)
view = ss[["n_orders", "n_late", "late_rate", "late_ci_low", "late_ci_high", "median_duration", "mean_duration"]].copy()
for c in ["late_rate", "late_ci_low", "late_ci_high"]:
    view[c] = (100 * view[c]).round(2)
display(view.rename(columns={"late_rate": "late_%", "late_ci_low": "CI_low_%", "late_ci_high": "CI_high_%"}))

# proportion difference: src (Newcombe) vs independent Newcombe + pooled z + scipy chi-square 2x2
diff = sa.proportion_diff_ci(k1, n1, k0, n0)
def wil(k, n, z=ZC):
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h
l1, u1 = wil(k1, n1); l0, u0 = wil(k0, n0); p1, p0 = k1 / n1, k0 / n0
nc_lo = (p1 - p0) - math.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2)
nc_hi = (p1 - p0) + math.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)
pooled = (k1 + k0) / (n1 + n0)
z_manual = (p1 - p0) / math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n0))
zt = sa.two_proportion_ztest(k1, n1, k0, n0)
tab2 = np.array([[k1, n1 - k1], [k0, n0 - k0]])
chi2_2x2, p_2x2, _, _ = stats.chi2_contingency(tab2, correction=False)
print(f"same-state minus cross-state late rate = {100 * diff['diff']:+.2f} pp; Newcombe 95% CI (src) {100 * diff['ci_low']:+.2f} to {100 * diff['ci_high']:+.2f}; "
      f"(by hand) {100 * nc_lo:+.2f} to {100 * nc_hi:+.2f}")
print(f"pooled z (src) = {zt['z']:+.3f}; (by hand) = {z_manual:+.3f}; z^2 = {z_manual ** 2:.3f}  vs  scipy chi-square (2x2, no correction) = {chi2_2x2:.3f}")
assert abs(diff["ci_low"] - nc_lo) < 1e-9 and abs(diff["ci_high"] - nc_hi) < 1e-9 and abs(z_manual ** 2 - chi2_2x2) < 1e-6
assert abs(zt["z"] - z_manual) < 1e-9 and abs(MAN["same_state"]["chi2"] - chi2_2x2) < 1e-6

# effect measures: risk ratio and odds ratio with CIs, Fisher exact test
rr = p1 / p0
se_rr = math.sqrt(1 / k1 - 1 / n1 + 1 / k0 - 1 / n0)
OR_crude, or_lo, or_hi = or_table(k1, n1 - k1, k0, n0 - k0)
fisher = stats.fisher_exact(tab2)
print(f"risk ratio (same / cross) = {rr:.3f} (95% CI {math.exp(math.log(rr) - ZC * se_rr):.3f} to {math.exp(math.log(rr) + ZC * se_rr):.3f}); "
      f"odds ratio = {OR_crude:.3f} (95% CI {or_lo:.3f} to {or_hi:.3f}); Fisher exact p = {fmt_p(max(fisher.pvalue, P_FLOOR))}, scipy OR = {fisher.statistic:.3f}")
''')

code('''
# ---- delivery duration: Mann-Whitney (src and scipy), with a bootstrap CI for the difference in medians -----------
mw = sa.kruskal_or_mannwhitney(d, "delivery_duration_days", "same_state")
x_cross = d.loc[d.same_state == 0, "delivery_duration_days"].to_numpy()
x_same = d.loc[d.same_state == 1, "delivery_duration_days"].to_numpy()
u, p_u = stats.mannwhitneyu(x_cross, x_same, alternative="two-sided")
rb = 2 * u / (len(x_cross) * len(x_same)) - 1            # rank-biserial: > 0 means cross-state tends to be LONGER
print(f"Mann-Whitney (src): U = {mw['statistic']:.0f}, p = {fmt_p(mw['p_value'])}, rank-biserial = {mw['effect_size']:+.3f} [{mw['effect_label']}] "
      f"(first level = same_state=0 i.e. cross-state)")
print(f"Mann-Whitney (scipy direct): U = {u:.0f}, p = {fmt_p(max(p_u, P_FLOOR))}, rank-biserial = {rb:+.3f}")
assert abs(u - mw["statistic"]) < 1e-6 and abs(rb - mw["effect_size"]) < 1e-9
rng = np.random.default_rng(SEED)
B = 500
dm = np.array([np.median(rng.choice(x_cross, len(x_cross))) - np.median(rng.choice(x_same, len(x_same))) for _ in range(B)])
MED_DIFF = float(np.median(x_cross) - np.median(x_same))
MED_DIFF_CI = (float(np.percentile(dm, 2.5)), float(np.percentile(dm, 97.5)))
print(f"median duration: cross-state {np.median(x_cross):.2f} d (n={len(x_cross):,}) vs same-state {np.median(x_same):.2f} d (n={len(x_same):,}); "
      f"difference {MED_DIFF:+.2f} d, bootstrap 95% CI {MED_DIFF_CI[0]:+.2f} to {MED_DIFF_CI[1]:+.2f} ({B} resamples)")
log_test("RQ3", "Mann-Whitney duration by same_state", "mann-whitney", mw["n"], mw["p_value"], "rank-biserial", mw["effect_size"], mw["effect_label"])
''')

md("s4_confound")

code('''
# ---- Mantel-Haenszel: same_state x late, stratified by customer_state (adjusts for destination region) ----------
strata = []
for s_, g in d.groupby("customer_state", observed=True):
    a_ = int(((g.same_state == 1) & (g.late == 1)).sum()); b_ = int(((g.same_state == 1) & (g.late == 0)).sum())
    c_ = int(((g.same_state == 0) & (g.late == 1)).sum()); d_ = int(((g.same_state == 0) & (g.late == 0)).sum())
    strata.append((s_, a_, b_, c_, d_))
S = pd.DataFrame(strata, columns=["customer_state", "a_same_late", "b_same_ok", "c_cross_late", "d_cross_ok"])
S["n"] = S[["a_same_late", "b_same_ok", "c_cross_late", "d_cross_ok"]].sum(axis=1)
S["n_same"] = S.a_same_late + S.b_same_ok; S["n_cross"] = S.c_cross_late + S.d_cross_ok
inform = S[(S.n_same > 0) & (S.n_cross > 0)].copy()                     # strata that contain both groups
R = (inform.a_same_late * inform.d_cross_ok / inform.n).sum()
Sx = (inform.b_same_ok * inform.c_cross_late / inform.n).sum()
OR_MH = R / Sx
# Robins-Breslow-Greenland variance of log(OR_MH)
P_ = (inform.a_same_late + inform.d_cross_ok) / inform.n; Q_ = (inform.b_same_ok + inform.c_cross_late) / inform.n
Rk = inform.a_same_late * inform.d_cross_ok / inform.n; Sk = inform.b_same_ok * inform.c_cross_late / inform.n
var = (P_ * Rk).sum() / (2 * R ** 2) + ((P_ * Sk + Q_ * Rk).sum()) / (2 * R * Sx) + (Q_ * Sk).sum() / (2 * Sx ** 2)
OR_MH_CI = (math.exp(math.log(OR_MH) - ZC * math.sqrt(var)), math.exp(math.log(OR_MH) + ZC * math.sqrt(var)))
# MH chi-square test of OR = 1 (no continuity correction)
E_a = (inform.n_same * (inform.a_same_late + inform.c_cross_late) / inform.n)
V_a = (inform.n_same * inform.n_cross * (inform.a_same_late + inform.c_cross_late) * (inform.b_same_ok + inform.d_cross_ok)
       / (inform.n ** 2 * (inform.n - 1)))
chi2_mh = (inform.a_same_late.sum() - E_a.sum()) ** 2 / V_a.sum()
p_mh = max(float(stats.chi2.sf(chi2_mh, 1)), P_FLOOR)
print(f"{len(inform)} of {len(S)} customer states contain both same-state and cross-state orders.")
print(f"Crude odds ratio (late, same vs cross)            : {OR_crude:.3f}  (95% CI {or_lo:.3f} to {or_hi:.3f})")
print(f"Mantel-Haenszel odds ratio, adjusted for state     : {OR_MH:.3f}  (95% CI {OR_MH_CI[0]:.3f} to {OR_MH_CI[1]:.3f}); MH chi2 = {chi2_mh:.1f}, p = {fmt_p(p_mh)}")
print(f"Share of the crude log-odds difference that remains after adjustment: {math.log(OR_MH) / math.log(OR_crude):.1%}")

big = inform[(inform.n_same >= MIN_N) & (inform.n_cross >= MIN_N)].copy()
orw = big.apply(lambda r: pd.Series(or_table(r.a_same_late, r.b_same_ok, r.c_cross_late, r.d_cross_ok), index=["OR", "lo", "hi"]), axis=1)
big = pd.concat([big, orw], axis=1)
big["late_same_%"] = 100 * big.a_same_late / big.n_same; big["late_cross_%"] = 100 * big.c_cross_late / big.n_cross
big = big.sort_values("n", ascending=False).reset_index(drop=True)
print(f"\\nStrata with n >= {MIN_N} in BOTH groups ({len(big)} states): same-state odds ratio below 1 in {(big.OR < 1).sum()}, above 1 in {(big.OR > 1).sum()}")
display(big[["customer_state", "n_same", "n_cross", "late_same_%", "late_cross_%", "OR", "lo", "hi"]].round(3))
''')

code('''
# ---- Figure 23: same vs cross state: late rate, duration distributions, and state-stratified odds ratios --------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [0.8, 1.2, 1.2]})
ax = axes[0]
labs = [f"Same state\\n(n={n1:,})", f"Cross state\\n(n={n0:,})"]
rates = np.array([p1, p0]) * 100
err = np.array([[p1 - l1, p0 - l0], [u1 - p1, u0 - p0]]) * 100
ax.bar(labs, rates, yerr=err, color=[P["blue"], P["slate_light"]], width=0.55, capsize=5)
for i, r_ in enumerate(rates):
    ax.text(i, r_ + err[1, i] + 0.2, f"{r_:.2f}%", ha="center", fontsize=9, color=P["navy"])
ax.axhline(BASE * 100, color=P["navy"], ls=":", lw=1.1, label=f"overall {pct(BASE)}")
ax.set_ylabel("Late-delivery rate (%)"); ax.set_title("Late rate (Wilson 95% CI)"); ax.legend(); ax.grid(axis="x", visible=False)
ax.set_ylim(0, max(rates + err[1]) * 1.2)

ax = axes[1]
bins = np.linspace(0, 60, 61)
ax.hist(np.clip(x_same, 0, 60), bins=bins, density=True, alpha=0.65, color=P["blue"], label=f"same state (median {np.median(x_same):.1f} d)")
ax.hist(np.clip(x_cross, 0, 60), bins=bins, density=True, alpha=0.55, color=P["slate_light"], label=f"cross state (median {np.median(x_cross):.1f} d)")
ax.set_xlabel("Delivery duration (days; > 60 shown at 60)"); ax.set_ylabel("Density")
ax.set_title("Delivery duration by seller-customer geography"); ax.legend()

ax = axes[2]
bg = big.sort_values("OR").reset_index(drop=True)
yy = np.arange(len(bg))
ax.hlines(yy, bg.lo, bg.hi, color=P["slate_light"], lw=1.8)
ax.scatter(bg.OR, yy, color=[P["red"] if o > 1 else P["green"] for o in bg.OR], s=34, zorder=3)
ax.axvline(1, color=P["navy"], lw=1)
ax.axvline(OR_MH, color=P["blue"], ls="--", lw=1.4, label=f"Mantel-Haenszel pooled OR = {OR_MH:.2f}")
ax.axvline(OR_crude, color=P["amber"], ls=":", lw=1.6, label=f"crude OR = {OR_crude:.2f}")
ax.set_xscale("log"); ax.set_yticks(yy); ax.set_yticklabels([f"{s_} (n={int(n_):,})" for s_, n_ in zip(bg.customer_state, bg.n)], fontsize=8)
ax.set_xlabel("Odds ratio of late delivery, same vs cross state (log scale)\\n< 1 = same-state orders are late less often")
ax.set_title(f"Within each customer state (n >= {MIN_N} in both groups)"); ax.legend(fontsize=8, loc="upper left")
ax.grid(axis="y", visible=False)
save_show(fig, "23_same_vs_cross_state")
log_test("RQ3", "chi-square-2x2 / Mantel-Haenszel same_state x late | customer_state", "mantel-haenszel", int(inform.n.sum()), p_mh,
         "MH odds ratio", OR_MH, "adjusted for state")
''')

code('''
# ---- Is the same-state difference STABLE over time?  The pooled numbers above average over very different periods. ----
# Windows are the chronological train / validation / test windows of the prediction model (models/model_metrics.json),
# so this table can be compared directly with the model card in the app.
WIN = json.loads(config.MODEL_METRICS_PATH.read_text())["split"]["windows"]
ts = pd.to_datetime(d["order_purchase_timestamp"])
rows = []
for wname, (w0, w1) in WIN.items():
    wm = (ts >= pd.Timestamp(w0)) & ((ts < pd.Timestamp(w1)) if w1 else True)
    g1, g0 = d[wm & (d.same_state == 1)], d[wm & (d.same_state == 0)]
    k1_, n1_, k0_, n0_ = int(g1.late.sum()), len(g1), int(g0.late.sum()), len(g0)
    dw_ = sa.proportion_diff_ci(k1_, n1_, k0_, n0_)
    rows.append({"window": wname, "from": w0, "to (excl.)": w1 or "end of data", "n same-state": n1_, "late % same-state": 100 * k1_ / n1_,
                 "n cross-state": n0_, "late % cross-state": 100 * k0_ / n0_, "difference (pp)": 100 * dw_["diff"],
                 "CI low (pp)": 100 * dw_["ci_low"], "CI high (pp)": 100 * dw_["ci_high"]})
STAB = pd.DataFrame(rows).set_index("window")
display(STAB.round(2))

# the same comparison month by month (2017-01 onwards; every month has n >= MIN_N in both groups, checked below)
mo = (d[d.ym >= "2017-01"].groupby(["ym", "same_state"]).late.agg(["mean", "size"]).unstack("same_state"))
mo.columns = [f"{a}_{'same' if b == 1 else 'cross'}" for a, b in mo.columns]
mo["gap_pp"] = 100 * (mo["mean_same"] - mo["mean_cross"])
assert (mo[["size_same", "size_cross"]] >= MIN_N).all().all(), "a month has fewer than MIN_N orders in one group"
N_MONTHS, N_MONTHS_SAME_HIGHER = len(mo), int((mo.gap_pp > 0).sum())
print(f"Month by month from 2017-01 ({N_MONTHS} months, n >= {MIN_N} in both groups): same-state orders have the HIGHER late rate in "
      f"{N_MONTHS_SAME_HIGHER} of {N_MONTHS} months: {', '.join(mo.index[mo.gap_pp > 0])}.")
display((mo.loc[mo.gap_pp > 0, ["mean_same", "size_same", "mean_cross", "size_cross", "gap_pp"]]
         .assign(mean_same=lambda t: 100 * t.mean_same, mean_cross=lambda t: 100 * t.mean_cross)
         .rename(columns={"mean_same": "late % same", "size_same": "n same", "mean_cross": "late % cross", "size_cross": "n cross", "gap_pp": "gap (pp)"})).round(2))

fig, ax = plt.subplots(figsize=(11, 3.8))
xm_ = np.arange(len(mo))
ax.plot(xm_, 100 * mo.mean_cross, "o-", color=P["slate_light"], lw=2, ms=4, label="cross-state orders")
ax.plot(xm_, 100 * mo.mean_same, "o-", color=P["blue"], lw=2, ms=4, label="same-state orders")
ax.set_xticks(xm_); ax.set_xticklabels(mo.index, rotation=60, ha="right", fontsize=8)
ax.set_xlabel("Purchase month"); ax.set_ylabel("Late-delivery rate (%)")
ax.set_title("Same-state vs cross-state late rate by purchase month: the ordering is not constant")
ax.legend(); pct_axis(ax, "y"); plt.tight_layout(); plt.show()
''')

md("s4_interp")

# =============================================================================
# 5. RQ4 weight / freight
# =============================================================================
md("s5_intro")

code('''
pairs = [("total_weight_g", "delivery_duration_days"), ("total_freight", "delivery_duration_days"),
         ("freight_ratio", "delivery_duration_days"), ("estimated_delivery_days", "delivery_duration_days"),
         ("total_weight_g", "total_freight")]
SPEAR = {}
for x_, y_ in pairs:
    SPEAR[(x_, y_)] = spearman_full(d, x_, y_, B=200 if y_ == "delivery_duration_days" else 0)

rows = []
for (x_, y_), r_ in SPEAR.items():
    srcr = sa.spearman_test(d, x_, y_)
    rows.append({"x": x_, "y": y_, "n pairs": r_["n"], "rho (scipy)": r_["rho"], "rho (src)": srcr["rho"],
                 "rho (Pearson on ranks)": r_["rho_pearson_on_ranks"],
                 "Fisher-z 95% CI (src)": f"{srcr['ci_low']:.3f} to {srcr['ci_high']:.3f}",
                 "Fisher-z CI (hand, classic SE)": f"{r_['ci_classic'][0]:.3f} to {r_['ci_classic'][1]:.3f}",
                 "bootstrap 95% CI": (f"{r_['ci_boot'][0]:.3f} to {r_['ci_boot'][1]:.3f}" if "ci_boot" in r_ else "-"),
                 "p": fmt_p(r_["p"]), "strength": strength(r_["rho"])})
    assert abs(srcr["rho"] - r_["rho"]) < 1e-10 and abs(r_["rho"] - r_["rho_pearson_on_ranks"]) < 1e-10
    assert abs(srcr["ci_low"] - r_["ci_bw"][0]) < 1e-9
    if y_ == "delivery_duration_days":
        log_test("RQ4", f"Spearman {x_} ~ {y_}", "spearman", r_["n"], r_["p"], "Spearman rho", r_["rho"], strength(r_["rho"]))
display(pd.DataFrame(rows))
print("scipy, src and Pearson-on-ranks agree to 1e-10; src Fisher-z limits equal the Bonett-Wright interval recomputed by hand.")
''')

code('''
# ---- relationships with the binary outcome (rank association with is_late; no normal-theory CI) + partial rho ----
rows = []
for x_ in ["total_weight_g", "total_freight", "freight_ratio", "estimated_delivery_days"]:
    r_ = sa.spearman_test(d, x_, "late")
    rows.append({"x": x_, "y": "is_late (0/1)", "n pairs": r_["n"], "rho": r_["rho"], "p": fmt_p(r_["p_value"]), "strength": r_["strength_label"]})
    log_test("RQ4", f"Spearman {x_} ~ is_late", "spearman", r_["n"], r_["p_value"], "Spearman rho", r_["rho"], r_["strength_label"])
display(pd.DataFrame(rows))

def partial_spearman(frame, x, y, z):
    f = frame[[x, y, z]].astype(float).dropna()
    rx, ry, rz = (stats.rankdata(f[c]) for c in (x, y, z))
    rxy, rxz, ryz = (np.corrcoef(a, b)[0, 1] for a, b in ((rx, ry), (rx, rz), (ry, rz)))
    return (rxy - rxz * ryz) / math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2)), len(f)

PART = {}
PART["freight|weight"] = partial_spearman(d, "total_freight", "delivery_duration_days", "total_weight_g")
PART["weight|freight"] = partial_spearman(d, "total_weight_g", "delivery_duration_days", "total_freight")
print(f"Partial Spearman, freight ~ duration controlling for weight : {PART['freight|weight'][0]:+.3f} (n = {PART['freight|weight'][1]:,})")
print(f"Partial Spearman, weight ~ duration controlling for freight : {PART['weight|freight'][0]:+.3f}")
STRAT = {}
for g_, name in [(1, "same state"), (0, "cross state")]:
    sub = d[d.same_state == g_]
    STRAT[name] = (sa.spearman_test(sub, "total_freight", "delivery_duration_days")["rho"],
                   sa.spearman_test(sub, "total_weight_g", "delivery_duration_days")["rho"], len(sub))
    print(f"Within {name:12s} (n = {len(sub):,}): freight ~ duration rho = {STRAT[name][0]:+.3f}, weight ~ duration rho = {STRAT[name][1]:+.3f}")
''')

code('''
# ---- Figure 24: weight / freight vs duration (hexbin) + binned medians, late rate ---------------------------------
wb = an.weight_bins(d, min_orders=MIN_N)
fb = an.freight_bins(d, min_orders=MIN_N)
fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))
ax = axes[0, 0]
m_ = d["total_weight_g"].notna() & (d["total_weight_g"] > 0)
hb = ax.hexbin(np.log10(d.loc[m_, "total_weight_g"]), np.clip(d.loc[m_, "delivery_duration_days"], 0, 60), gridsize=45, bins="log", cmap="Blues", mincnt=1)
ax.set_xlabel("Total order weight (g, log10 scale)"); ax.set_ylabel("Delivery duration (days; > 60 shown at 60)")
ax.set_xticks([1, 2, 3, 4, 5]); ax.set_xticklabels(["10", "100", "1,000", "10,000", "100,000"])
r_w = SPEAR[("total_weight_g", "delivery_duration_days")]["rho"]
ax.set_title(f"Weight vs duration: Spearman rho = {r_w:+.2f} (n={SPEAR[('total_weight_g', 'delivery_duration_days')]['n']:,})")
fig.colorbar(hb, ax=ax, label="orders per cell (log)")
ax = axes[0, 1]
hb = ax.hexbin(np.log10(d["total_freight"].clip(lower=1)), np.clip(d["delivery_duration_days"], 0, 60), gridsize=45, bins="log", cmap="Blues", mincnt=1)
ax.set_xlabel("Total freight (R$, log10 scale; values < R$1 shown at R$1)"); ax.set_ylabel("Delivery duration (days; > 60 shown at 60)")
ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels(["1", "10", "100", "1,000"])
r_f = SPEAR[("total_freight", "delivery_duration_days")]["rho"]
ax.set_title(f"Freight vs duration: Spearman rho = {r_f:+.2f} (n={SPEAR[('total_freight', 'delivery_duration_days')]['n']:,})")
fig.colorbar(hb, ax=ax, label="orders per cell (log)")

def binned_panel(ax, tab, xlabel, title):
    tt = tab.reset_index(drop=True)
    x = np.arange(len(tt))
    ax.bar(x, 100 * tt.late_rate, color=P["red"], alpha=0.8, width=0.6,
           yerr=[100 * (tt.late_rate - tt.late_ci_low), 100 * (tt.late_ci_high - tt.late_rate)], capsize=3, label="late rate (Wilson 95% CI)")
    ax.axhline(BASE * 100, color=P["navy"], ls=":", lw=1.1)
    ax.set_xticks(x); ax.set_xticklabels([f"{b}\\nn={int(n):,}" for b, n in zip(tt.bin, tt.n_orders)], fontsize=7, rotation=25, ha="right")
    ax.set_xlabel(xlabel); ax.set_ylabel("Late-delivery rate (%)"); ax.set_title(title); ax.grid(axis="x", visible=False); ax.set_ylim(0, 100 * tt.late_ci_high.max() * 1.45)
    a2 = ax.twinx(); a2.plot(x, tt.median_duration, "o-", color=P["blue"], lw=2, label="median duration (days)")
    a2.set_ylim(0, tt.median_duration.max() * 1.25); a2.set_ylabel("Median delivery duration (days)", color=P["blue"]); a2.grid(False); a2.spines["right"].set_visible(True)
    h1, l1_ = ax.get_legend_handles_labels(); h2, l2_ = a2.get_legend_handles_labels(); ax.legend(h1 + h2, l1_ + l2_, loc="upper left", fontsize=8)
binned_panel(axes[1, 0], wb, "Total order weight bin (g)", "Late rate and median duration by weight bin")
binned_panel(axes[1, 1], fb, "Total freight bin (R$)", "Late rate and median duration by freight bin")
save_show(fig, "24_weight_freight_vs_duration")
''')

code('''
# ---- Figure 25: Spearman rho with Fisher-z and bootstrap confidence intervals --------------------------------------
cont = [(k, v) for k, v in SPEAR.items() if "ci_boot" in v]
fig, ax = plt.subplots(figsize=(9.5, 3.9))
for i, ((x_, y_), r_) in enumerate(cont[::-1]):
    ax.hlines(i + 0.12, *r_["ci_bw"], color=P["blue"], lw=3, label="Fisher-z 95% CI (Bonett-Wright SE)" if i == 0 else None)
    ax.hlines(i - 0.12, *r_["ci_boot"], color=P["amber"], lw=3, label="bootstrap 95% CI (200 resamples)" if i == 0 else None)
    ax.scatter([r_["rho"]], [i], color=P["navy"], zorder=3, s=30)
    ax.text(r_["rho"] + 0.012, i, f"rho = {r_['rho']:+.3f}", ha="left", va="center", fontsize=8.5, color=P["navy"])
ax.set_yticks(range(len(cont))); ax.set_yticklabels([f"{x_.replace('_', ' ')} ~ duration\\n(n={r_['n']:,})" for (x_, _), r_ in cont[::-1]], fontsize=8.5)
ax.axvline(0, color=P["navy"], lw=1)
ax.axvspan(-0.1, 0.1, color=P["border"], alpha=0.6, label="|rho| < 0.1 (negligible)")
ax.set_xlim(-0.13, 0.68); ax.set_ylim(-0.6, len(cont) - 0.4)
ax.set_xlabel("Spearman rank correlation with delivery duration (descriptive association, not causal)")
ax.set_title("Rank correlations with delivery duration and their uncertainty"); ax.legend(loc="upper right", fontsize=8)
ax.grid(axis="y", visible=False)
save_show(fig, "25_spearman_confidence_intervals")
''')

md("s5_interp")

# =============================================================================
# 6. RQ5 time
# =============================================================================
md("s6_intro")

code('''
mp = an.monthly_performance(d, min_orders=MIN_N)
rel_m = mp[mp["reliable"]].reset_index(drop=True)
print(f"{len(mp)} purchase months in the data; {len(rel_m)} have n >= {MIN_N} ({', '.join(mp.loc[~mp.reliable, 'purchase_year_month'])} excluded as tiny samples).")
view = rel_m[["purchase_year_month", "n_orders", "n_late", "late_rate", "late_ci_low", "late_ci_high", "mean_duration"]].copy()
for c in ["late_rate", "late_ci_low", "late_ci_high"]:
    view[c] = (100 * view[c]).round(2)
view.rename(columns={"late_rate": "late_%", "late_ci_low": "CI_low_%", "late_ci_high": "CI_high_%"})
''')

code('''
# ---- spike months: identified from the monthly CI screen (interval excludes the overall rate AND rate far above) --
mon = rel_m.copy()
mon["above"] = mon.late_ci_low > BASE
SPIKE_FACTOR = 1.5
spike_df = mon[mon.late_ci_low >= SPIKE_FACTOR * BASE]           # whole Wilson interval at least 1.5x the overall rate
SPIKES = spike_df["purchase_year_month"].tolist()
print(f"Months whose Wilson interval lies entirely at or above {SPIKE_FACTOR}x the overall rate ({pct(SPIKE_FACTOR * BASE)}):", SPIKES)
spike_mask = d.ym.isin(SPIKES)
base_win = d[(d.ym >= "2017-01") & ~spike_mask]
ks, ns_ = int(d.loc[spike_mask, "late"].sum()), int(spike_mask.sum())
ko, no = int(base_win["late"].sum()), len(base_win)
SPK = sa.proportion_diff_ci(ks, ns_, ko, no); SPKZ = sa.two_proportion_ztest(ks, ns_, ko, no)
print(f"Spike months: {ks:,}/{ns_:,} late = {pct(ks / ns_)}   other months from 2017-01: {ko:,}/{no:,} = {pct(ko / no)}   "
      f"difference {100 * SPK['diff']:+.1f} pp (Newcombe 95% CI {100 * SPK['ci_low']:+.1f} to {100 * SPK['ci_high']:+.1f}), p {fmt_p(SPKZ['p_value'])}")
print(f"Spike months hold {ns_ / N:.1%} of all orders but {ks / K_LATE:.1%} of all late orders.")

# ---- Figure 26: monthly late rate with Wilson CI and order volume --------------------------------------------------
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11.5, 6.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.15]})
# calendar axis: months that are absent or below the n threshold stay as gaps (NaN), so the line is not drawn across them
cal = pd.period_range(rel_m.purchase_year_month.iloc[0], rel_m.purchase_year_month.iloc[-1], freq="M").astype(str)
rm = rel_m.set_index("purchase_year_month").reindex(cal)
xm = np.arange(len(cal))
gap_months = [m_ for m_ in cal if m_ not in set(rel_m.purchase_year_month)]
ax.fill_between(xm, 100 * rm.late_ci_low, 100 * rm.late_ci_high, color=P["red"], alpha=0.18, label="Wilson 95% CI")
ax.vlines(xm, 100 * rm.late_ci_low, 100 * rm.late_ci_high, color=P["red"], alpha=0.45, lw=1)   # also shows isolated months
ax.plot(xm, 100 * rm.late_rate, "o-", color=P["red"], lw=2, ms=4, label="monthly late rate")
ax.axhline(BASE * 100, color=P["navy"], ls=":", lw=1.2, label=f"overall {pct(BASE)}")
for i, m_ in enumerate(cal):
    if m_ in SPIKES:
        ax.axvspan(i - 0.5, i + 0.5, color=P["amber"], alpha=0.25)
        ax.annotate(f"{m_}\\n{100 * rm.late_rate.iloc[i]:.1f}%", (i, 100 * rm.late_rate.iloc[i]), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, color=P["navy"])
ax.set_ylabel("Late-delivery rate (%)"); ax.set_title("Late rate by purchase month (months with n >= %d; shaded = interval entirely above 1.5x the overall rate)" % MIN_N)
pct_axis(ax, "y"); ax.set_ylim(0, 100 * rel_m.late_ci_high.max() * 1.18); ax.legend(loc="upper left", fontsize=8.5)
if gap_months:
    ax.text(0.01, 0.62, "gap = no month with n >= %d: %s" % (MIN_N, ", ".join(gap_months)), transform=ax.transAxes, fontsize=8, color=P["slate"])
ax2.bar(xm, rm.n_orders.fillna(0), color=P["blue_light"], width=0.7)
ax2.set_ylabel("Orders (n)"); ax2.set_xlabel("Purchase month"); ax2.set_xticks(xm); ax2.set_xticklabels(cal, rotation=60, ha="right", fontsize=8)
ax2.grid(axis="x", visible=False)
save_show(fig, "26_monthly_late_rate")
''')

code('''
# ---- chi-square for time: month-of-year (pooled over years; src) vs a proper year-month test ---------------------
s_m = CHI["purchase_month"]
print(f"Month of year (pooled across years): chi2 = {s_m['chi2']:.1f}, dof = {s_m['dof']}, p = {fmt_p(s_m['p_value'])}, V = {s_m['cramers_v']:.3f} [{s_m['effect_label']}], "
      f"bias-corrected V bootstrap 95% CI {VCI['purchase_month'][0]:.3f} to {VCI['purchase_month'][1]:.3f}")
ym_frame = d[d.ym.isin(rel_m.purchase_year_month)]
ym_chi = sa.chi_square_test(ym_frame, "ym", "late"); ym_man = manual_chi2(ym_frame, "ym", "late")
assert abs(ym_chi["chi2"] - ym_man["chi2"]) < 1e-6
print(f"Year-month ({ym_chi['n_levels_tested']} months with n >= {MIN_N}): chi2 = {ym_chi['chi2']:.1f}, dof = {ym_chi['dof']}, p = {fmt_p(ym_chi['p_value'])}, "
      f"V = {ym_chi['cramers_v']:.3f} [{ym_chi['effect_label']}] (hand-computed chi2 agrees)")
dow = sa.chi_square_test(d, "purchase_dow", "late")
dow_tab = an.binned_stat(d, "purchase_dow", [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], quantile=False, min_orders=MIN_N)
dow_tab["bin"] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][: len(dow_tab)]
print(f"Day of week: chi2 = {dow['chi2']:.1f}, dof = {dow['dof']}, p = {fmt_p(dow['p_value'])}, V = {dow['cramers_v']:.3f} [{dow['effect_label']}]")
log_test("RQ5", "chi-square year-month x late", "chi-square", ym_chi["n"], ym_chi["p_value"], "Cramer's V", ym_chi["cramers_v"], ym_chi["effect_label"])
log_test("RQ5", "chi-square purchase_dow x late", "chi-square", dow["n"], dow["p_value"], "Cramer's V", dow["cramers_v"], dow["effect_label"])
display(dow_tab[["bin", "n_orders", "n_late", "late_rate", "late_ci_low", "late_ci_high"]].assign(
    late_rate=lambda t: (100 * t.late_rate).round(2), late_ci_low=lambda t: (100 * t.late_ci_low).round(2),
    late_ci_high=lambda t: (100 * t.late_ci_high).round(2)).rename(columns={"bin": "weekday", "late_rate": "late_%", "late_ci_low": "CI_low_%", "late_ci_high": "CI_high_%"}))
''')

code('''
# ---- Figure 27: month-of-year x year late-rate heatmap (shows why pooling months across years is misleading) ------
hm = d.groupby(["purchase_month", "purchase_year"], observed=True)["late"].agg(["size", "sum"]).reset_index()
hm["rate"] = hm["sum"] / hm["size"]
years = sorted(hm.purchase_year.unique())
rate = np.full((12, len(years)), np.nan); cnt = np.zeros((12, len(years)), dtype=int)
for _, r_ in hm.iterrows():
    rate[int(r_.purchase_month) - 1, years.index(r_.purchase_year)] = r_.rate
    cnt[int(r_.purchase_month) - 1, years.index(r_.purchase_year)] = int(r_["size"])
shown = np.where(cnt >= MIN_N, rate, np.nan)
fig, ax = plt.subplots(figsize=(6.6, 7.4))
im = ax.imshow(100 * shown, cmap="Reds", aspect="auto", vmin=0, vmax=np.nanmax(100 * shown))
ax.set_xticks(range(len(years))); ax.set_xticklabels([str(int(y)) for y in years]); ax.set_yticks(range(12))
ax.set_yticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
for i in range(12):
    for j in range(len(years)):
        if cnt[i, j] == 0:
            continue
        txt = f"{100 * rate[i, j]:.1f}%\\nn={cnt[i, j]:,}" if cnt[i, j] >= MIN_N else f"n={cnt[i, j]}\\n(too few)"
        ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                color="white" if np.isfinite(shown[i, j]) and shown[i, j] > 0.6 * np.nanmax(shown) else P["navy"])
ax.set_xlabel("Purchase year"); ax.set_ylabel("Purchase month")
ax.set_title("Late rate by month and year (blank = no data or n < %d)" % MIN_N)
ax.grid(False); fig.colorbar(im, ax=ax, label="Late-delivery rate (%)", fraction=0.05)
save_show(fig, "27_month_year_heatmap")
print("Years with data per calendar month (orders):")
pd.DataFrame(cnt, index=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], columns=[int(y) for y in years])
''')

md("s6_interp")

# =============================================================================
# 7. RQ6 processing time
# =============================================================================
md("s7_intro")

code('''
pt = an.late_by_processing_bucket(d, min_orders=MIN_N)
view = pt[["bin", "n_orders", "n_late", "late_rate", "late_ci_low", "late_ci_high", "mean_duration"]].copy()
for c in ["late_rate", "late_ci_low", "late_ci_high"]:
    view[c] = (100 * view[c]).round(2)
display(view.rename(columns={"bin": "days from purchase to carrier hand-over", "late_rate": "late_%", "late_ci_low": "CI_low_%", "late_ci_high": "CI_high_%"}))

n_proc = int(d.processing_days.notna().sum())
print(f"processing_days available for {n_proc:,} of {N:,} orders ({N - n_proc:,} missing/invalid carrier timestamps are excluded from this section only).")
SP_PROC = {}
for y_ in ["delivery_duration_days"]:
    SP_PROC[y_] = spearman_full(d, "processing_days", y_, B=200)
r_pd = SP_PROC["delivery_duration_days"]
srcp = sa.spearman_test(d, "processing_days", "delivery_duration_days")
assert abs(srcp["rho"] - r_pd["rho"]) < 1e-10
sp_late = sa.spearman_test(d, "processing_days", "late")
sp_c2c = sa.spearman_test(d, "carrier_to_customer_days", "late")
sp_pc = sa.spearman_test(d, "processing_days", "carrier_to_customer_days")
print(f"processing ~ duration : rho = {r_pd['rho']:+.3f}, Fisher-z CI {r_pd['ci_bw'][0]:.3f} to {r_pd['ci_bw'][1]:.3f}, bootstrap CI {r_pd['ci_boot'][0]:.3f} to {r_pd['ci_boot'][1]:.3f}  (partly mechanical: processing is a component of duration)")
print(f"processing ~ is_late  : rho = {sp_late['rho']:+.3f}, p {fmt_p(sp_late['p_value'])} [{sp_late['strength_label']}], n = {sp_late['n']:,}")
print(f"carrier->customer days ~ is_late: rho = {sp_c2c['rho']:+.3f} [{sp_c2c['strength_label']}] (the other component of duration, for comparison)")
print(f"processing ~ carrier->customer days: rho = {sp_pc['rho']:+.3f} (a slow hand-over does not mechanically make the transit leg longer)")

pb = d[d.processing_days.notna()].copy()
pb["bucket"] = pd.cut(pb.processing_days, [-0.001, 1, 2, 3, 5, 7, 14, np.inf], labels=pt["bin"].tolist())
chi_pb = sa.chi_square_test(pb, "bucket", "late")
x_late = pb.loc[pb.late == 1, "processing_days"]; x_ok = pb.loc[pb.late == 0, "processing_days"]
mw_p = stats.mannwhitneyu(x_late, x_ok, alternative="two-sided")
rb_p = 2 * mw_p.statistic / (len(x_late) * len(x_ok)) - 1
print(f"chi-square processing bucket x late: chi2 = {chi_pb['chi2']:.1f}, dof = {chi_pb['dof']}, p = {fmt_p(chi_pb['p_value'])}, V = {chi_pb['cramers_v']:.3f} [{chi_pb['effect_label']}], n = {chi_pb['n']:,}")
print(f"Mann-Whitney processing days late (n={len(x_late):,}, median {x_late.median():.2f} d) vs on time (n={len(x_ok):,}, median {x_ok.median():.2f} d): "
      f"p {fmt_p(max(mw_p.pvalue, P_FLOOR))}, rank-biserial = {rb_p:+.3f}")
log_test("RQ6", "Spearman processing_days ~ is_late (retrospective)", "spearman", sp_late["n"], sp_late["p_value"], "Spearman rho", sp_late["rho"], sp_late["strength_label"])
log_test("RQ6", "Spearman processing_days ~ duration (retrospective)", "spearman", r_pd["n"], r_pd["p"], "Spearman rho", r_pd["rho"], strength(r_pd["rho"]))
log_test("RQ6", "chi-square processing bucket x late (retrospective)", "chi-square", chi_pb["n"], chi_pb["p_value"], "Cramer's V", chi_pb["cramers_v"], chi_pb["effect_label"])
log_test("RQ6", "Mann-Whitney processing_days by late (retrospective)", "mann-whitney", len(x_late) + len(x_ok), max(mw_p.pvalue, P_FLOOR), "rank-biserial", rb_p, sa._rank_biserial_label(rb_p))

# ---- Figure 28 ----
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
x = np.arange(len(pt))
colr = [P["red"] if lo_ > BASE else P["green"] if hi_ < BASE else P["slate_light"] for lo_, hi_ in zip(pt.late_ci_low, pt.late_ci_high)]
ax.bar(x, 100 * pt.late_rate, color=colr, width=0.62,
       yerr=[100 * (pt.late_rate - pt.late_ci_low), 100 * (pt.late_ci_high - pt.late_rate)], capsize=4)
for xi, (r_, n_) in enumerate(zip(pt.late_rate, pt.n_orders)):
    ax.text(xi, 100 * pt.late_ci_high.iloc[xi] + 1.2, f"{100 * r_:.1f}%\\nn={int(n_):,}", ha="center", fontsize=8, color=P["navy"],
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
ax.axhline(BASE * 100, color=P["navy"], ls=":", lw=1.2, label=f"overall {pct(BASE)}")
ax.set_xticks(x); ax.set_xticklabels(pt["bin"]); ax.set_ylim(0, 100 * pt.late_ci_high.max() * 1.25)
ax.set_xlabel("Days from purchase to carrier hand-over (RETROSPECTIVE: known only after the seller ships)")
ax.set_ylabel("Late-delivery rate (%)"); ax.set_title("Late rate by seller processing time (Wilson 95% CI)"); ax.legend(loc="upper left"); ax.grid(axis="x", visible=False)
bp = ax2.boxplot([x_ok.clip(upper=20), x_late.clip(upper=20)], tick_labels=[f"On time\\n(n={len(x_ok):,})", f"Late\\n(n={len(x_late):,})"], showfliers=False,
                 patch_artist=True, widths=0.5, medianprops={"color": P["navy"], "lw": 2})
for patch, c in zip(bp["boxes"], [P["green"], P["red"]]):
    patch.set_facecolor(c); patch.set_alpha(0.55)
ax2.set_ylabel("Processing time (days; > 20 d clipped at 20)"); ax2.set_title("Processing time by outcome (box = IQR)"); ax2.grid(axis="x", visible=False)
save_show(fig, "28_processing_time_vs_lateness")
''')

md("s7_interp")

# =============================================================================
# 8. Sensitivity + sample size
# =============================================================================
md("s8_intro")

code('''
# ---- (a) Sensitivity: repeat the chi-square battery without the spike months ---------------------------------------
d_ns = d[~d.ym.isin(SPIKES)]
rows = []
for col in ["customer_state", "product_category", "same_state", "purchase_month", "purchase_dow"]:
    known = (lambda f: f[f["product_category"] != UNKNOWN_CAT]) if col == "product_category" else (lambda f: f)   # RQ2 excludes 'unknown'
    full = sa.chi_square_test(known(d), col, "late"); ns_r = sa.chi_square_test(known(d_ns), col, "late")
    rows.append({"variable": col, "V (all orders)": full["cramers_v"], "V (without spike months)": ns_r["cramers_v"],
                 "change": ns_r["cramers_v"] / full["cramers_v"] - 1, "p (without spike months)": fmt_p(ns_r["p_value"]),
                 "label (without)": ns_r["effect_label"], "n (without)": ns_r["n"]})
SENS = pd.DataFrame(rows)
SENS["change"] = SENS["change"].map(lambda v: f"{v:+.0%}")
print(f"Without {', '.join(SPIKES)}: n = {len(d_ns):,}, overall late rate = {pct(d_ns.late.mean())} (vs {pct(BASE)} with them).")
SENS
''')

code('''
# ---- (b) Significance depends on n, effect size does not: random subsamples of the data -------------------------------
rng = np.random.default_rng(SEED)
sizes = [200, 500, 1000, 2000, 5000, 20000, N]
REP = 300
rows = []
for n_s in sizes:
    ps, vs = [], []
    reps = 1 if n_s == N else REP
    for _ in range(reps):
        idx = rng.choice(N, n_s, replace=False) if n_s < N else np.arange(N)
        sub = d.iloc[idx]
        t_ = pd.crosstab(sub.same_state, sub.late).reindex(index=[0, 1], columns=[0, 1], fill_value=0).to_numpy()
        if t_.sum(axis=0).min() == 0 or t_.sum(axis=1).min() == 0:
            continue
        c2, p_, _, _ = stats.chi2_contingency(t_, correction=False)
        ps.append(p_); vs.append(math.sqrt(c2 / t_.sum()))
    ps, vs = np.array(ps), np.array(vs)
    rows.append({"subsample n": n_s, "repetitions": len(ps), "median Cramer V": np.median(vs),
                 "median p-value": np.median(ps), "share of draws with p < 0.05": float((ps < ALPHA).mean())})
SUBS = pd.DataFrame(rows)
display(SUBS.assign(**{"median p-value": SUBS["median p-value"].map(fmt_p), "share of draws with p < 0.05": SUBS["share of draws with p < 0.05"].map(lambda v: f"{v:.0%}")}))
''')

# =============================================================================
# 9. Holm + summary
# =============================================================================
md("s9_intro")

code('''
# ---- (c) Cross-check: every test in the src battery against a direct scipy computation -----------------------------
battery = sa.run_all_tests(d)
summ = battery["summary"]
chk = []
for key, r in battery["spearman"].items():
    x_, y_ = key.split("~")
    f = d[[x_, y_]].astype(float).dropna()
    rho_s, p_s = stats.spearmanr(f[x_], f[y_])
    chk.append({"test": f"spearman {key}", "src": r["rho"], "scipy": float(rho_s), "|diff|": abs(r["rho"] - rho_s)})
for col, r in battery["chi_square"].items():
    ref = MAN[col] if col != "product_category" else manual_chi2(d, col)      # the battery runs on ALL orders (incl. 'unknown'), so does its reference
    chk.append({"test": f"chi-square {col}", "src": r["chi2"], "scipy": ref["chi2"], "|diff|": abs(r["chi2"] - ref["chi2"])})
gc = battery["group_comparison"]["delivery_duration_days~customer_state"]
big_states = d["customer_state"].value_counts()
big_states = big_states[big_states >= 30].index
H, pH = stats.kruskal(*[g["delivery_duration_days"].to_numpy() for _, g in d[d.customer_state.isin(big_states)].groupby("customer_state", observed=True)])
chk.append({"test": "kruskal duration by customer_state", "src": gc["statistic"], "scipy": float(H), "|diff|": abs(gc["statistic"] - H)})
chk = pd.DataFrame(chk)
display(chk)
assert chk["|diff|"].max() < 1e-6
print(f"Maximum absolute difference between src and direct scipy over {len(chk)} statistics: {chk['|diff|'].max():.2e}")
log_test("RQ1", "Kruskal-Wallis duration by customer_state", "kruskal", gc["n"], gc["p_value"], "epsilon-squared", gc["effect_size"], gc["effect_label"])
''')

code('''
# ---- Holm adjustment over the whole notebook-wide family of tests --------------------------------------------------
led = pd.DataFrame(LEDGER)
led["p_holm"] = sa.holm_adjust(led["p_value"].to_numpy())
led["significant_raw"] = led["p_value"] < ALPHA
led["significant_holm"] = led["p_holm"] < ALPHA
m_tests = len(led)
shown = led.assign(p_value=led.p_value.map(fmt_p), p_holm=led.p_holm.map(fmt_p))[
    ["rq", "test", "n", "p_value", "p_holm", "significant_holm", "effect_name", "effect", "effect_label"]]
print(f"{m_tests} hypothesis tests were run in this notebook's primary family; Holm-adjusted at alpha = {ALPHA}:")
print(f"  significant before adjustment: {int(led.significant_raw.sum())}   after Holm: {int(led.significant_holm.sum())}")
shown
''')

code('''
# ---- Figure 29: effect sizes at a glance + p-value vs sample size ---------------------------------------------------
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})
bars = [("Cramer's V", r["test"].replace(" x late", ""), r["effect"], r["effect_label"]) for _, r in led.iterrows() if r["effect_name"] == "Cramer's V"]
sp_rows = [("|Spearman rho|", r["test"].replace("Spearman ", ""), abs(r["effect"]), r["effect_label"]) for _, r in led.iterrows() if r["effect_name"] == "Spearman rho"]
allb = pd.DataFrame(bars + sp_rows, columns=["kind", "label", "value", "verdict"]).sort_values("value").reset_index(drop=True)
ax.barh(allb.label, allb.value, color=[P["blue"] if k == "Cramer's V" else P["amber"] for k in allb.kind], height=0.62)
for i, (v, vd) in enumerate(zip(allb.value, allb.verdict)):
    ax.text(v + 0.006, i, f"{v:.3f}  ({vd})", va="center", fontsize=8, color=P["navy"])
ax.axvline(0.1, color=P["slate_light"], ls=":"); ax.axvline(0.3, color=P["slate_light"], ls=":")
ax.text(0.1, -0.9, "0.1 small", fontsize=7.5, color=P["slate_light"], ha="center"); ax.text(0.3, -0.9, "0.3 medium", fontsize=7.5, color=P["slate_light"], ha="center")
ax.set_xlim(0, max(allb.value) * 1.35); ax.set_ylim(-1.2, len(allb) - 0.4)
ax.set_xlabel("Effect size (Cramer's V for association with late delivery; |Spearman rho| for rank correlations)")
ax.set_title("How big are the associations? (effect size, not p-value)"); ax.grid(axis="y", visible=False)
ax.legend(handles=[Patch(color=P["blue"], label="Cramer's V (chi-square)"), Patch(color=P["amber"], label="|Spearman rho|")], loc="lower right", fontsize=8.5)

ax2.plot(SUBS["subsample n"], SUBS["share of draws with p < 0.05"] * 100, "o-", color=P["red"], lw=2, label="share of random subsamples with p < 0.05")
ax2.set_xscale("log"); ax2.set_ylim(0, 105); ax2.set_xlabel("Subsample size n (log scale)"); ax2.set_ylabel("Subsamples with p < 0.05 (%)")
a3 = ax2.twinx(); a3.plot(SUBS["subsample n"], SUBS["median Cramer V"], "s--", color=P["blue"], lw=1.8, label="median Cramer's V")
a3.set_ylabel("Median Cramer's V", color=P["blue"]); a3.set_ylim(0, max(SUBS["median Cramer V"]) * 1.3); a3.grid(False); a3.spines["right"].set_visible(True)
h1, l1_ = ax2.get_legend_handles_labels(); h2, l2_ = a3.get_legend_handles_labels(); ax2.legend(h1 + h2, l1_ + l2_, loc="lower right", fontsize=8)
ax2.set_title("Same-state x late: p-value depends on n,\\neffect size does not")
save_show(fig, "29_effect_sizes_and_sample_size")
''')

md("s9_interp")

code('''
# ---- Findings table: one row per research question, numbers computed from the objects above ------------------------
def top_state_rows():
    s_ = st[st.reliable].sort_values("late_rate", ascending=False)
    return s_.iloc[0], s_.iloc[-1]
hi_s, lo_s = top_state_rows()
findings = pd.DataFrame([
    ("RQ1 Region", "customer_state x late", CHI["customer_state"]["n"], f"V = {CHI['customer_state']['cramers_v']:.3f} ({CHI['customer_state']['effect_label']})",
     f"{hi_s.customer_state} {pct(hi_s.late_rate)} (n={int(hi_s.n_orders):,}) to {lo_s.customer_state} {pct(lo_s.late_rate)} (n={int(lo_s.n_orders):,})",
     f"{int(state_vs_rest.sig_holm.sum())} of {len(state_vs_rest)} states differ from the rest after Holm"),
    ("RQ2 Category", "product_category x late", CHI["product_category"]["n"], f"V = {CHI['product_category']['cramers_v']:.3f} ({CHI['product_category']['effect_label']})",
     f"{len(rel_cat)} categories with n>={MIN_N}: {pct(rel_cat.late_rate.min())} to {pct(rel_cat.late_rate.max())}",
     f"{int(cat_vs_rest.sig_holm.sum())} of {len(cat_vs_rest)} categories differ from the rest after Holm"),
    ("RQ3 Same vs cross state", "same_state x late", n1 + n0, f"V = {CHI['same_state']['cramers_v']:.3f} ({CHI['same_state']['effect_label']}); MH OR = {OR_MH:.2f}",
     f"{pct(p1)} (n={n1:,}) vs {pct(p0)} (n={n0:,}); duration median {np.median(x_same):.1f} vs {np.median(x_cross):.1f} d",
     f"crude OR {OR_crude:.2f} shrinks to {OR_MH:.2f} after adjusting for state"),
    ("RQ4 Weight / freight", "Spearman vs duration", SPEAR[("total_freight", "delivery_duration_days")]["n"],
     f"rho(weight) = {SPEAR[('total_weight_g', 'delivery_duration_days')]['rho']:+.2f}, rho(freight) = {SPEAR[('total_freight', 'delivery_duration_days')]['rho']:+.2f}",
     f"weight {strength(SPEAR[('total_weight_g', 'delivery_duration_days')]['rho'])}; freight {strength(SPEAR[('total_freight', 'delivery_duration_days')]['rho'])}",
     f"freight partial rho given weight = {PART['freight|weight'][0]:+.2f}"),
    ("RQ5 Time", "year-month x late", ym_chi["n"], f"V = {ym_chi['cramers_v']:.3f} ({ym_chi['effect_label']})",
     f"spike months {pct(ks / ns_)} vs other months {pct(ko / no)}", f"{', '.join(SPIKES)} hold {ks / K_LATE:.0%} of late orders"),
    ("RQ6 Processing (retrospective)", "processing_days ~ is_late", sp_late["n"], f"rho = {sp_late['rho']:+.2f} ({sp_late['strength_label']})",
     f"late rate {pct(pt.late_rate.iloc[0])} (<=1 d) to {pct(pt.late_rate.iloc[-1])} (>14 d)", "not available at purchase time"),
], columns=["question", "test", "n", "effect size", "key numbers", "context"])
findings
''')

code('''
# ---- Machine-readable key results (the build script compares these with its own independent computation) ----------
def _f(x): return float(x)
_mm = rel_m[rel_m.purchase_year_month >= "2017-01"]
KEY_RESULTS = {
    "N": int(N), "K_LATE": int(K_LATE), "BASE": _f(BASE), "BASE_CI_LOW": _f(an.wilson_ci(K_LATE, N)[0]), "BASE_CI_HIGH": _f(an.wilson_ci(K_LATE, N)[1]),
    "chi2_state": _f(CHI["customer_state"]["chi2"]), "v_state": _f(CHI["customer_state"]["cramers_v"]), "dof_state": int(CHI["customer_state"]["dof"]),
    "v_state_corr": _f(CHI["customer_state"]["cramers_v_corrected"]), "v_state_lo": _f(VCI["customer_state"][0]), "v_state_hi": _f(VCI["customer_state"][1]),
    "chi2_cat": _f(CHI["product_category"]["chi2"]), "v_cat": _f(CHI["product_category"]["cramers_v"]), "dof_cat": int(CHI["product_category"]["dof"]),
    "v_cat_corr": _f(CHI["product_category"]["cramers_v_corrected"]), "v_cat_lo": _f(VCI["product_category"][0]), "v_cat_hi": _f(VCI["product_category"][1]),
    "chi2_same": _f(CHI["same_state"]["chi2"]), "v_same": _f(CHI["same_state"]["cramers_v"]),
    "chi2_month": _f(CHI["purchase_month"]["chi2"]), "v_month": _f(CHI["purchase_month"]["cramers_v"]),
    "v_ym": _f(ym_chi["cramers_v"]), "v_dow": _f(dow["cramers_v"]),
    "state_hi_name": str(hi_s.customer_state), "state_hi_rate": _f(hi_s.late_rate), "state_hi_n": int(hi_s.n_orders),
    "state_lo_name": str(lo_s.customer_state), "state_lo_rate": _f(lo_s.late_rate), "state_lo_n": int(lo_s.n_orders),
    "n_state_sig_raw": int(state_vs_rest.sig_raw.sum()), "n_state_sig_holm": int(state_vs_rest.sig_holm.sum()), "n_state_tested": int(len(state_vs_rest)),
    "n_cat_sig_raw": int(cat_vs_rest.sig_raw.sum()), "n_cat_sig_holm": int(cat_vs_rest.sig_holm.sum()), "n_cat_tested": int(len(cat_vs_rest)),
    "stab_train_same": _f(STAB.loc["train", "late % same-state"] / 100), "stab_train_cross": _f(STAB.loc["train", "late % cross-state"] / 100),
    "stab_val_same": _f(STAB.loc["validation", "late % same-state"] / 100), "stab_val_cross": _f(STAB.loc["validation", "late % cross-state"] / 100),
    "stab_test_same": _f(STAB.loc["test", "late % same-state"] / 100), "stab_test_cross": _f(STAB.loc["test", "late % cross-state"] / 100),
    "stab_test_gap_lo_pp": _f(STAB.loc["test", "CI low (pp)"]), "stab_test_gap_hi_pp": _f(STAB.loc["test", "CI high (pp)"]),
    "stab_n_months": int(N_MONTHS), "stab_n_months_same_higher": int(N_MONTHS_SAME_HIGHER), "stab_months_same_higher": ",".join(mo.index[mo.gap_pp > 0]),
    "fb_n_bins": int(len(fb)), "fb_first_label": str(fb["bin"].iloc[0]), "fb_last_label": str(fb["bin"].iloc[-1]),
    "fb_first_rate": _f(fb.late_rate.iloc[0]), "fb_last_rate": _f(fb.late_rate.iloc[-1]), "fb_first_n": int(fb.n_orders.iloc[0]), "fb_last_n": int(fb.n_orders.iloc[-1]),
    "fb_monotone": bool((np.diff(fb.late_rate.to_numpy()) >= 0).all()), "fb_first_median_dur": _f(fb.median_duration.iloc[0]), "fb_last_median_dur": _f(fb.median_duration.iloc[-1]),
    "rho_freight_late": _f(sa.spearman_test(d, "total_freight", "late")["rho"]),
    "n_unknown_cat": int(N_UNKNOWN), "n_cat_known": int(N_CAT), "rate_unknown_cat": _f(d.loc[d["product_category"] == UNKNOWN_CAT, "late"].mean()),
    "n_cat_reliable": int(len(rel_cat)), "n_cat_all": int(len(cp)), "cat_rate_min": _f(rel_cat.late_rate.min()), "cat_rate_max": _f(rel_cat.late_rate.max()),
    "rate_same": _f(p1), "rate_cross": _f(p0), "n_same": int(n1), "n_cross": int(n0), "diff_same_cross_pp": _f(100 * diff["diff"]),
    "diff_ci_lo_pp": _f(100 * diff["ci_low"]), "diff_ci_hi_pp": _f(100 * diff["ci_high"]),
    "or_crude": _f(OR_crude), "or_mh": _f(OR_MH), "or_mh_lo": _f(OR_MH_CI[0]), "or_mh_hi": _f(OR_MH_CI[1]), "or_crude_lo": _f(or_lo), "or_crude_hi": _f(or_hi),
    "median_same": _f(np.median(x_same)), "median_cross": _f(np.median(x_cross)), "mw_rank_biserial": _f(rb),
    "rho_weight_dur": _f(SPEAR[("total_weight_g", "delivery_duration_days")]["rho"]), "rho_freight_dur": _f(SPEAR[("total_freight", "delivery_duration_days")]["rho"]),
    "rho_fratio_dur": _f(SPEAR[("freight_ratio", "delivery_duration_days")]["rho"]), "rho_promise_dur": _f(SPEAR[("estimated_delivery_days", "delivery_duration_days")]["rho"]),
    "rho_weight_freight": _f(SPEAR[("total_weight_g", "total_freight")]["rho"]),
    "rho_freight_dur_lo": _f(SPEAR[("total_freight", "delivery_duration_days")]["ci_bw"][0]), "rho_freight_dur_hi": _f(SPEAR[("total_freight", "delivery_duration_days")]["ci_bw"][1]),
    "partial_freight_given_weight": _f(PART["freight|weight"][0]), "partial_weight_given_freight": _f(PART["weight|freight"][0]),
    "spike_rate": _f(ks / ns_), "other_rate": _f(ko / no), "spike_n": int(ns_), "other_n": int(no), "spike_share_orders": _f(ns_ / N), "spike_share_late": _f(ks / K_LATE),
    "n_spikes": int(len(SPIKES)), "spike_list": ",".join(SPIKES),
    "rho_proc_late": _f(sp_late["rho"]), "rho_proc_dur": _f(r_pd["rho"]), "rho_c2c_late": _f(sp_c2c["rho"]), "rho_proc_c2c": _f(sp_pc["rho"]),
    "proc_rate_first": _f(pt.late_rate.iloc[0]), "proc_rate_last": _f(pt.late_rate.iloc[-1]), "proc_n_last": int(pt.n_orders.iloc[-1]), "proc_n_first": int(pt.n_orders.iloc[0]),
    "proc_median_late": _f(x_late.median()), "proc_median_ok": _f(x_ok.median()), "n_proc": int(n_proc),
    "v_state_ns": _f(sa.chi_square_test(d_ns, "customer_state", "late")["cramers_v"]), "v_same_ns": _f(sa.chi_square_test(d_ns, "same_state", "late")["cramers_v"]),
    "v_month_ns": _f(sa.chi_square_test(d_ns, "purchase_month", "late")["cramers_v"]), "v_cat_ns": _f(sa.chi_square_test(d_ns[d_ns["product_category"] != UNKNOWN_CAT], "product_category", "late")["cramers_v"]),
    "n_ns": int(len(d_ns)), "base_ns": _f(d_ns.late.mean()),
    "rj_rate": _f(RJ_SP["p1"]), "sp_rate": _f(RJ_SP["p2"]), "rj_n": int(RJ_SP["na"]), "sp_n": int(RJ_SP["nb"]),
    "rj_sp_diff_pp": _f(100 * RJ_SP["diff"]), "rj_sp_lo_pp": _f(100 * RJ_SP["ci_low"]), "rj_sp_hi_pp": _f(100 * RJ_SP["ci_high"]),
    "n_tests_family": int(m_tests), "n_sig_raw": int(led.significant_raw.sum()), "n_sig_holm": int(led.significant_holm.sum()),
    "p_holm_max_nonsig_name": str(led.loc[~led.significant_holm, "test"].iloc[0]) if (~led.significant_holm).any() else "",
    "sp_cat_v": _f(s_sp["cramers_v"]), "sp_same_rate": _f(sp_only.loc[sp_only.same_state == 1, "late"].mean()), "sp_cross_rate": _f(sp_only.loc[sp_only.same_state == 0, "late"].mean()),
    "n_strata_both": int(len(big)), "n_strata_or_below1": int((big.OR < 1).sum()), "n_strata_or_above1": int((big.OR > 1).sum()),
    "sub_share_200": _f(SUBS["share of draws with p < 0.05"].iloc[0]), "sub_share_2000": _f(SUBS.loc[SUBS["subsample n"] == 2000, "share of draws with p < 0.05"].iloc[0]),
    "sub_share_20000": _f(SUBS.loc[SUBS["subsample n"] == 20000, "share of draws with p < 0.05"].iloc[0]), "sub_v_full": _f(SUBS["median Cramer V"].iloc[-1]),
    "rho_state_dur": _f(rho_dur.statistic), "rho_state_same": _f(rho_same.statistic),
    "cat_same_share_min": _f(cat_state_mix.same_state_share.min()), "cat_same_share_max": _f(cat_state_mix.same_state_share.max()),
    "kw_state_eps2": _f(gc["effect_size"]), "mw_med_diff": _f(MED_DIFF), "mw_med_lo": _f(MED_DIFF_CI[0]), "mw_med_hi": _f(MED_DIFF_CI[1]),
    "dur_median_all": _f(d.delivery_duration_days.median()),
    "rho_freight_dur_same": _f(STRAT["same state"][0]), "rho_freight_dur_cross": _f(STRAT["cross state"][0]),
    "same_sp_share": _f(S.loc[S.customer_state == "SP", "n_same"].iloc[0] / S["n_same"].sum()),
    "m_hi_name": str(_mm.loc[_mm.late_rate.idxmax(), "purchase_year_month"]), "m_hi_rate": _f(_mm.late_rate.max()), "m_hi_n": int(_mm.loc[_mm.late_rate.idxmax(), "n_orders"]),
    "m_lo_name": str(_mm.loc[_mm.late_rate.idxmin(), "purchase_year_month"]), "m_lo_rate": _f(_mm.late_rate.min()), "m_lo_n": int(_mm.loc[_mm.late_rate.idxmin(), "n_orders"]),
    "p_state": _f(CHI["customer_state"]["p_value"]), "p_cat": _f(CHI["product_category"]["p_value"]), "p_same": _f(CHI["same_state"]["p_value"]),
    "p_fratio_late": _f(sa.spearman_test(d, "freight_ratio", "late")["p_value"]), "p_state_same": _f(rho_same.pvalue),
    "est_same": _f(d.loc[d.same_state == 1, "estimated_delivery_days"].astype(float).mean()), "est_cross": _f(d.loc[d.same_state == 0, "estimated_delivery_days"].astype(float).mean()),
    "max_abs_rho_late": _f(max(abs(sa.spearman_test(d, c_, "late")["rho"]) for c_ in ["total_weight_g", "total_freight", "freight_ratio", "estimated_delivery_days"])),
    "rho_promise_late": _f(sa.spearman_test(d, "estimated_delivery_days", "late")["rho"]),
    "r_2017_02": _f(mp.set_index("purchase_year_month").loc["2017-02", "late_rate"]), "r_2018_02": _f(mp.set_index("purchase_year_month").loc["2018-02", "late_rate"]),
    "r_2017_03": _f(mp.set_index("purchase_year_month").loc["2017-03", "late_rate"]), "r_2018_03": _f(mp.set_index("purchase_year_month").loc["2018-03", "late_rate"]),
    "proc_rb": _f(rb_p),
    "cat_holm_names": ", ".join(sorted(cat_vs_rest.loc[cat_vs_rest.sig_holm, "product_category"])),
    "state_notholm_names": ", ".join(sorted(state_vs_rest.loc[state_vs_rest.sig_raw & ~state_vs_rest.sig_holm, "customer_state"])),
}
print("KEY_RESULTS " + json.dumps(KEY_RESULTS))
pd.Series({k: v for k, v in KEY_RESULTS.items() if not isinstance(v, str)}).to_frame("value").T.T.head(12)
''')

md("s10_limits")
md("s10_conclusion")

# =============================================================================
# Independent computation of every number quoted in the narrative
# =============================================================================


def compute_values() -> dict:
    """Recompute, from the parquet and plain scipy/numpy, every number the narrative quotes."""
    import math

    import numpy as np
    import pandas as pd
    from scipy import stats

    from src import config as cfg
    from src import analytics as an_
    from src import statistical_analysis as sa_

    d = pd.read_parquet(cfg.ORDERS_DELIVERED_PATH)
    d["late"] = d["is_late"].astype(int)
    d["ym"] = d["purchase_year_month"].astype(str)
    N = len(d); K = int(d.late.sum()); base = K / N
    V: dict = {"N": N, "K_LATE": K, "BASE": base}
    V["BASE_CI_LOW"], V["BASE_CI_HIGH"] = an_.wilson_ci(K, N)
    MIN_N = cfg.DEFAULT_MIN_ORDERS

    def chi(frame, col):
        return sa_.chi_square_test(frame, col, "late")

    dk = d[d.product_category != "unknown"]            # RQ2 leaves out the missing-category bucket
    NK, KK = len(dk), int(dk.late.sum())
    cs = {c: chi(dk if c == "product_category" else d, c) for c in ["customer_state", "product_category", "same_state", "purchase_month"]}
    V.update(v_state_corr=cs["customer_state"]["cramers_v_corrected"], v_cat_corr=cs["product_category"]["cramers_v_corrected"],
             chi2_state=cs["customer_state"]["chi2"], v_state=cs["customer_state"]["cramers_v"], dof_state=cs["customer_state"]["dof"],
             chi2_cat=cs["product_category"]["chi2"], v_cat=cs["product_category"]["cramers_v"], dof_cat=cs["product_category"]["dof"],
             chi2_same=cs["same_state"]["chi2"], v_same=cs["same_state"]["cramers_v"],
             chi2_month=cs["purchase_month"]["chi2"], v_month=cs["purchase_month"]["cramers_v"])
    # state table
    st = an_.state_performance(d, MIN_N)
    rel = st[st.reliable].sort_values("late_rate", ascending=False)
    hi, lo = rel.iloc[0], rel.iloc[-1]
    V.update(state_hi_name=str(hi.customer_state), state_hi_rate=hi.late_rate, state_hi_n=int(hi.n_orders),
             state_lo_name=str(lo.customer_state), state_lo_rate=lo.late_rate, state_lo_n=int(lo.n_orders))

    def vs_rest(tab, K=K, N=N):
        t = tab[tab.reliable]
        ps = []
        for _, r in t.iterrows():
            z = sa_.two_proportion_ztest(r.n_late, r.n_orders, K - r.n_late, N - r.n_orders)
            ps.append(z["p_value"])
        ps = np.array(ps)
        h = sa_.holm_adjust(ps)
        return int((ps < 0.05).sum()), int((h < 0.05).sum()), len(ps)

    V["n_state_sig_raw"], V["n_state_sig_holm"], V["n_state_tested"] = vs_rest(st)
    cp = an_.category_performance(dk, MIN_N)
    V["n_cat_sig_raw"], V["n_cat_sig_holm"], V["n_cat_tested"] = vs_rest(cp, KK, NK)
    V["n_unknown_cat"], V["n_cat_known"] = N - NK, NK
    V["rate_unknown_cat"] = float(d.loc[d.product_category == "unknown", "late"].mean())
    rc = cp[cp.reliable]
    V.update(n_cat_reliable=len(rc), n_cat_all=len(cp), cat_rate_min=rc.late_rate.min(), cat_rate_max=rc.late_rate.max())
    # same state
    g = d.groupby("same_state").late.agg(["sum", "size"])
    k1, n1, k0, n0 = int(g.loc[1, "sum"]), int(g.loc[1, "size"]), int(g.loc[0, "sum"]), int(g.loc[0, "size"])
    df_ = sa_.proportion_diff_ci(k1, n1, k0, n0)
    V.update(rate_same=k1 / n1, rate_cross=k0 / n0, n_same=n1, n_cross=n0, diff_same_cross_pp=100 * df_["diff"],
             diff_ci_lo_pp=100 * df_["ci_low"], diff_ci_hi_pp=100 * df_["ci_high"])
    a, b, c_, dd = k1, n1 - k1, k0, n0 - k0
    ZC = float(stats.norm.ppf(0.975))
    orc = a * dd / (b * c_); se = math.sqrt(1 / a + 1 / b + 1 / c_ + 1 / dd)
    V.update(or_crude=orc, or_crude_lo=math.exp(math.log(orc) - ZC * se), or_crude_hi=math.exp(math.log(orc) + ZC * se))
    num = den = 0.0
    P_, Q_, R_, S_ = [], [], [], []
    for _, gg in d.groupby("customer_state", observed=True):
        n = len(gg)
        A = int(((gg.same_state == 1) & (gg.late == 1)).sum()); B = int(((gg.same_state == 1) & (gg.late == 0)).sum())
        C = int(((gg.same_state == 0) & (gg.late == 1)).sum()); D = int(((gg.same_state == 0) & (gg.late == 0)).sum())
        if (A + B) == 0 or (C + D) == 0:
            continue
        r_, s_ = A * D / n, B * C / n
        num += r_; den += s_
        P_.append((A + D) / n); Q_.append((B + C) / n); R_.append(r_); S_.append(s_)
    P_, Q_, R_, S_ = map(np.array, (P_, Q_, R_, S_))
    var = (P_ * R_).sum() / (2 * num ** 2) + (P_ * S_ + Q_ * R_).sum() / (2 * num * den) + (Q_ * S_).sum() / (2 * den ** 2)
    omh = num / den
    V.update(or_mh=omh, or_mh_lo=math.exp(math.log(omh) - ZC * math.sqrt(var)), or_mh_hi=math.exp(math.log(omh) + ZC * math.sqrt(var)))
    V["median_same"] = float(d.loc[d.same_state == 1, "delivery_duration_days"].median())
    V["median_cross"] = float(d.loc[d.same_state == 0, "delivery_duration_days"].median())
    x_c = d.loc[d.same_state == 0, "delivery_duration_days"].to_numpy(); x_s = d.loc[d.same_state == 1, "delivery_duration_days"].to_numpy()
    u = stats.mannwhitneyu(x_c, x_s, alternative="two-sided").statistic
    V["mw_rank_biserial"] = 2 * u / (len(x_c) * len(x_s)) - 1
    V["dur_median_all"] = float(d.delivery_duration_days.median())

    def rho(x, y):
        f = d[[x, y]].astype(float).dropna()
        return float(stats.spearmanr(f[x], f[y]).statistic), len(f)

    V["rho_weight_dur"] = rho("total_weight_g", "delivery_duration_days")[0]
    V["rho_freight_dur"] = rho("total_freight", "delivery_duration_days")[0]
    V["rho_fratio_dur"] = rho("freight_ratio", "delivery_duration_days")[0]
    V["rho_promise_dur"] = rho("estimated_delivery_days", "delivery_duration_days")[0]
    V["rho_weight_freight"] = rho("total_weight_g", "total_freight")[0]
    V["rho_proc_late"] = rho("processing_days", "late")[0]
    V["rho_proc_dur"] = rho("processing_days", "delivery_duration_days")[0]
    V["rho_c2c_late"] = rho("carrier_to_customer_days", "late")[0]
    V["rho_proc_c2c"] = rho("processing_days", "carrier_to_customer_days")[0]
    r_ = V["rho_freight_dur"]; n_ = len(d)
    se_ = math.sqrt((1 + r_ * r_ / 2) / (n_ - 3))
    V["rho_freight_dur_lo"], V["rho_freight_dur_hi"] = math.tanh(math.atanh(r_) - ZC * se_), math.tanh(math.atanh(r_) + ZC * se_)
    f = d[["total_freight", "delivery_duration_days", "total_weight_g"]].astype(float).dropna()
    rx, ry, rz = (stats.rankdata(f[c]) for c in f.columns)
    rxy, rxz, ryz = np.corrcoef(rx, ry)[0, 1], np.corrcoef(rx, rz)[0, 1], np.corrcoef(ry, rz)[0, 1]
    V["partial_freight_given_weight"] = (rxy - rxz * ryz) / math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    V["partial_weight_given_freight"] = (ryz - rxz * rxy) / math.sqrt((1 - rxz ** 2) * (1 - rxy ** 2))
    # time
    mp = an_.monthly_performance(d, MIN_N)
    rm = mp[mp.reliable]
    spikes = rm[rm.late_ci_low >= 1.5 * base].purchase_year_month.tolist()
    V["n_spikes"], V["spike_list"] = len(spikes), ",".join(spikes)
    sm = d.ym.isin(spikes)
    ks, ns_ = int(d.loc[sm, "late"].sum()), int(sm.sum())
    ow = d[(d.ym >= "2017-01") & ~sm]
    V.update(spike_rate=ks / ns_, other_rate=float(ow.late.mean()), spike_n=ns_, other_n=len(ow), spike_share_orders=ns_ / N, spike_share_late=ks / K)
    ymf = d[d.ym.isin(rm.purchase_year_month)]
    V["v_ym"] = chi(ymf.assign(ym=ymf.ym), "ym")["cramers_v"]
    V["v_dow"] = chi(d, "purchase_dow")["cramers_v"]
    # processing
    pt = an_.late_by_processing_bucket(d, min_orders=MIN_N)
    V.update(proc_rate_first=pt.late_rate.iloc[0], proc_rate_last=pt.late_rate.iloc[-1], proc_n_last=int(pt.n_orders.iloc[-1]),
             proc_n_first=int(pt.n_orders.iloc[0]), n_proc=int(d.processing_days.notna().sum()))
    V["proc_median_late"] = float(d.loc[d.late == 1, "processing_days"].median())
    V["proc_median_ok"] = float(d.loc[d.late == 0, "processing_days"].median())
    # sensitivity
    ns = d[~sm]
    V.update(v_state_ns=chi(ns, "customer_state")["cramers_v"], v_same_ns=chi(ns, "same_state")["cramers_v"],
             v_month_ns=chi(ns, "purchase_month")["cramers_v"], v_cat_ns=chi(ns[ns.product_category != "unknown"], "product_category")["cramers_v"],
             n_ns=len(ns), base_ns=float(ns.late.mean()))
    rj = d[d.customer_state == "RJ"].late; sp_ = d[d.customer_state == "SP"].late
    dj = sa_.proportion_diff_ci(rj.sum(), len(rj), sp_.sum(), len(sp_))
    V.update(rj_rate=rj.mean(), sp_rate=sp_.mean(), rj_n=len(rj), sp_n=len(sp_), rj_sp_diff_pp=100 * dj["diff"],
             rj_sp_lo_pp=100 * dj["ci_low"], rj_sp_hi_pp=100 * dj["ci_high"])
    sub = d[d.customer_state == "SP"]
    V["sp_cat_v"] = sa_.chi_square_test(sub[sub.product_category != "unknown"], "product_category", "late")["cramers_v"]
    V["sp_same_rate"] = float(sub.loc[sub.same_state == 1, "late"].mean()); V["sp_cross_rate"] = float(sub.loc[sub.same_state == 0, "late"].mean())
    ms = dk.groupby("product_category", observed=True)["same_state"].mean()
    ns_c = dk.groupby("product_category", observed=True).size()
    ms = ms[ns_c >= MIN_N]
    V["cat_same_share_min"], V["cat_same_share_max"] = float(ms.min()), float(ms.max())
    gc = sa_.kruskal_or_mannwhitney(d, "delivery_duration_days", "customer_state", min_group_size=30)
    V["kw_state_eps2"] = gc["effect_size"]
    sr = st[st.reliable]
    V["rho_state_dur"] = float(stats.spearmanr(sr.late_rate, sr.median_duration).statistic)
    V["rho_state_same"] = float(stats.spearmanr(sr.late_rate, sr.customer_state.map(d.groupby("customer_state", observed=True).same_state.mean())).statistic)
    for g_, nm in [(1, "same"), (0, "cross")]:
        sub_ = d[d.same_state == g_][["total_freight", "delivery_duration_days"]].astype(float).dropna()
        V[f"rho_freight_dur_{nm}"] = float(stats.spearmanr(sub_.total_freight, sub_.delivery_duration_days).statistic)
    same_n = d[d.same_state == 1]
    V["same_sp_share"] = float((same_n.customer_state == "SP").mean())
    mm = rm[rm.purchase_year_month >= "2017-01"]
    V.update(m_hi_name=str(mm.loc[mm.late_rate.idxmax(), "purchase_year_month"]), m_hi_rate=float(mm.late_rate.max()), m_hi_n=int(mm.loc[mm.late_rate.idxmax(), "n_orders"]),
             m_lo_name=str(mm.loc[mm.late_rate.idxmin(), "purchase_year_month"]), m_lo_rate=float(mm.late_rate.min()), m_lo_n=int(mm.loc[mm.late_rate.idxmin(), "n_orders"]))

    def flags(tab, K=K, N=N):
        t = tab[tab.reliable]
        ps = np.array([sa_.two_proportion_ztest(r.n_late, r.n_orders, K - r.n_late, N - r.n_orders)["p_value"] for _, r in t.iterrows()])
        h = sa_.holm_adjust(ps)
        return t.reset_index(drop=True), ps, h

    V.update(p_state=cs["customer_state"]["p_value"], p_cat=cs["product_category"]["p_value"], p_same=cs["same_state"]["p_value"],
             p_fratio_late=sa_.spearman_test(d, "freight_ratio", "late")["p_value"],
             p_state_same=float(stats.spearmanr(sr.late_rate, sr.customer_state.map(d.groupby("customer_state", observed=True).same_state.mean())).pvalue))
    est = d.groupby("same_state").estimated_delivery_days.apply(lambda v: v.astype(float).mean())
    V.update(est_same=float(est.loc[1]), est_cross=float(est.loc[0]))
    V["max_abs_rho_late"] = max(abs(float(stats.spearmanr(d[c].astype(float).dropna(), d.loc[d[c].notna(), "late"]).statistic))
                                for c in ["total_weight_g", "total_freight", "freight_ratio", "estimated_delivery_days"])
    V["rho_promise_late"] = float(stats.spearmanr(d.estimated_delivery_days.astype(float), d.late).statistic)
    mpi = mp.set_index("purchase_year_month")
    for k_ in ["2017-02", "2018-02", "2017-03", "2018-03"]:
        V["r_" + k_.replace("-", "_")] = float(mpi.loc[k_, "late_rate"])
    pm = d[d.processing_days.notna()]
    xl, xo = pm.loc[pm.late == 1, "processing_days"].to_numpy(), pm.loc[pm.late == 0, "processing_days"].to_numpy()
    V["proc_rb"] = 2 * stats.mannwhitneyu(xl, xo, alternative="two-sided").statistic / (len(xl) * len(xo)) - 1
    V["med_diff"] = V["median_cross"] - V["median_same"]
    t_, ps_, h_ = flags(cp, KK, NK)
    V["cat_holm_names"] = ", ".join(sorted(t_.loc[h_ < 0.05, "product_category"]))
    # same-state gap over time (windows from the model's metrics file) and by month
    import json as _json
    wins = _json.loads(cfg.MODEL_METRICS_PATH.read_text())["split"]["windows"]
    tsd = pd.to_datetime(d.order_purchase_timestamp)
    for key_, wn in [("train", "train"), ("val", "validation"), ("test", "test")]:
        w0, w1 = wins[wn]
        wm = (tsd >= pd.Timestamp(w0)) & ((tsd < pd.Timestamp(w1)) if w1 else True)
        V[f"stab_{key_}_same"] = float(d.loc[wm & (d.same_state == 1), "late"].mean())
        V[f"stab_{key_}_cross"] = float(d.loc[wm & (d.same_state == 0), "late"].mean())
    wm = (tsd >= pd.Timestamp(wins["test"][0]))
    a_, b_ = d[wm & (d.same_state == 1)].late, d[wm & (d.same_state == 0)].late
    dtest = sa_.proportion_diff_ci(int(a_.sum()), len(a_), int(b_.sum()), len(b_))
    V["stab_test_gap_lo_pp"], V["stab_test_gap_hi_pp"] = 100 * dtest["ci_low"], 100 * dtest["ci_high"]
    mrate = d[d.ym >= "2017-01"].pivot_table(index="ym", columns="same_state", values="late", aggfunc="mean")
    higher = mrate.index[mrate[1] > mrate[0]]
    V["stab_n_months"], V["stab_n_months_same_higher"], V["stab_months_same_higher"] = len(mrate), len(higher), ",".join(higher)
    fbt = an_.freight_bins(d, min_orders=MIN_N)
    V.update(fb_n_bins=len(fbt), fb_first_label=str(fbt["bin"].iloc[0]), fb_last_label=str(fbt["bin"].iloc[-1]),
             fb_first_rate=float(fbt.late_rate.iloc[0]), fb_last_rate=float(fbt.late_rate.iloc[-1]),
             fb_first_n=int(fbt.n_orders.iloc[0]), fb_last_n=int(fbt.n_orders.iloc[-1]),
             fb_monotone=bool((np.diff(fbt.late_rate.to_numpy()) >= 0).all()),
             fb_first_median_dur=float(fbt.median_duration.iloc[0]), fb_last_median_dur=float(fbt.median_duration.iloc[-1]))
    V["rho_freight_late"] = float(stats.spearmanr(d.total_freight.astype(float), d.late).statistic)
    t_, ps_, h_ = flags(st)
    V["state_notholm_names"] = ", ".join(sorted(t_.loc[(ps_ < 0.05) & ~(h_ < 0.05), "customer_state"]))
    return V


# Values the narrative needs that are not (or not exactly) in KEY_RESULTS are derived here from V.
def narrative_values(V: dict) -> dict:
    """String-formatted placeholders for the narrative («name»), all derived from computed values."""
    pc = lambda x, nd=2: f"{100 * x:.{nd}f}%"  # noqa: E731

    def fp(p):
        return "< 1e-300" if p <= 1.0001 * 2.2250738585072014e-308 else (f"= {p:.1e}" if p < 0.001 else f"= {p:.2f}")
    f3 = lambda x: f"{x:+.3f}"  # noqa: E731
    out = {
        "N": f"{V['N']:,}", "K_LATE": f"{V['K_LATE']:,}", "BASE": pc(V["BASE"]), "BASE_CI": f"{pc(V['BASE_CI_LOW'])} to {pc(V['BASE_CI_HIGH'])}",
        "v_state_corr": f"{V['v_state_corr']:.3f}", "v_cat_corr": f"{V['v_cat_corr']:.3f}", "v_state": f"{V['v_state']:.3f}", "v_cat": f"{V['v_cat']:.3f}", "v_same": f"{V['v_same']:.3f}", "v_month": f"{V['v_month']:.3f}",
        "v_ym": f"{V['v_ym']:.3f}", "v_dow": f"{V['v_dow']:.3f}",
        "state_hi": f"{V['state_hi_name']} ({pc(V['state_hi_rate'], 1)}, n = {V['state_hi_n']:,})",
        "state_lo": f"{V['state_lo_name']} ({pc(V['state_lo_rate'], 1)}, n = {V['state_lo_n']:,})",
        "state_hi_rate": pc(V["state_hi_rate"], 1), "state_lo_rate": pc(V["state_lo_rate"], 1),
        "n_state_sig_raw": str(V["n_state_sig_raw"]), "n_state_sig_holm": str(V["n_state_sig_holm"]), "n_state_tested": str(V["n_state_tested"]),
        "n_cat_sig_raw": str(V["n_cat_sig_raw"]), "n_cat_sig_holm": str(V["n_cat_sig_holm"]), "n_cat_tested": str(V["n_cat_tested"]),
        "n_cat_reliable": str(V["n_cat_reliable"]), "n_cat_all": str(V["n_cat_all"]),
        "cat_rate_min": pc(V["cat_rate_min"], 1), "cat_rate_max": pc(V["cat_rate_max"], 1),
        "rate_same": pc(V["rate_same"]), "rate_cross": pc(V["rate_cross"]), "n_same": f"{V['n_same']:,}", "n_cross": f"{V['n_cross']:,}",
        "diff_pp": f"{V['diff_same_cross_pp']:+.2f}", "diff_ci": f"{V['diff_ci_lo_pp']:+.2f} to {V['diff_ci_hi_pp']:+.2f}",
        "or_crude": f"{V['or_crude']:.2f}", "or_mh": f"{V['or_mh']:.2f}", "or_mh_ci": f"{V['or_mh_lo']:.2f} to {V['or_mh_hi']:.2f}",
        "median_same": f"{V['median_same']:.1f}", "median_cross": f"{V['median_cross']:.1f}", "mw_rb": f"{V['mw_rank_biserial']:+.2f}",
        "rho_weight_dur": f3(V["rho_weight_dur"]), "rho_freight_dur": f3(V["rho_freight_dur"]), "rho_fratio_dur": f3(V["rho_fratio_dur"]),
        "rho_promise_dur": f3(V["rho_promise_dur"]), "rho_weight_freight": f3(V["rho_weight_freight"]),
        "rho_freight_ci": f"{V['rho_freight_dur_lo']:.3f} to {V['rho_freight_dur_hi']:.3f}",
        "partial_fw": f3(V["partial_freight_given_weight"]), "partial_wf": f3(V["partial_weight_given_freight"]),
        "spike_list": V["spike_list"].replace(",", ", "), "n_spikes": str(V["n_spikes"]),
        "spike_rate": pc(V["spike_rate"], 1), "other_rate": pc(V["other_rate"], 1), "spike_n": f"{V['spike_n']:,}", "other_n": f"{V['other_n']:,}",
        "spike_share_orders": pc(V["spike_share_orders"], 1), "spike_share_late": pc(V["spike_share_late"], 1),
        "rho_proc_late": f3(V["rho_proc_late"]), "rho_proc_dur": f3(V["rho_proc_dur"]), "rho_c2c_late": f3(V["rho_c2c_late"]), "rho_proc_c2c": f3(V["rho_proc_c2c"]),
        "proc_rate_first": pc(V["proc_rate_first"], 1), "proc_rate_last": pc(V["proc_rate_last"], 1),
        "proc_n_first": f"{V['proc_n_first']:,}", "proc_n_last": f"{V['proc_n_last']:,}", "n_proc": f"{V['n_proc']:,}",
        "proc_median_late": f"{V['proc_median_late']:.2f}", "proc_median_ok": f"{V['proc_median_ok']:.2f}",
        "v_state_ns": f"{V['v_state_ns']:.3f}", "v_same_ns": f"{V['v_same_ns']:.3f}", "v_month_ns": f"{V['v_month_ns']:.3f}", "v_cat_ns": f"{V['v_cat_ns']:.3f}",
        "n_ns": f"{V['n_ns']:,}", "base_ns": pc(V["base_ns"]),
        "rj": pc(V["rj_rate"], 1), "sp": pc(V["sp_rate"], 1), "rj_n": f"{V['rj_n']:,}", "sp_n": f"{V['sp_n']:,}",
        "rj_sp_diff": f"{V['rj_sp_diff_pp']:+.1f}", "rj_sp_ci": f"{V['rj_sp_lo_pp']:+.1f} to {V['rj_sp_hi_pp']:+.1f}",
        "sp_cat_v": f"{V['sp_cat_v']:.3f}", "sp_same_rate": pc(V["sp_same_rate"]), "sp_cross_rate": pc(V["sp_cross_rate"]),
        "cat_same_min": pc(V["cat_same_share_min"], 0), "cat_same_max": pc(V["cat_same_share_max"], 0),
        "kw_eps2": f"{V['kw_state_eps2']:.2f}",
        "rho_state_dur": f3(V["rho_state_dur"]), "rho_state_same": f3(V["rho_state_same"]),
        "dur_median_all": f"{V['dur_median_all']:.1f}",
        "p_state": fp(V["p_state"]), "p_cat": fp(V["p_cat"]), "p_same": fp(V["p_same"]), "p_fratio_late": fp(V["p_fratio_late"]), "p_state_same": fp(V["p_state_same"]),
        "est_same": f"{V['est_same']:.1f}", "est_cross": f"{V['est_cross']:.1f}", "rho_promise_late": f3(V["rho_promise_late"]), "max_abs_rho_late": f"{V['max_abs_rho_late']:.3f}",
        "r_2017_02": pc(V["r_2017_02"], 1), "r_2018_02": pc(V["r_2018_02"], 1), "r_2017_03": pc(V["r_2017_03"], 1), "r_2018_03": pc(V["r_2018_03"], 1),
        "proc_rb": f"{V['proc_rb']:.2f}", "med_diff": f"{V['med_diff']:.1f}",
        "n_strata_both": str(V["n_strata_both"]), "n_strata_or_below1": str(V["n_strata_or_below1"]), "n_strata_or_above1": str(V["n_strata_or_above1"]),
        "n_sig_raw": str(V["n_sig_raw"]), "n_sig_holm": str(V["n_sig_holm"]), "n_tests_family": str(V["n_tests_family"]),
        "p_holm_max_nonsig_name": V["p_holm_max_nonsig_name"].replace("_", " "),
        "sub_share_200": f"{V['sub_share_200']:.0%}", "sub_share_2000": f"{V['sub_share_2000']:.0%}", "sub_share_20000": f"{V['sub_share_20000']:.0%}",
        "sub_v_full": f"{V['sub_v_full']:.3f}",
        "rho_fd_same": f3(V["rho_freight_dur_same"]), "rho_fd_cross": f3(V["rho_freight_dur_cross"]),
        "same_sp_share": pc(V["same_sp_share"], 1),
        "m_hi": f"{V['m_hi_name']} ({pc(V['m_hi_rate'], 1)}, n = {V['m_hi_n']:,})", "m_lo": f"{V['m_lo_name']} ({pc(V['m_lo_rate'], 1)}, n = {V['m_lo_n']:,})",
        "v_cat_ci": f"{V['v_cat_lo']:.3f} to {V['v_cat_hi']:.3f}",
        "n_unknown_cat": f"{V['n_unknown_cat']:,}", "pct_unknown_cat": pc(V["n_unknown_cat"] / V["N"], 1), "n_cat_known": f"{V['n_cat_known']:,}",
        "rate_unknown_cat": pc(V["rate_unknown_cat"], 1),
        "stab_train_same": pc(V["stab_train_same"], 1), "stab_train_cross": pc(V["stab_train_cross"], 1),
        "stab_val_same": pc(V["stab_val_same"], 1), "stab_val_cross": pc(V["stab_val_cross"], 1),
        "stab_test_same": pc(V["stab_test_same"], 1), "stab_test_cross": pc(V["stab_test_cross"], 1),
        "stab_train_gap": f"{100 * (V['stab_train_same'] - V['stab_train_cross']):+.1f}", "stab_val_gap": f"{100 * (V['stab_val_same'] - V['stab_val_cross']):+.1f}",
        "stab_test_gap": f"{100 * (V['stab_test_same'] - V['stab_test_cross']):+.1f}",
        "stab_test_ci": f"{V['stab_test_gap_lo_pp']:+.1f} to {V['stab_test_gap_hi_pp']:+.1f}",
        "stab_n_months": str(V["stab_n_months"]), "stab_n_months_same_higher": str(V["stab_n_months_same_higher"]),
        "stab_months_same_higher": V["stab_months_same_higher"].replace(",", ", "),
        "fb_ratio": f"{V['fb_last_rate'] / V['fb_first_rate']:.1f}",
        "fb_first_rate": pc(V["fb_first_rate"], 1), "fb_last_rate": pc(V["fb_last_rate"], 1), "fb_first_n": f"{V['fb_first_n']:,}", "fb_last_n": f"{V['fb_last_n']:,}",
        "fb_first_label": V["fb_first_label"], "fb_last_label": V["fb_last_label"], "fb_n_bins": str(V["fb_n_bins"]),
        "fb_first_dur": f"{V['fb_first_median_dur']:.1f}", "fb_last_dur": f"{V['fb_last_median_dur']:.1f}", "rho_freight_late": f3(V["rho_freight_late"]),
        "cat_holm_names": V["cat_holm_names"].replace("_", " "), "state_notholm_names": V["state_notholm_names"],
    }
    return out


def fill(text: str, nv: dict) -> str:
    def rep(m):
        key = m.group(1)
        if key not in nv:
            raise KeyError(f"narrative placeholder «{key}» has no computed value")
        return nv[key]
    return re.sub(r"«([A-Za-z0-9_]+)»", rep, text)


def parse_key_results(nb_) -> dict:
    """Return the KEY_RESULTS JSON printed by the executed notebook."""
    key = None
    for c in nb_.cells:
        if c.cell_type == "code":
            for o in c.get("outputs", []):
                t = o.get("text", "") if o.get("output_type") == "stream" else ""
                m = re.search(r"KEY_RESULTS (\{.*\})", t)
                if m:
                    key = json.loads(m.group(1))
    assert key is not None, "KEY_RESULTS not found in executed notebook"
    return key


def verify_against_notebook(key: dict, V: dict) -> None:
    """Compare the executed notebook's KEY_RESULTS with the independently computed values."""
    bad = []
    for k, v in V.items():
        if k not in key:
            continue
        nv_ = key[k]
        if isinstance(v, str) or isinstance(nv_, str):
            ok = str(v) == str(nv_)
        else:
            ok = abs(float(v) - float(nv_)) <= 1e-7 * max(1.0, abs(float(v)))
        if not ok:
            bad.append((k, v, nv_))
    assert not bad, f"narrative values disagree with the executed notebook: {bad}"
    print(f"verified {sum(k in key for k in V)} independently computed numbers against the executed notebook outputs")


def build_notebook() -> "nbf.NotebookNode":
    """Code cells + markdown cells that still hold their raw «placeholders» (filled after execution)."""
    nb_cells = []
    for kind, payload in cells:
        if kind == "md":
            c = nbf.v4.new_markdown_cell(TXT.MD.get(payload, f"(text pending: {payload})").strip("\n"))
            c.metadata["_key"] = payload
            nb_cells.append(c)
        else:
            nb_cells.append(nbf.v4.new_code_cell(payload))
    nb = nbf.v4.new_notebook(cells=nb_cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


def fill_markdown(nb_, nv: dict) -> None:
    for c in nb_.cells:
        if c.cell_type == "markdown":
            c.source = fill(c.source, nv)
            c.metadata.pop("_key", None)


if __name__ == "__main__":
    t0 = time.time()
    V = compute_values()                                # independent of the notebook
    nb = build_notebook()
    client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(HERE)}})
    client.execute()
    key = parse_key_results(nb)
    verify_against_notebook(key, V)
    V_all = {**key, **V}                                # notebook-only quantities come from the executed outputs
    TXT.check(V_all)                                    # qualitative claims in the narrative must hold
    fill_markdown(nb, narrative_values(V_all))
    nbf.write(nb, NB_PATH)
    print(f"executed and wrote {NB_PATH} in {time.time() - t0:.1f}s")
