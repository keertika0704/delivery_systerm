"""Generate and execute notebooks/03_model_training.ipynb.

Usage (from anywhere):  .venv/bin/python notebooks/_build_nb03.py
The narrative text lives in notebooks/_nb03_text.py; code cells live in this file.  After execution the
script cross-checks every number quoted in the Markdown against the values the notebook computed
(the "Appendix" cell prints them as JSON) and against models/model_metrics.json, and fails on drift.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

from _nb03_text import MD  # type: ignore  # same directory

HERE = Path(__file__).resolve().parent
NB_PATH = HERE / "03_model_training.ipynb"
MARK = "###QUOTED###"

cells: list = []


def md(key: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(MD.get(key, f"*(narrative `{key}` pending)*").strip("\n")))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# =============================================================================
# 0. Title + setup
# =============================================================================
md("title")

code(r'''
import sys, json, time, textwrap, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

# Resolve the project root whether the notebook is run from notebooks/ or from the root.
_here = Path.cwd().resolve()
ROOT = next((p for p in [_here, *_here.parents] if (p / "src" / "config.py").is_file()), None)
if ROOT is None:
    raise RuntimeError("Run this notebook from inside the DeliveryIQ project folder.")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config, prediction as pred, train_model as tm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 140)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

P = config.PALETTE
MODEL_COLORS = {"dummy_prior": P["slate_light"], "logistic_regression": P["blue"],
                "random_forest": "#0891B2", "hist_gradient_boosting": "#7C3AED"}
plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 110, "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.edgecolor": P["slate_light"], "axes.labelcolor": P["slate"],
    "axes.titlecolor": P["navy"], "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": P["border"], "grid.linewidth": 0.7,
    "axes.axisbelow": True, "xtick.color": P["slate"], "ytick.color": P["slate"],
    "font.size": 10, "legend.frameon": False,
})
FIG_DIR = config.FIGURES_DIR
FIG_DIR.mkdir(parents=True, exist_ok=True)
T0 = time.time()


def save_show(fig, name: str) -> None:
    """Save a figure as PNG under reports/figures/ (for GitHub) and display it inline."""
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    plt.show()


def wrap(text: str, width: int = 118) -> None:
    print(textwrap.fill(str(text), width=width))


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


# Numbers quoted in the Markdown are registered here as formatted strings; the build script checks
# that every one of them appears in the narrative, so text and computed values cannot drift apart.
QUOTED: dict[str, str] = {}

saved = json.load(open(config.MODEL_METRICS_PATH))
print(f"Project root : {ROOT}")
print(f"Saved model metrics generated at {saved['generated_at']}  (scikit-learn {saved['library_versions']['scikit-learn']})")
print(f"Selected model on disk       : {saved['selected_model']['name']}")
''')

# =============================================================================
# 1. Problem framing
# =============================================================================
md("s1_intro")

code(r'''
df = tm.load_training_frame()          # delivered, labelled orders, sorted by purchase time
y_all = df[tm.TARGET]
n_all, n_late_all = len(df), int(y_all.sum())
print(f"Analysis set: {n_all:,} delivered orders, purchases {df['order_purchase_timestamp'].min():%Y-%m-%d} .. "
      f"{df['order_purchase_timestamp'].max():%Y-%m-%d}")
print(f"Late orders : {n_late_all:,}  ({pct(y_all.mean(), 2)})   on time / early: {n_all - n_late_all:,}  ({pct(1 - y_all.mean(), 2)})")
print(f"Class ratio : {(n_all - n_late_all) / n_late_all:.1f} on-time orders for every late order")

est, dlv = df["order_estimated_delivery_date"], df["order_delivered_customer_date"]
ts_late = dlv > est                                  # naive: compare full timestamps
cal_late = df["delivery_delay_days"] > 0             # project rule: calendar dates
same_day = (dlv.dt.normalize() == est.dt.normalize()) & ts_late
rule = pd.DataFrame({
    "rule": ["calendar-date rule (used: delivered date > estimated date)", "timestamp rule (delivered time > estimated midnight)"],
    "late orders": [int(cal_late.sum()), int(ts_late.sum())],
    "late rate": [cal_late.mean(), ts_late.mean()],
})
display(rule)
print(f"Orders that arrive during the promised calendar day but after its midnight timestamp: {int(same_day.sum()):,} "
      f"(= {int(ts_late.sum()) - int(cal_late.sum()):,} difference between the two rules)")
assert int(same_day.sum()) == int(ts_late.sum()) - int(cal_late.sum())
assert (cal_late.astype(int) == y_all).all()
QUOTED["n_all"] = f"{n_all:,}"
QUOTED["n_late_all"] = f"{n_late_all:,}"
QUOTED["late_rate_all"] = pct(y_all.mean(), 2)
QUOTED["class_ratio"] = f"{(n_all - n_late_all) / n_late_all:.1f}"
QUOTED["n_same_day"] = f"{int(same_day.sum()):,}"
QUOTED["ts_late_rate"] = pct(ts_late.mean(), 2)
''')

code(r'''
delay = df["delivery_delay_days"].astype(float)
lo, hi = -40, 40
beyond = int(((delay < lo) | (delay > hi)).sum())
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), gridspec_kw={"width_ratios": [1, 2.2]})

ax = axes[0]
counts = [n_all - n_late_all, n_late_all]
bars = ax.bar(["On time / early", "Late"], counts, color=[P["green"], P["red"]], width=0.6)
for b, v in zip(bars, counts):
    ax.annotate(f"{v:,}\n({v / n_all:.2%})", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=9, color=P["slate"])
ax.set_ylim(0, n_all * 1.15)
ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
ax.set_ylabel("Delivered orders (n)"); ax.set_title("Target balance (is_late)")
ax.grid(axis="x", visible=False)

ax = axes[1]
vals = delay.clip(lo, hi)
edges = np.arange(lo - 0.5, hi + 1.5, 1)
neg = vals[vals <= 0]; posv = vals[vals > 0]
ax.hist(neg, bins=edges, color=P["green"], label="on time / early (delay <= 0 days)")
ax.hist(posv, bins=edges, color=P["red"], label="late (delay > 0 days)")
ax.axvline(0.5, color=P["navy"], lw=1.2, ls="--")
ax.annotate("promised date\n(calendar rule: late only if\ndelivered after this day)", (0.5, ax.get_ylim()[1] * 0.38), xytext=(8, 0),
            textcoords="offset points", fontsize=8.5, color=P["navy"])
ax.set_xlabel(f"Delivery delay (days; delivered date minus promised date; clipped to [{lo}, {hi}], {beyond:,} orders beyond)")
ax.set_ylabel("Delivered orders (n)"); ax.set_title("Most orders arrive early; lateness is a thin right tail")
ax.legend(loc="upper right")
save_show(fig, "30_target_balance_and_delay")
''')

md("s1_interp")

# =============================================================================
# 2. Feature selection and leakage
# =============================================================================
md("s2_intro")

code(r'''
def group_of(col: str) -> str:
    if col in config.LEAKAGE_COLUMNS:
        return "3 Leakage / outcome (never a feature)"
    if col in config.PURCHASE_TIME_FEATURES:
        return "1 Purchase-time allow-list"
    if col in config.PURCHASE_TIME_UNUSED:
        return "2 Known at purchase, left out of the allow-list"
    if col in config.NON_FEATURE_COLUMNS:
        return "4 Non-feature (key / split key / time-drift proxy)"
    return "? unclassified"

col_table = pd.DataFrame({"column": df.columns, "group": [group_of(c) for c in df.columns]}).sort_values(["group", "column"])
assert not (col_table["group"] == "? unclassified").any(), "a processed column is not classified"
summary = (col_table.groupby("group")["column"].agg(n="size", columns=lambda s: ", ".join(s))
           .reset_index())
display(summary)
print(f"Every one of the {len(df.columns)} processed columns is in exactly one group: "
      f"{col_table['column'].is_unique and not (col_table['group'] == '? unclassified').any()}")
''')

code(r'''
# Feature selection exactly as the training script does it (redundancy screen + calendar-month rule).
parts = tm.chronological_split(df)
tr, va, te = parts["train"], parts["validation"], parts["test"]
t = time.time()
numeric, categorical, dropped = tm.select_features(parts, quick=False)
cols = numeric + categorical
print(f"select_features finished in {time.time() - t:.0f}s")
print(f"\nModel inputs ({len(cols)}): {len(numeric)} numeric + {len(categorical)} categorical")
print("  numeric    :", ", ".join(numeric))
print("  categorical:", ", ".join(categorical))
assert cols == saved["features"]["numeric"] + saved["features"]["categorical"], "feature list differs from the saved model"
print("-> identical to the feature list stored with the saved model: True")

drop_df = pd.DataFrame({"candidate": list(dropped), "why it is not a model input": list(dropped.values())})
display(drop_df)
''')

md("s2_leak_intro")

code(r'''
# DEMONSTRATION OF LEAKAGE -- NOT A MODEL.  Same pipeline and logistic regression as the real model,
# but with one or more outcome-side columns added to the allow-list.  These columns do not exist at purchase time.
variants = {
    "Honest: purchase-time allow-list only": [],
    "LEAK: + processing_days (seller hand-over time)": ["processing_days"],
    "LEAK: + carrier_to_customer_days": ["carrier_to_customer_days"],
    "LEAK: + delivery_duration_days (purchase -> delivery)": ["delivery_duration_days"],
}
rows = []
for label, extra in variants.items():
    ncols = numeric + extra
    pipe = tm.build_pipeline(LogisticRegression(max_iter=3000, C=1.0, random_state=config.RANDOM_STATE), ncols, categorical, scale=True)
    pipe.fit(tr[ncols + categorical], tr[tm.TARGET])
    row = {"variant": label}
    for wname, part in (("validation", va), ("test", te)):
        p = pipe.predict_proba(part[ncols + categorical])[:, 1]
        row[f"{wname} ROC-AUC"] = roc_auc_score(part[tm.TARGET], p)
        row[f"{wname} PR-AUC"] = average_precision_score(part[tm.TARGET], p)
    rows.append(row)
leak = pd.DataFrame(rows).set_index("variant")
display(leak)

QUOTED["leak_honest_test_auc"] = f"{leak.iloc[0]['test ROC-AUC']:.3f}"
QUOTED["leak_proc_test_auc"] = f"{leak.iloc[1]['test ROC-AUC']:.3f}"
QUOTED["leak_proc_test_pr"] = f"{leak.iloc[1]['test PR-AUC']:.3f}"
QUOTED["leak_c2c_test_auc"] = f"{leak.iloc[2]['test ROC-AUC']:.3f}"
QUOTED["leak_dur_test_auc"] = f"{leak.iloc[3]['test ROC-AUC']:.3f}"
QUOTED["leak_dur_test_pr"] = f"{leak.iloc[3]['test PR-AUC']:.3f}"
QUOTED["leak_dur_val_auc"] = f"{leak.iloc[3]['validation ROC-AUC']:.3f}"

fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9), sharey=False)
short = ["allow-list\nonly", "+ processing_days", "+ carrier_to\n_customer_days", "+ delivery_\nduration_days"]
for ax, metric, base in ((axes[0], "test ROC-AUC", 0.5), (axes[1], "test PR-AUC", te[tm.TARGET].mean())):
    colors = [P["blue"]] + [P["red"]] * 3
    bars = ax.bar(short, leak[metric].to_numpy(), color=colors, width=0.62)
    for b, v in zip(bars, leak[metric]):
        ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=9, color=P["slate"])
    ax.axhline(base, color=P["slate"], ls="--", lw=1, label=f"no skill ({base:.3f})")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_ylabel(metric.replace("test ", "Test-window ") + " (higher = better)")
    ax.set_title(metric.replace("test ", "Test-window "))
    ax.grid(axis="x", visible=False); ax.set_ylim(0, 1.08)
fig.suptitle("DEMONSTRATION OF LEAKAGE - not a model. Red bars use information that does not exist at purchase time.",
             x=0.01, ha="left", fontsize=10.5, color=P["red"], fontweight="bold")
save_show(fig, "31_leakage_demonstration")
''')

md("s2_interp")

# =============================================================================
# 3. Chronological split
# =============================================================================
md("s3_intro")

code(r'''
notes, monthly = tm.split_notes(df, parts)
win = pd.DataFrame({k: tm._window_summary(v) for k, v in parts.items()}).T
win["late_rate"] = win["late_rate"].astype(float)
win["window rule (start inclusive, end exclusive)"] = [f"{tm.TRAIN_START} .. {tm.VALIDATION_START}", f"{tm.VALIDATION_START} .. {tm.TEST_START}", f"{tm.TEST_START} .. end of data"]
display(win[["window rule (start inclusive, end exclusive)", "start", "end", "n", "n_late", "late_rate"]])

n_before = int((df["order_purchase_timestamp"] < tm.TRAIN_START).sum())
print(f"\nOrders purchased before {tm.TRAIN_START} (excluded, too sparse): {n_before:,}")
print(f"Total modelled orders: {sum(len(v) for v in parts.values()):,}  (+ {n_before} excluded = {n_all:,})")
assert sum(len(v) for v in parts.values()) + n_before == n_all
for k, v in parts.items():
    assert v["order_id"].is_unique
assert tr["order_purchase_timestamp"].max() < va["order_purchase_timestamp"].min()
assert va["order_purchase_timestamp"].max() < te["order_purchase_timestamp"].min()
print("Strictly chronological, no overlap between windows: True")
for k in ("train", "validation", "test"):
    s = saved["split"][k]
    assert (s["n"], s["n_late"]) == (len(parts[k]), int(parts[k][tm.TARGET].sum())), k
print("Counts identical to models/model_metrics.json: True")
for k, v in parts.items():
    QUOTED[f"{k}_n"] = f"{len(v):,}"
    QUOTED[f"{k}_n_late"] = f"{int(v[tm.TARGET].sum()):,}"
    QUOTED[f"{k}_rate"] = pct(v[tm.TARGET].mean(), 1)
QUOTED["n_before"] = f"{n_before:,}"
''')

code(r'''
mo = pd.DataFrame(monthly)
mo = mo[mo["month"] >= tm.TRAIN_START[:7]].reset_index(drop=True)
pos = np.arange(len(mo))
def window_of(m):
    return "train" if m < tm.VALIDATION_START[:7] else ("validation" if m < tm.TEST_START[:7] else "test")
mo["window"] = mo["month"].map(window_of)
wcol = {"train": P["blue"], "validation": P["amber"], "test": P["red"]}
z = 1.96
k = mo["late_rate"] * mo["n"]
phat = mo["late_rate"]
den = 1 + z**2 / mo["n"]
ctr = (phat + z**2 / (2 * mo["n"])) / den
hw = z * np.sqrt(phat * (1 - phat) / mo["n"] + z**2 / (4 * mo["n"] ** 2)) / den

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11.5, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
b1 = pos[mo["window"] == "validation"][0] - 0.5
b2 = pos[mo["window"] == "test"][0] - 0.5
for a in (ax, ax2):
    a.axvspan(-0.5, b1, color=P["blue"], alpha=0.05)
    a.axvspan(b1, b2, color=P["amber"], alpha=0.10)
    a.axvspan(b2, pos[-1] + 0.5, color=P["red"], alpha=0.07)
    a.axvline(b1, color=P["slate"], lw=1, ls="--"); a.axvline(b2, color=P["slate"], lw=1, ls="--")
ax.errorbar(pos, mo["late_rate"], yerr=[mo["late_rate"] - (ctr - hw), (ctr + hw) - mo["late_rate"]], fmt="none", ecolor=P["slate_light"], lw=1, capsize=2)
ax.plot(pos, mo["late_rate"], color=P["slate_light"], lw=1)
ax.scatter(pos, mo["late_rate"], c=[wcol[w] for w in mo["window"]], s=38, zorder=3)
for w, ls in (("train", None), ("validation", None), ("test", None)):
    m = mo["window"] == w
    ax.hlines(parts[w][tm.TARGET].mean(), pos[m][0] - 0.45, pos[m][-1] + 0.45, color=wcol[w], lw=2.2, zorder=2)
    ax.text(pos[m].mean(), 0.295, f"{w}\n{parts[w][tm.TARGET].mean():.1%} late\nn={len(parts[w]):,}",
            ha="center", va="top", fontsize=8.5, color=wcol[w], fontweight="bold")
top = mo.sort_values("late_rate", ascending=False).head(3)
for _, r in top.iterrows():
    ax.annotate(f"{r['month']}: {r['late_rate']:.1%} (n={int(r['n']):,})", (pos[mo['month'] == r['month']][0], r["late_rate"]),
                xytext=(-8, 4), textcoords="offset points", fontsize=8.5, color=P["navy"], ha="right")
ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0)); ax.set_ylim(0, 0.30)
ax.set_ylabel("Late rate of delivered orders\n(by purchase month; bars = 95% Wilson interval)")
ax.set_title("Late rate is not stationary: the three evaluation windows sit in different delay regimes")
ax2.bar(pos, mo["n"], color=[wcol[w] for w in mo["window"]], width=0.7)
ax2.set_ylabel("Orders per month (n)"); ax2.set_xlabel("Purchase month (dashed lines = validation and test start)")
ax2.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
ax2.set_xticks(pos); ax2.set_xticklabels(mo["month"], rotation=60, ha="right", fontsize=8.5)
save_show(fig, "32_monthly_late_rate_with_splits")
print("Months plotted: ", len(mo), "| smallest monthly n:", int(mo["n"].min()))
''')

code(r'''
print("Notes generated by the training script from the data (split_notes):\n")
for i, n_ in enumerate(notes, 1):
    wrap(f"{i}. {n_}")
    print()
''')

code(r'''
# Is the shift also visible in the features themselves?  Per window: promise length, same-state share, and the
# same-state vs other-state late-rate gap (descriptive, computed from the labelled data).
rows = []
for k_, v in parts.items():
    g = v.groupby(v["same_state"].astype(int))[tm.TARGET].agg(["mean", "size"])
    rows.append({"window": k_, "n": len(v), "median promise (days)": v["estimated_delivery_days"].astype(float).median(),
                 "same-state share": (v["same_state"].astype(int) == 1).mean(),
                 "late rate, same-state": g.loc[1, "mean"], "late rate, other-state": g.loc[0, "mean"],
                 "gap (same - other, pp)": 100 * (g.loc[1, "mean"] - g.loc[0, "mean"])})
drift = pd.DataFrame(rows).set_index("window")
display(drift)
for k_ in drift.index:
    QUOTED[f"gap_{k_}"] = f"{drift.loc[k_, 'gap (same - other, pp)']:+.1f}"
    QUOTED[f"promise_{k_}"] = f"{drift.loc[k_, 'median promise (days)']:.0f} days"

shift = tm.support_shift(parts, numeric)
for k_, v in shift.items():
    print(f"{k_:<11}: {pct(v['share_outside'], 2):>6} of {v['n']:,} orders have a numeric feature outside the TRAIN range "
          f"(n={v['n_outside']:,}, late rate {pct(v['late_rate_outside'], 1)} vs {pct(v['late_rate_inside'], 1)} for the rest)")
QUOTED["test_outside_share"] = pct(shift["test"]["share_outside"], 1)
QUOTED["test_outside_n"] = f"{shift['test']['n_outside']:,}"
QUOTED["test_outside_rate"] = pct(shift["test"]["late_rate_outside"], 1)
QUOTED["test_inside_rate"] = pct(shift["test"]["late_rate_inside"], 1)
''')

code(r'''
# Why a random split would flatter the model: shuffle the SAME 2017-01..2018-08 orders into 80 % / 20 %
# and train the SAME logistic regression.  (Demonstration only; this is NOT how the model is evaluated.)
pool = pd.concat([tr, va, te])
rtr, rte = train_test_split(pool, test_size=0.20, stratify=pool[tm.TARGET], random_state=config.RANDOM_STATE)
rand_pipe = tm.build_pipeline(LogisticRegression(max_iter=3000, C=1.0, random_state=config.RANDOM_STATE), numeric, categorical, scale=True)
rand_pipe.fit(rtr[cols], rtr[tm.TARGET])
pr_ = rand_pipe.predict_proba(rte[cols])[:, 1]
chron_pipe = tm.build_pipeline(LogisticRegression(max_iter=3000, C=1.0, random_state=config.RANDOM_STATE), numeric, categorical, scale=True)
chron_pipe.fit(tr[cols], tr[tm.TARGET])
pc_ = chron_pipe.predict_proba(te[cols])[:, 1]
cmp_split = pd.DataFrame({
    "evaluation": ["random 80/20 split of all modelled orders", "chronological: train on earliest, test on latest window"],
    "test n": [len(rte), len(te)], "test late rate": [rte[tm.TARGET].mean(), te[tm.TARGET].mean()],
    "ROC-AUC": [roc_auc_score(rte[tm.TARGET], pr_), roc_auc_score(te[tm.TARGET], pc_)],
    "PR-AUC": [average_precision_score(rte[tm.TARGET], pr_), average_precision_score(te[tm.TARGET], pc_)],
    "PR-AUC lift over no-skill": [average_precision_score(rte[tm.TARGET], pr_) / rte[tm.TARGET].mean(), average_precision_score(te[tm.TARGET], pc_) / te[tm.TARGET].mean()],
})
display(cmp_split.set_index("evaluation"))
QUOTED["rand_auc"] = f"{cmp_split.loc[0, 'ROC-AUC']:.3f}"
QUOTED["rand_lift"] = f"{cmp_split.loc[0, 'PR-AUC lift over no-skill']:.1f}"
QUOTED["chron_auc"] = f"{cmp_split.loc[1, 'ROC-AUC']:.3f}"
QUOTED["chron_lift"] = f"{cmp_split.loc[1, 'PR-AUC lift over no-skill']:.1f}"
''')

md("s3_interp")

# =============================================================================
# 4. Pipeline
# =============================================================================
md("s4_intro")

code(r'''
lr_template = tm.build_pipeline(LogisticRegression(max_iter=3000, C=1.0, random_state=config.RANDOM_STATE), numeric, categorical, scale=True)
print(lr_template)
''')

code(r'''
steps = pd.DataFrame([
    ("coerce", "prediction.coerce_feature_frame", "numeric -> float64, categorical -> plain strings; raises if a column is missing", "stateless"),
    ("num / impute", "SimpleImputer(median, add_indicator=True)", "fill missing numbers with the TRAIN median; add a 0/1 'was missing' column where any value was missing in train", "fitted on train only"),
    ("num / scale", "StandardScaler (logistic regression only)", "mean 0, SD 1 so the L2 penalty treats features alike; trees skip this step", "fitted on train only"),
    ("cat / impute", "SimpleImputer(constant 'missing')", "an unknown category becomes its own level", "stateless"),
    ("cat / onehot", f"OneHotEncoder(min_frequency={tm.MIN_FREQUENCY}, handle_unknown='ignore')", "one 0/1 column per frequent level; rarer levels are pooled into 'infrequent'; unseen levels become all-zero", "levels learned on train only"),
    ("model", "Dummy / LogisticRegression / RandomForest / HistGradientBoosting", "the estimator being compared", "fitted on train only"),
], columns=["step", "component", "what it does", "state learned from"])
display(steps)

lr_template.fit(tr[cols], tr[tm.TARGET])
prep = lr_template.named_steps["prep"]
Xt = prep.transform(lr_template.named_steps["coerce"].transform(tr[cols]))
names = prep.get_feature_names_out()
print(f"\nInput columns: {len(cols)}   ->   design-matrix columns after preprocessing: {Xt.shape[1]}  (train rows: {Xt.shape[0]:,})")
kinds = pd.Series([n_.split("__")[0] + ("_missing_flag" if "missingindicator" in n_ else "") for n_ in names]).value_counts()
display(kinds.rename("columns").to_frame())
QUOTED["design_cols"] = f"{Xt.shape[1]} design-matrix columns"
''')

code(r'''
# Evidence that nothing is learned from validation / test data: the fitted imputer medians equal the TRAIN medians.
imp = prep.named_transformers_["num"].named_steps["impute"]
fitted_med = pd.Series(imp.statistics_, index=numeric)
train_med = tr[numeric].astype("float64").median()
miss = pd.DataFrame({"missing share (train)": tr[numeric].astype("float64").isna().mean(),
                     "median used for imputation": fitted_med,
                     "median of train": train_med,
                     "median of test (for contrast)": te[numeric].astype("float64").median()})
display(miss[miss["missing share (train)"] > 0].assign(**{"n missing (train)": lambda d: (d["missing share (train)"] * len(tr)).round().astype(int)}))
assert np.allclose(fitted_med.to_numpy(), train_med.reindex(numeric).to_numpy(), equal_nan=True)
print("Imputer medians == train-window medians: True")

oh = prep.named_transformers_["cat"].named_steps["onehot"]
rows = []
for c, levels in zip(categorical, oh.categories_):
    vc = tr[c].astype(str).value_counts()
    rows.append({"feature": c, "levels in train": len(levels), "levels kept as own column": int((vc >= tm.MIN_FREQUENCY).sum()),
                 "levels pooled as 'infrequent'": int((vc < tm.MIN_FREQUENCY).sum()),
                 "train orders in pooled levels": int(vc[vc < tm.MIN_FREQUENCY].sum())})
display(pd.DataFrame(rows).set_index("feature"))
''')

md("s4_interp")

# =============================================================================
# 5. Models
# =============================================================================
md("s5_intro")

code(r'''
yva, yte, ytr = va[tm.TARGET].to_numpy(), te[tm.TARGET].to_numpy(), tr[tm.TARGET].to_numpy()
fitted: dict[str, dict] = {}
t = time.time()
dummy = tm.build_pipeline(DummyClassifier(strategy="prior"), numeric, categorical, scale=False).fit(tr[cols], ytr)
fitted["dummy_prior"] = {"base": dummy, "best_params": {"strategy": "prior"}, "grid": [],
                         "calibration": {"estimator": dummy, "calibrated": False, "method": None,
                                         "diagnostics": {"reason": "constant prior probability; nothing to calibrate"}}}
tuning_note = {}
for name, (factory, full_grid) in tm.candidate_estimators(quick=False).items():
    t1 = time.time()
    if name == "logistic_regression":
        grid, tuning_note[name] = full_grid, f"full grid of {len(full_grid)} settings re-run here"
    else:   # the tree ensembles are the slow ones: refit only the saved best setting (grid results are shown from the JSON)
        grid, tuning_note[name] = [saved["models"][name]["best_params"]], f"saved best setting refitted (grid of {len(full_grid)} settings read from model_metrics.json)"
    res = tm.fit_and_tune(name, factory, grid, parts, numeric, categorical)
    if name == "logistic_regression":
        assert res["best_params"] == saved["models"][name]["best_params"], "tuning disagrees with the saved model"
    res["calibration"] = tm.calibrate(res["base"], parts, cols)
    fitted[name] = res
    print(f"{name:<24} fitted in {time.time() - t1:5.1f}s | {tuning_note[name]} | best {res['best_params']} | calibration: {res['calibration']['method']}")
print(f"\nAll models fitted in {time.time() - t:.0f}s")
''')

code(r'''
# Hyper-parameter grids: validation PR-AUC per setting (logistic regression: re-run here; others: saved results).
rows = []
for name in ("logistic_regression", "random_forest", "hist_gradient_boosting"):
    grid = fitted[name]["grid"] if name == "logistic_regression" else saved["models"][name]["tuning_grid"]
    for g in grid:
        rows.append({"model": tm.MODEL_LABELS[name], "setting": json.dumps(g["params"]), "validation PR-AUC": g["validation_pr_auc"],
                     "chosen": g["params"] == saved["models"][name]["best_params"]})
grid_df = pd.DataFrame(rows)
display(grid_df.style.apply(lambda r: ["font-weight: bold; background-color: #DBEAFE" if r["chosen"] else "" for _ in r], axis=1)
        .format({"validation PR-AUC": "{:.4f}"}).hide(axis="index"))
lr_rerun = pd.DataFrame(fitted["logistic_regression"]["grid"])
lr_saved = pd.DataFrame(saved["models"]["logistic_regression"]["tuning_grid"])
print("Logistic-regression grid re-run vs saved grid: max |difference| in validation PR-AUC =",
      f"{np.abs(lr_rerun['validation_pr_auc'].to_numpy() - lr_saved['validation_pr_auc'].to_numpy()).max():.2e}")

# Calibration decision per model (judged on the LATER half of validation, fitted on the earlier half)
rows = []
for name in ("logistic_regression", "random_forest", "hist_gradient_boosting"):
    d = fitted[name]["calibration"]["diagnostics"]
    rows.append({"model": tm.MODEL_LABELS[name], "raw Brier": d["raw"]["brier"], "sigmoid Brier": d["sigmoid"]["brier"],
                 "isotonic Brier": d["isotonic"]["brier"], "decision": d["decision"]})
display(pd.DataFrame(rows).set_index("model"))
''')

code(r'''
# Scores on both windows with the deployment rules of the training script: calibrated estimator, exact-zero floor,
# F1-optimal threshold chosen on VALIDATION only.
val_raw = {n_: r["calibration"]["estimator"].predict_proba(va[cols])[:, 1] for n_, r in fitted.items()}
floors = {n_: tm.probability_floor(val_raw[n_], yva) for n_ in fitted}
val_scores = {n_: np.maximum(val_raw[n_], floors[n_]) for n_ in fitted}
test_scores = {n_: tm._proba(r["calibration"]["estimator"], te[cols], floors[n_]) for n_, r in fitted.items()}
test_base = {n_: r["base"].predict_proba(te[cols])[:, 1] for n_, r in fitted.items()}
val_base = {n_: r["base"].predict_proba(va[cols])[:, 1] for n_, r in fitted.items()}
thresholds = {n_: (0.5 if n_ == "dummy_prior" else tm.f1_optimal_threshold(yva, val_scores[n_])) for n_ in fitted}

met = {n_: {"validation": tm.classification_metrics(yva, val_scores[n_], thresholds[n_]),
            "test": tm.classification_metrics(yte, test_scores[n_], thresholds[n_])} for n_ in fitted}

# Reconcile with the numbers stored in models/model_metrics.json (written by `python -m src.train_model`).
keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier", "log_loss"]
rows, worst_by_model = [], {}
for n_ in fitted:
    for w in ("validation", "test"):
        dev_ = max(abs(met[n_][w][k_] - saved["models"][n_][w][k_]) for k_ in keys)
        worst_by_model[n_] = max(worst_by_model.get(n_, 0.0), dev_)
        rows.append({"model": tm.MODEL_LABELS[n_], "window": w, "this notebook: PR-AUC": met[n_][w]["pr_auc"],
                     "saved: PR-AUC": saved["models"][n_][w]["pr_auc"], "this notebook: ROC-AUC": met[n_][w]["roc_auc"],
                     "saved: ROC-AUC": saved["models"][n_][w]["roc_auc"], "max |diff| over 8 metrics": dev_})
display(pd.DataFrame(rows).set_index(["model", "window"]).style.format("{:.4f}", subset=pd.IndexSlice[:, ["this notebook: PR-AUC", "saved: PR-AUC", "this notebook: ROC-AUC", "saved: ROC-AUC"]]).format("{:.1e}", subset=["max |diff| over 8 metrics"]))
for n_ in fitted:
    assert abs(thresholds[n_] - saved["models"][n_]["threshold"]) < 1e-9, n_
print("Largest deviation from the saved metrics, per model:", {k_: f"{v_:.1e}" for k_, v_ in worst_by_model.items()})
worst = max(v_ for k_, v_ in worst_by_model.items() if k_ != "random_forest")
assert worst < 1e-9, f"notebook disagrees with the saved metrics (max diff {worst})"
assert worst_by_model["random_forest"] < 1e-3, "random forest deviates by more than float-summation noise"
print("Dummy, logistic regression and gradient boosting reproduce models/model_metrics.json to floating-point precision.")
print("The random forest agrees to within the small deviation explained below.")
''')

code(r'''
# Why the random forest is not bit-identical: its predict_proba sums tree outputs across threads, so the last bits of each
# probability depend on thread timing (~1e-16).  The isotonic calibrator then sorts / pools tied scores, which amplifies that
# noise to ~1e-4 in the metrics.  Re-running the SAME calibration on the SAME forest shows it:
rf_base = fitted["random_forest"]["base"]
p_a = rf_base.predict_proba(va[cols])[:, 1]; p_b = rf_base.predict_proba(va[cols])[:, 1]
print(f"two predict_proba calls on the same forest: max |difference| = {np.abs(p_a - p_b).max():.1e}")
reps = []
for _ in range(3):
    c_ = tm.calibrate(rf_base, parts, cols)["estimator"]
    reps.append(average_precision_score(yva, c_.predict_proba(va[cols])[:, 1]))
print("validation PR-AUC of the re-calibrated forest, three repeats:", ", ".join(f"{v_:.5f}" for v_ in reps),
      f"| saved: {saved['models']['random_forest']['validation']['pr_auc']:.5f}")
print(f"spread between repeats: {max(reps) - min(reps):.1e}  (same order as the deviation from the saved file: {worst_by_model['random_forest']:.1e})")
''')

md("s5_interp")

# =============================================================================
# 6. Evaluation
# =============================================================================
md("s6_intro")

code(r'''
def metric_table(window: str) -> pd.DataFrame:
    rows = {}
    for n_ in fitted:
        m = met[n_][window]
        rows[tm.MODEL_LABELS[n_]] = {"accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"], "F1": m["f1"],
                                     "ROC-AUC": m["roc_auc"], "PR-AUC": m["pr_auc"], "Brier": m["brier"], "log-loss": m["log_loss"],
                                     "flagged share": m["predicted_positive_rate"], "lift in top 10 %": m["lift_top10pct"]}
    return pd.DataFrame(rows).T

print(f"TEST window (n={len(te):,}, {pct(yte.mean(), 2)} late) - thresholds frozen on validation; never used for fitting or selection")
display(metric_table("test"))
print(f"VALIDATION window (n={len(va):,}, {pct(yva.mean(), 2)} late) - tuned / in-sample for the selected model, NOT an unbiased estimate")
display(metric_table("validation"))

# Uncertainty on the test window: paired bootstrap of ROC-AUC / PR-AUC (same resamples for every model).
t = time.time()
boot_te = tm.bootstrap_scores(yte, test_scores, 300, config.RANDOM_STATE + 1)
ci = pd.DataFrame({tm.MODEL_LABELS[n_]: {"test ROC-AUC 95% interval": f"{tm._ci(boot_te[n_]['roc_auc'])[0]:.3f} - {tm._ci(boot_te[n_]['roc_auc'])[1]:.3f}",
                                           "test PR-AUC 95% interval": f"{tm._ci(boot_te[n_]['pr_auc'])[0]:.3f} - {tm._ci(boot_te[n_]['pr_auc'])[1]:.3f}"} for n_ in fitted}).T
display(ci)
print(f"bootstrap done in {time.time() - t:.0f}s;  no-skill PR-AUC on test = late rate = {yte.mean():.3f}")
for n_ in ("logistic_regression", "hist_gradient_boosting"):
    assert [round(v, 9) for v in tm._ci(boot_te[n_]["pr_auc"])] == [round(v, 9) for v in saved["models"][n_]["test_ci95"]["pr_auc"]]
print("Bootstrap intervals identical to the saved ones (logistic regression, gradient boosting): True")
ci_lr_auc, ci_lr_pr = tm._ci(boot_te["logistic_regression"]["roc_auc"]), tm._ci(boot_te["logistic_regression"]["pr_auc"])

L, D = met["logistic_regression"], met["dummy_prior"]
for w in ("validation", "test"):
    QUOTED[f"lr_{w}_auc"] = f"{L[w]['roc_auc']:.3f}"
    QUOTED[f"lr_{w}_pr"] = f"{L[w]['pr_auc']:.3f}"
    QUOTED[f"lr_{w}_acc"] = pct(L[w]["accuracy"], 1)
    QUOTED[f"lr_{w}_prec"] = pct(L[w]["precision"], 1)
    QUOTED[f"lr_{w}_rec"] = pct(L[w]["recall"], 1)
    QUOTED[f"lr_{w}_f1"] = f"{L[w]['f1']:.3f}"
    QUOTED[f"lr_{w}_brier"] = f"{L[w]['brier']:.4f}"
    QUOTED[f"lr_{w}_flag"] = pct(L[w]["predicted_positive_rate"], 1)
    QUOTED[f"lr_{w}_lift10"] = f"{L[w]['lift_top10pct']:.1f}"
QUOTED["dummy_test_acc"] = pct(D["test"]["accuracy"], 1)
QUOTED["dummy_test_brier"] = f"{D['test']['brier']:.4f}"
QUOTED["dummy_val_pr"] = f"{D['validation']['pr_auc']:.3f}"
QUOTED["dummy_test_pr"] = f"{D['test']['pr_auc']:.3f}"
QUOTED["lr_test_auc_ci"] = f"{ci_lr_auc[0]:.3f} - {ci_lr_auc[1]:.3f}"
QUOTED["lr_test_pr_ci"] = f"{ci_lr_pr[0]:.3f} - {ci_lr_pr[1]:.3f}"
# the random forest is quoted from the saved file (its last digits move with thread timing, see section 5)
for n_, key in (("random_forest", "rf"), ("hist_gradient_boosting", "hgb")):
    src_ = saved["models"][n_]
    QUOTED[f"{key}_test_auc"] = f"{src_['test']['roc_auc']:.3f}"
    QUOTED[f"{key}_test_pr"] = f"{src_['test']['pr_auc']:.3f}"
    QUOTED[f"{key}_val_auc"] = f"{src_['validation']['roc_auc']:.3f}"
    QUOTED[f"{key}_val_pr"] = f"{src_['validation']['pr_auc']:.3f}"
''')

code(r'''
# Accuracy is misleading at ~5 % prevalence: compare "always predict on time" with the model at its validation threshold.
acc = pd.DataFrame({
    "late orders found (recall)": [met["dummy_prior"]["test"]["recall"], met["logistic_regression"]["test"]["recall"]],
    "late orders found (n)": [met["dummy_prior"]["test"]["confusion_matrix"][1][1], met["logistic_regression"]["test"]["confusion_matrix"][1][1]],
    "accuracy": [met["dummy_prior"]["test"]["accuracy"], met["logistic_regression"]["test"]["accuracy"]],
}, index=["Always predict 'on time' (dummy)", "Logistic regression at the validation-chosen threshold"])
display(acc)
QUOTED["lr_test_tp"] = f"{met['logistic_regression']['test']['confusion_matrix'][1][1]:,}"
QUOTED["test_n_late"] = f"{int(yte.sum()):,}"
''')

code(r'''
# Confusion matrices on the TEST window (threshold frozen on validation).
fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
for ax, n_ in zip(axes, fitted):
    cm = np.array(met[n_]["test"]["confusion_matrix"])
    rate = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(rate, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({rate[i, j]:.1%} of row)", ha="center", va="center", fontsize=9,
                    color="white" if rate[i, j] > 0.55 else P["navy"])
    ax.set_xticks([0, 1], ["predicted\non time", "predicted\nlate"]); ax.set_yticks([0, 1], ["actually\non time", "actually\nlate"])
    ax.grid(False)
    ax.set_title(f"{tm.MODEL_LABELS[n_]}\nthreshold {thresholds[n_]:.3f}", fontsize=10)
fig.suptitle(f"Confusion matrices, test window (n={len(te):,}); cell shows count and share of the true class", x=0.01, ha="left", fontsize=11, color=P["slate"])
save_show(fig, "33_confusion_matrices_test")
''')

code(r'''
# ROC, precision-recall and calibration curves on the TEST window (all models, final deployed scores).
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))
ax = axes[0]
for n_ in fitted:
    fpr, tpr, _ = roc_curve(yte, test_scores[n_])
    ax.plot(fpr, tpr, color=MODEL_COLORS[n_], lw=2 if n_ == "logistic_regression" else 1.4, ls="--" if n_ == "dummy_prior" else "-",
            label=f"{tm.MODEL_LABELS[n_]} (AUC {met[n_]['test']['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], color=P["border"], lw=1)
ax.set_xlabel("False-positive rate (share of on-time orders flagged)"); ax.set_ylabel("True-positive rate (share of late orders flagged)")
ax.set_title("ROC curve, test window"); ax.legend(loc="lower right", fontsize=8)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)

ax = axes[1]
for n_ in ("logistic_regression", "random_forest", "hist_gradient_boosting"):
    pr_, rc_, _ = precision_recall_curve(yte, test_scores[n_])
    ax.plot(rc_, pr_, color=MODEL_COLORS[n_], lw=2 if n_ == "logistic_regression" else 1.4,
            label=f"{tm.MODEL_LABELS[n_]} (PR-AUC {met[n_]['test']['pr_auc']:.3f})")
ax.axhline(yte.mean(), color=P["slate"], ls="--", lw=1.2, label=f"no skill / dummy = late rate ({yte.mean():.3f}; PR-AUC {met['dummy_prior']['test']['pr_auc']:.3f})")
ax.set_xlabel("Recall (share of late orders found)"); ax.set_ylabel("Precision (share of flagged orders that are late)")
ax.set_title("Precision-recall, test (y-axis clipped at 0.35)"); ax.legend(loc="upper right", fontsize=8)
ax.set_xlim(0, 1); ax.set_ylim(0, 0.35)

ax = axes[2]
mx = 0
for n_ in ("logistic_regression", "random_forest", "hist_gradient_boosting"):
    c_ = tm.curve_payloads(yte, test_scores[n_])["cal"]
    ax.plot(c_["mean_predicted"], c_["fraction_positive"], marker="o", ms=4.5, color=MODEL_COLORS[n_], lw=1.6, label=tm.MODEL_LABELS[n_])
    mx = max(mx, max(c_["mean_predicted"]), max(c_["fraction_positive"]))
    if n_ == "logistic_regression":   # bins are NOT equal-sized (tied scores, duplicates='drop'): show n so small bins are visible
        for x_, y_, k_ in zip(c_["mean_predicted"], c_["fraction_positive"], c_["bin_counts"]):
            if k_ < 600:   # only the small bins are labelled (the others hold 1,000-2,000 orders each)
                ax.annotate(f"n={k_:,}", (x_, y_), xytext=(5, -11), textcoords="offset points", fontsize=8, color=MODEL_COLORS[n_], fontweight="bold")
ax.plot([0, mx * 1.05], [0, mx * 1.05], color=P["slate"], ls="--", lw=1)
ax.text(mx * 1.0, mx * 0.93, "perfect calibration", rotation=30, fontsize=8.5, color=P["slate"], ha="right")
ax.set_xlabel("Mean predicted late probability (score bin; uneven sizes, n shown for bins under 600)"); ax.set_ylabel("Observed late rate in the score bin")
ax.set_title("Reliability, test window"); ax.legend(loc="upper left", fontsize=8)
ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0)); ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
# Top (highest-scored) bin of the deployed model on test: small, so quote its size and a Wilson 95% interval.
_c = tm.curve_payloads(yte, test_scores["logistic_regression"])["cal"]
_n, _p = int(_c["bin_counts"][-1]), float(_c["fraction_positive"][-1])
_z = 1.959964; _den = 1 + _z**2 / _n
_ctr = (_p + _z**2 / (2 * _n)) / _den; _half = _z * np.sqrt(_p * (1 - _p) / _n + _z**2 / (4 * _n**2)) / _den
print(f"Top score bin (logistic regression, test): n={_n:,} of {len(yte):,} orders ({_n / len(yte):.1%}); "
      f"mean predicted {_c['mean_predicted'][-1]:.1%}, observed {_p:.1%} (~{round(_p * _n)} late orders), Wilson 95% interval {_ctr - _half:.1%} - {_ctr + _half:.1%}")
print("Bin sizes:", _c["bin_counts"])
QUOTED["top_bin_n"] = f"n={_n:,}"
QUOTED["top_bin_pred"] = pct(_c["mean_predicted"][-1], 1)
QUOTED["top_bin_obs"] = pct(_p, 1)
QUOTED["top_bin_ci"] = f"{_ctr - _half:.1%} - {_ctr + _half:.1%}"
fig.suptitle(f"Test window, n={len(te):,}, {pct(yte.mean(), 1)} late. Scores are those used in the app (calibrated, zero-floored).", x=0.01, ha="left", fontsize=10.5, color=P["slate"])
save_show(fig, "34_roc_pr_calibration_test")
''')

code(r'''
# The selected model on validation vs test: the same code, the same threshold, two windows.
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
sel_name = "logistic_regression"
for w, ys, ps, col, ls in (("validation (tuned, in-sample)", yva, val_scores[sel_name], P["amber"], "-"), ("test (out-of-sample)", yte, test_scores[sel_name], P["red"], "-")):
    fpr, tpr, _ = roc_curve(ys, ps); axes[0].plot(fpr, tpr, color=col, lw=2, label=f"{w}: AUC {roc_auc_score(ys, ps):.3f}")
    pr_, rc_, _ = precision_recall_curve(ys, ps); axes[1].plot(rc_, pr_, color=col, lw=2, label=f"{w}: PR-AUC {average_precision_score(ys, ps):.3f}")
    axes[1].axhline(ys.mean(), color=col, ls=":", lw=1)
    c_ = tm.curve_payloads(ys, ps)["cal"]; axes[2].plot(c_["mean_predicted"], c_["fraction_positive"], marker="o", ms=4.5, color=col, lw=1.6, label=w)
    for x_, y_, k_ in zip(c_["mean_predicted"], c_["fraction_positive"], c_["bin_counts"]):
        if k_ < 600:   # only the small bins are labelled
            axes[2].annotate(f"n={k_:,}", (x_, y_), xytext=(5, 5 if col == P["amber"] else -11), textcoords="offset points", fontsize=8, color=col, fontweight="bold")
axes[0].plot([0, 1], [0, 1], color=P["border"]); axes[0].set_xlabel("False-positive rate"); axes[0].set_ylabel("True-positive rate"); axes[0].set_title("ROC"); axes[0].legend(loc="lower right", fontsize=8)
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision (dotted = no-skill late rate of that window)"); axes[1].set_title("Precision-recall"); axes[1].legend(loc="upper right", fontsize=8); axes[1].set_ylim(0, 0.45)
axes[2].plot([0, 0.32], [0, 0.32], color=P["slate"], ls="--", lw=1); axes[2].set_xlabel("Mean predicted late probability (score bin; uneven sizes, n shown for bins under 600)"); axes[2].set_ylabel("Observed late rate in the score bin")
axes[2].set_title("Calibration"); axes[2].legend(loc="upper left", fontsize=8)
axes[2].xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0)); axes[2].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
fig.suptitle("Logistic regression (selected model): validation vs test - the ranking signal and the calibration both weaken on the later window", x=0.01, ha="left", fontsize=10.5, color=P["slate"])
save_show(fig, "35_selected_model_validation_vs_test")
''')

md("s6_interp")

# =============================================================================
# 7. Threshold and risk bands
# =============================================================================
md("s7_intro")

code(r'''
sel = "logistic_regression"
ps_v, ps_t = val_scores[sel], test_scores[sel]
th = thresholds[sel]
bands_th = tm.risk_thresholds(ps_v)
assert bands_th == saved["risk_bands"]["thresholds"], "risk-band cut points differ from the saved model"

def prf_curve(ys, ps, grid):
    prec, rec, f1, flag = [], [], [], []
    for g in grid:
        pred_ = ps >= g
        tp = int((pred_ & (ys == 1)).sum()); fp = int((pred_ & (ys == 0)).sum()); fn = int((~pred_ & (ys == 1)).sum())
        p_ = tp / (tp + fp) if tp + fp else np.nan; r_ = tp / (tp + fn) if tp + fn else 0.0
        prec.append(p_); rec.append(r_); f1.append(2 * p_ * r_ / (p_ + r_) if tp else 0.0); flag.append(pred_.mean())
    return np.array(prec), np.array(rec), np.array(f1), np.array(flag)

grid = np.linspace(0.02, 0.30, 141)
pv, rv, fv, gv = prf_curve(yva, ps_v, grid)
pt, rt, ft, gt = prf_curve(yte, ps_t, grid)
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), sharey=True)
for ax, (p_, r_, f_, ttl, ls) in zip(axes, ((pv, rv, fv, "Validation window (used to choose the threshold)", "-"), (pt, rt, ft, "Test window (shown for contrast only, NOT used)", "-"))):
    ax.plot(grid, p_, color=P["blue"], lw=2, label="precision"); ax.plot(grid, r_, color=P["green"], lw=2, label="recall"); ax.plot(grid, f_, color=P["navy"], lw=2, ls="--", label="F1")
    ax.axvline(th, color=P["red"], lw=1.4); ax.text(th + 0.003, 0.9, f"F1-optimal on validation\nthreshold = {th:.3f}", color=P["red"], fontsize=8.5)
    for name_, c_ in (("moderate", P["amber"]), ("high", P["red"])):
        ax.axvline(bands_th[name_], color=c_, lw=1, ls=":")
        ax.text(bands_th[name_] - 0.002, 0.52 if name_ == "moderate" else 0.62, f"{name_} band\ncut {bands_th[name_]:.3f}", color=c_, fontsize=8, ha="right")
    ax.set_title(ttl, fontsize=11); ax.set_xlabel("Probability threshold (flag order as late if score >= threshold)")
    ax.set_ylim(0, 1.0); ax.legend(loc="upper right")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
axes[0].set_ylabel("Precision / recall / F1")
save_show(fig, "36_threshold_selection")

tab = pd.DataFrame({"validation": {"flagged share of orders": met[sel]["validation"]["predicted_positive_rate"], "precision": met[sel]["validation"]["precision"],
                                   "recall": met[sel]["validation"]["recall"], "F1": met[sel]["validation"]["f1"], "no-skill precision (late rate)": yva.mean()},
                    "test": {"flagged share of orders": met[sel]["test"]["predicted_positive_rate"], "precision": met[sel]["test"]["precision"],
                             "recall": met[sel]["test"]["recall"], "F1": met[sel]["test"]["f1"], "no-skill precision (late rate)": yte.mean()}})
print(f"Operating point: flag an order as late when probability >= {th:.4f}  (the F1-optimal cut on validation)")
display(tab)
''')

code(r'''
val_bands = tm.band_table(yva, ps_v, bands_th)
test_bands = tm.band_table(yte, ps_t, bands_th)
assert [round(b["late_rate"], 9) for b in val_bands] == [round(b["late_rate"], 9) for b in saved["risk_bands"]["validation"]]
assert [round(b["late_rate"], 9) for b in test_bands] == [round(b["late_rate"], 9) for b in saved["risk_bands"]["test"]]
vd, td = tm.band_diagnostics(val_bands), tm.band_diagnostics(test_bands)

rows = []
for v, t_ in zip(val_bands, test_bands):
    rows.append({"band": v["band"],
                 "validation n": v["n"], "validation share": vd["shares"][v["band"]], "validation late n": v["n_late"], "validation late rate": v["late_rate"],
                 "validation 95% interval": f"{pct(v['late_ci_low'], 1)} - {pct(v['late_ci_high'], 1)}",
                 "test n": t_["n"], "test share": td["shares"][t_["band"]], "test late n": t_["n_late"], "test late rate": t_["late_rate"],
                 "test 95% interval": f"{pct(t_['late_ci_low'], 1)} - {pct(t_['late_ci_high'], 1)}"})
band_df = pd.DataFrame(rows).set_index("band")
display(band_df)
print("Validation:", vd["note"]); print("Test      :", td["note"])

for b in test_bands:
    key = b["band"].lower()
    QUOTED[f"tb_{key}_n"] = f"{b['n']:,}"
    QUOTED[f"tb_{key}_rate"] = pct(b["late_rate"], 1)
    QUOTED[f"tb_{key}_ci"] = f"{pct(b['late_ci_low'], 1)} - {pct(b['late_ci_high'], 1)}"
    QUOTED[f"tb_{key}_share"] = pct(td["shares"][b["band"]], 1)
for b in val_bands:
    key = b["band"].lower()
    QUOTED[f"vb_{key}_rate"] = pct(b["late_rate"], 1)
    QUOTED[f"vb_{key}_share"] = pct(vd["shares"][b["band"]], 1)
QUOTED["cut_moderate"] = f"{bands_th['moderate']:.3f}"
QUOTED["cut_high"] = f"{bands_th['high']:.3f}"
QUOTED["f1_threshold"] = f"{th:.3f}"

fig, ax = plt.subplots(figsize=(10.5, 4.4))
w = 0.36
x = np.arange(3)
for off, bands, lab, col in ((-w / 2, val_bands, "validation (cuts chosen here)", P["amber"]), (w / 2, test_bands, "test (out-of-sample)", P["red"])):
    r_ = np.array([b["late_rate"] for b in bands]); lo_ = np.array([b["late_ci_low"] for b in bands]); hi_ = np.array([b["late_ci_high"] for b in bands])
    bars = ax.bar(x + off, r_, w, color=col, alpha=0.9, label=lab, yerr=[r_ - lo_, hi_ - r_], capsize=4, error_kw={"ecolor": P["navy"], "lw": 1})
    for b_, bd, rr, hh in zip(bars, bands, r_, hi_):
        ax.text(b_.get_x() + b_.get_width() / 2, max(hh, 0.052) + 0.004, f"{rr:.1%}\nn={bd['n']:,}", ha="center", va="bottom", fontsize=8.5, color=P["navy"])
ax.axhline(yva.mean(), color=P["amber"], ls=":", lw=1.5, label=f"validation overall late rate ({yva.mean():.1%})")
ax.axhline(yte.mean(), color=P["red"], ls=":", lw=1.5, label=f"test overall late rate ({yte.mean():.1%})")
ax.set_xticks(x, ["Low risk", "Moderate risk", "High risk"]); ax.set_ylim(0, 0.30)
ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0)); ax.set_ylabel("Observed late rate (bars: 95% Wilson interval)")
ax.set_title("Observed late rate per risk band: Low is clearly lower on test, Moderate vs High is barely separated")
ax.legend(loc="upper left"); ax.grid(axis="x", visible=False)
save_show(fig, "37_risk_bands_validation_vs_test")
''')

code(r'''
# Does the pooled band picture hide a group effect?  Same-state vs other-state orders on the TEST window (descriptive).
bd = tm.same_state_breakdown(yte, ps_t, te["same_state"].to_numpy(), bands_th)
rows = []
for g, lab in (("same_state", "same-state (seller and customer in one state)"), ("other_state", "other-state")):
    for band in ("Low", "Moderate", "High"):
        c_ = bd[g]["bands"][band]
        rows.append({"group": lab, "band": band, "n": c_["n"], "late n": c_["n_late"], "observed late rate": c_["late_rate"],
                     "95% interval": f"{pct(c_['late_ci_low'], 1)} - {pct(c_['late_ci_high'], 1)}"})
display(pd.DataFrame(rows).set_index(["group", "band"]))
for g in ("same_state", "other_state"):
    print(f"{g:<12}: n={bd[g]['n']:,}  observed late rate {pct(bd[g]['observed_late_rate'], 1)}  vs mean predicted {pct(bd[g]['mean_predicted'], 1)}")
assert bd == saved["risk_bands"]["test_by_same_state"] or abs(bd["same_state"]["observed_late_rate"] - saved["risk_bands"]["test_by_same_state"]["same_state"]["observed_late_rate"]) < 1e-9
QUOTED["ss_n"] = f"{bd['same_state']['n']:,}"
QUOTED["ss_obs"] = pct(bd["same_state"]["observed_late_rate"], 1)
QUOTED["ss_pred"] = pct(bd["same_state"]["mean_predicted"], 1)
QUOTED["os_n"] = f"{bd['other_state']['n']:,}"
QUOTED["os_obs"] = pct(bd["other_state"]["observed_late_rate"], 1)
QUOTED["os_pred"] = pct(bd["other_state"]["mean_predicted"], 1)
''')

md("s7_interp")

# =============================================================================
# 8. Importance and coefficients
# =============================================================================
md("s8_intro")

code(r'''
t = time.time()
est_sel = fitted[sel]["calibration"]["estimator"]
pi_v = permutation_importance(est_sel, va[cols], yva, scoring="average_precision", n_repeats=10, random_state=config.RANDOM_STATE, n_jobs=1)
pi_t = permutation_importance(est_sel, te[cols], yte, scoring="average_precision", n_repeats=10, random_state=config.RANDOM_STATE, n_jobs=1)
imp_df = pd.DataFrame({"validation": pi_v.importances_mean, "validation_sd": pi_v.importances_std,
                       "test": pi_t.importances_mean, "test_sd": pi_t.importances_std}, index=cols).sort_values("validation", ascending=False)
print(f"permutation importance done in {time.time() - t:.0f}s")
display(imp_df)
sv = {i["feature"]: i["importance_mean"] for i in saved["permutation_importance"]["items"]}
st_ = {i["feature"]: i["importance_mean"] for i in saved["permutation_importance"]["test_items"]}
assert max(abs(imp_df.loc[f, "validation"] - sv[f]) for f in cols) < 1e-9 and max(abs(imp_df.loc[f, "test"] - st_[f]) for f in cols) < 1e-9
print("Identical to the permutation importances stored in model_metrics.json: True")
flips = [f for f in cols if np.sign(imp_df.loc[f, "validation"]) != np.sign(imp_df.loc[f, "test"])]
print("Features whose importance changes sign between validation and test:", ", ".join(flips))
# Many of those flips are noise around zero.  A flip counts as CLEAR only if the importance is more than 2 SD (of the 10 shuffles)
# away from zero on BOTH windows, in opposite directions.
clear_flips = [f for f in flips if abs(imp_df.loc[f, "validation"]) > 2 * imp_df.loc[f, "validation_sd"] and abs(imp_df.loc[f, "test"]) > 2 * imp_df.loc[f, "test_sd"]]
big_flips = [f for f in clear_flips if abs(imp_df.loc[f, "test"]) > 0.01]
print("  ... of which more than 2 SD from zero on both windows:", ", ".join(clear_flips) or "none")
print("  ... of which also larger than 0.01 PR-AUC on test:", ", ".join(big_flips) or "none")

order = imp_df.index[::-1]
fig, ax = plt.subplots(figsize=(10.5, 6.2))
yy = np.arange(len(order)); h = 0.38
ax.barh(yy + h / 2, imp_df.loc[order, "validation"], h, xerr=imp_df.loc[order, "validation_sd"], color=P["amber"], label="validation (10 shuffles)", error_kw={"lw": 0.8})
ax.barh(yy - h / 2, imp_df.loc[order, "test"], h, xerr=imp_df.loc[order, "test_sd"], color=P["red"], label="test (10 shuffles; diagnostic only)", error_kw={"lw": 0.8})
ax.axvline(0, color=P["navy"], lw=0.8)
ax.set_yticks(yy, order); ax.set_xlabel("Drop in PR-AUC when the column is shuffled (positive = the model relied on it; negative = shuffling helped)")
ax.set_title("Permutation importance (selected model): predictive association, not cause")
ax.legend(loc="lower right"); ax.grid(axis="y", visible=False)
save_show(fig, "38_permutation_importance")

top_v, top_t = imp_df["validation"].idxmax(), imp_df["test"].idxmax()
QUOTED["pi_top_val"] = f"{top_v} ({imp_df.loc[top_v, 'validation']:.3f})"
QUOTED["pi_top_test"] = f"{top_t} ({imp_df.loc[top_t, 'test']:.3f})"
QUOTED["pi_same_val"] = f"{imp_df.loc['same_state', 'validation']:+.3f}"
QUOTED["pi_same_test"] = f"{imp_df.loc['same_state', 'test']:+.3f}"
QUOTED["pi_cust_val"] = f"{imp_df.loc['customer_state', 'validation']:.3f}"
QUOTED["pi_seller_test"] = f"{imp_df.loc['seller_state', 'test']:.3f}"
QUOTED["pi_n_flips"] = f"{len(flips)} of {len(cols)} features"
QUOTED["pi_n_clear"] = f"{len(clear_flips)} of those {len(flips)}"
QUOTED["pi_big_flips"] = ", ".join(f"`{f}`" for f in big_flips)
''')

code(r'''
# Coefficients of the (uncalibrated) logistic regression.  Numeric columns are standardised (coefficient = change in log-odds of
# "late" per +1 SD).  Every one-hot level has its own column (no dropped reference level) and the L2 penalty pulls the levels of one
# feature towards each other, so only DIFFERENCES between levels of the same feature are interpretable (e.g. same_state_1 vs same_state_0).
lr_model = fitted[sel]["base"].named_steps["model"]
nm = fitted[sel]["base"].named_steps["prep"].get_feature_names_out()
coef = pd.DataFrame({"column": nm, "coefficient (log-odds)": lr_model.coef_[0]})
coef["odds ratio"] = np.exp(coef["coefficient (log-odds)"])
# number of TRAIN orders at each one-hot level (column sums of the 0/1 design matrix), to show how thinly supported the extreme levels are
Xc = fitted[sel]["base"].named_steps["prep"].transform(fitted[sel]["base"].named_steps["coerce"].transform(tr[cols]))
coef["n train orders at level"] = np.where(coef["column"].str.startswith("cat__"), np.asarray(Xc.sum(axis=0)).ravel().round(), np.nan)
coef["n train orders at level"] = coef["n train orders at level"].astype("Int64")
coef = coef.sort_values("coefficient (log-odds)")
print(f"{len(coef)} coefficients; intercept {lr_model.intercept_[0]:.3f}")
display(pd.concat([coef.head(8).assign(side="lowest (lower late odds)"), coef.tail(8).iloc[::-1].assign(side="highest (higher late odds)")]).set_index("side"))
ssc = coef[coef["column"].str.contains("same_state")]
display(ssc)
c1 = float(coef.loc[coef["column"] == "cat__same_state_1", "coefficient (log-odds)"].iloc[0])
c0 = float(coef.loc[coef["column"] == "cat__same_state_0", "coefficient (log-odds)"].iloc[0])
print(f"same_state: level 1 minus level 0 = {c1 - c0:+.3f} log-odds  ->  odds ratio {np.exp(c1 - c0):.2f} (same-state vs other-state, all else equal, learned on the TRAIN window)")
QUOTED["same_state_or"] = f"{np.exp(c1 - c0):.2f}"
lo_row, hi_row = coef.iloc[0], coef.iloc[-1]
QUOTED["coef_lowest"] = f"`{lo_row['column'].replace('cat__', '')}` ({lo_row['coefficient (log-odds)']:+.2f})"
QUOTED["coef_highest"] = f"`{hi_row['column'].replace('cat__', '')}` ({hi_row['coefficient (log-odds)']:+.2f}, {int(hi_row['n train orders at level']):,} train orders)"
ext = pd.concat([coef.head(8), coef.tail(8)])
QUOTED["coef_ext_kinds"] = f"{ext['column'].str.replace('cat__', '').str.extract(r'^(customer_state|seller_state|product_category|same_state)')[0].nunique()} different features"
''')

md("s8_interp")

# =============================================================================
# 9. Selection + distribution shift
# =============================================================================
md("s9_intro")

code(r'''
t = time.time()
contenders = [n_ for n_ in fitted if n_ != "dummy_prior"]
val_pr = {n_: float(average_precision_score(yva, val_base[n_])) for n_ in fitted}
boot_va = tm.bootstrap_scores(yva, val_base, 300, config.RANDOM_STATE)
best, within, chosen = tm.choose_model(val_pr, {n_: boot_va[n_]["pr_auc"] for n_ in contenders})
rows = []
for n_ in fitted:
    c_ = tm._ci(boot_va[n_]["pr_auc"])
    rows.append({"model": tm.MODEL_LABELS[n_], "validation PR-AUC (uncalibrated)": val_pr[n_], "95% bootstrap interval": f"{c_[0]:.3f} - {c_[1]:.3f}",
                 "simplicity rank (1 = simplest)": tm.SIMPLICITY_RANK.get(n_, "-"), "within noise of the best": n_ in within,
                 "test PR-AUC (looked at AFTER the decision)": met[n_]["test"]["pr_auc"], "test ROC-AUC": met[n_]["test"]["roc_auc"]})
sel_df = pd.DataFrame(rows).set_index("model")
display(sel_df)
print(f"bootstrap done in {time.time() - t:.0f}s")
print(f"\nBest validation PR-AUC: {tm.MODEL_LABELS[best]};  within one paired-bootstrap SD of it: {[tm.MODEL_LABELS[n_] for n_ in within]};  chosen (simplest of those): {tm.MODEL_LABELS[chosen]}")
assert chosen == saved["selected_model"]["name"] == sel
print("Same choice as the saved model:", chosen == saved["selected_model"]["name"])
print()
wrap("Saved rationale: " + saved["selected_model"]["rationale"])
QUOTED["val_pr_lr_unc"] = f"{val_pr['logistic_regression']:.3f}"
QUOTED["val_pr_rf_unc"] = f"{val_pr['random_forest']:.3f}"
QUOTED["val_pr_hgb_unc"] = f"{val_pr['hist_gradient_boosting']:.3f}"
QUOTED["test_pr_lr_unc"] = f"{average_precision_score(yte, test_base['logistic_regression']):.3f}"
''')

code(r'''
# How the shift shows up for the selected model: window-level calibration and ranking (train rows are IN-SAMPLE).
def sel_score(frame):
    return tm._proba(est_sel, frame[cols], floors[sel])

p_tr = sel_score(tr)
rows = []
for lab, part, ps in (("train (in-sample)", tr, p_tr), ("validation (tuned)", va, ps_v), ("test (out-of-sample)", te, ps_t)):
    yy_ = part[tm.TARGET].to_numpy()
    rows.append({"window": lab, "n": len(part), "observed late rate": yy_.mean(), "mean predicted risk": ps.mean(),
                 "ROC-AUC": roc_auc_score(yy_, ps), "PR-AUC": average_precision_score(yy_, ps),
                 "lift (PR-AUC / late rate)": average_precision_score(yy_, ps) / yy_.mean(), "Brier": float(np.mean((ps - yy_) ** 2)),
                 "Brier of constant (train prior)": float(np.mean((tr[tm.TARGET].mean() - yy_) ** 2))})
shift_df = pd.DataFrame(rows).set_index("window")
display(shift_df)
QUOTED["train_insample_auc"] = f"{shift_df.iloc[0]['ROC-AUC']:.3f}"
QUOTED["test_meanpred"] = pct(shift_df.loc["test (out-of-sample)", "mean predicted risk"], 1)
QUOTED["val_meanpred"] = pct(shift_df.loc["validation (tuned)", "mean predicted risk"], 1)
QUOTED["test_brier_const"] = f"{shift_df.loc['test (out-of-sample)', 'Brier of constant (train prior)']:.4f}"
QUOTED["test_lift"] = f"{shift_df.loc['test (out-of-sample)', 'lift (PR-AUC / late rate)']:.1f}"
QUOTED["val_lift"] = f"{shift_df.loc['validation (tuned)', 'lift (PR-AUC / late rate)']:.1f}"

# monthly observed vs predicted
allm = pd.concat([tr.assign(_p=p_tr), va.assign(_p=ps_v), te.assign(_p=ps_t)])
mm = (allm.assign(month=allm["order_purchase_timestamp"].dt.strftime("%Y-%m")).groupby("month")
      .agg(n=(tm.TARGET, "size"), observed=(tm.TARGET, "mean"), predicted=("_p", "mean")).reset_index())
mm["window"] = mm["month"].map(window_of)
fig, ax = plt.subplots(figsize=(11.5, 4.6))
pp = np.arange(len(mm))
ax.axvspan(-0.5, b1, color=P["blue"], alpha=0.05); ax.axvspan(b1, b2, color=P["amber"], alpha=0.10); ax.axvspan(b2, pp[-1] + 0.5, color=P["red"], alpha=0.07)
ax.plot(pp, mm["observed"], color=P["navy"], marker="o", ms=5, lw=1.8, label="observed late rate")
ax.plot(pp, mm["predicted"], color=P["blue"], marker="s", ms=4.5, lw=1.8, ls="--", label="mean predicted risk (selected model)")
ax.set_xticks(pp, mm["month"], rotation=60, ha="right", fontsize=8.5)
ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0)); ax.set_ylabel("Late rate / mean predicted probability")
ax.set_xlabel("Purchase month (shaded: train | validation | test windows; train months are in-sample)")
ax.set_title("Observed late rate swings with the delay regime; mean predicted risk moves much less")
ax.set_yticks(np.arange(0, 0.21, 0.02))
ax.legend(loc="upper left")
save_show(fig, "39_observed_vs_predicted_monthly")
corr_train = float(np.corrcoef(mm.loc[mm["window"] == "train", "observed"], mm.loc[mm["window"] == "train", "predicted"])[0, 1])
print(f"Correlation of monthly observed vs monthly mean predicted risk inside the training window (n={int((mm['window'] == 'train').sum())} months): {corr_train:.2f}")
QUOTED["corr_train_months"] = f"{corr_train:.2f}"
mar = mm.loc[mm["month"] == "2018-03"].iloc[0]
QUOTED["mar18_obs"] = pct(mar["observed"], 1)
QUOTED["mar18_pred"] = pct(mar["predicted"], 1)
''')

code(r'''
# Test window split by whether the order's promise/size lies inside the TRAINING range (extrapolation vs interpolation).
tr_num = tr[numeric].astype("float64")
lo_, hi_ = tr_num.min(), tr_num.max()
te_num = te[numeric].astype("float64")
outside = ((te_num < lo_) | (te_num > hi_)).any(axis=1).to_numpy()
rows = []
for lab, m in (("inside the training feature range", ~outside), ("outside the training range (mostly shorter promises)", outside)):
    yy_ = yte[m]
    ok = len(np.unique(yy_)) == 2
    rows.append({"test orders": lab, "n": int(m.sum()), "n late": int(yy_.sum()), "observed late rate": yy_.mean(),
                 "mean predicted": ps_t[m].mean(), "ROC-AUC": roc_auc_score(yy_, ps_t[m]) if ok else np.nan,
                 "small sample (n < 100)?": int(m.sum()) < config.DEFAULT_MIN_ORDERS})
sup = pd.DataFrame(rows).set_index("test orders")
display(sup)
print(f"estimated_delivery_days: training range {lo_['estimated_delivery_days']:.0f}-{hi_['estimated_delivery_days']:.0f} days; "
      f"share of test orders below the training minimum: {(te_num['estimated_delivery_days'] < lo_['estimated_delivery_days']).mean():.1%}")
QUOTED["sup_in_auc"] = f"{sup.iloc[0]['ROC-AUC']:.3f}"
QUOTED["sup_out_auc"] = f"{sup.iloc[1]['ROC-AUC']:.3f}"
QUOTED["sup_out_meanpred"] = pct(sup.iloc[1]["mean predicted"], 1)
QUOTED["sup_out_obs"] = pct(sup.iloc[1]["observed late rate"], 1)
QUOTED["sup_in_obs"] = pct(sup.iloc[0]["observed late rate"], 1)
QUOTED["promise_min_train"] = f"{lo_['estimated_delivery_days']:.0f} to {hi_['estimated_delivery_days']:.0f} days"
''')

md("s9_interp")

# =============================================================================
# 10. Using the saved bundle
# =============================================================================
md("s10_intro")

code(r'''
bundle = pred.load_model()
print("Bundle keys:", ", ".join(sorted(bundle)))
print(f"\nmodel_name={bundle['model_name']} | calibrated={bundle['calibrated']} | decision_threshold={bundle['decision_threshold']:.4f} | "
      f"risk_thresholds={ {k: round(v, 4) for k, v in bundle['risk_thresholds'].items()} } | trained with scikit-learn {bundle['sklearn_version']}")
print("train window:", bundle["train_window"])

# 1. The bundle is the notebook's model: same scores on every test order (after the same exact-zero floor).
bundle_scores = pred._score(bundle["pipeline"], te, bundle["feature_columns"], float(bundle["probability_floor"]))
bundle_dev = float(np.max(np.abs(bundle_scores - ps_t)))
print(f"\nmax |bundle score - notebook score| over {len(te):,} test orders: {bundle_dev:.2e}")
assert bundle_dev < 1e-9
''')

code(r'''
# 2. predict_risk on three REAL orders from the TEST window, one drawn at random (fixed seed) from each risk band of the
#    deployed model, then compared with what actually happened.  Illustration only - three orders prove nothing.
rng = np.random.default_rng(config.RANDOM_STATE)
band_of = np.where(ps_t >= bands_th["high"], "High", np.where(ps_t >= bands_th["moderate"], "Moderate", "Low"))
picked = [int(rng.choice(np.flatnonzero(band_of == b))) for b in ("Low", "Moderate", "High")]
fields = ["customer_state", "seller_state", "product_category", "n_items", "total_price", "total_freight", "n_sellers",
          "total_weight_g", "total_volume_cm3", "max_product_length_cm", "max_product_height_cm", "max_product_width_cm", "estimated_delivery_days"]
rows, notes_ = [], []
for i in picked:
    r = te.iloc[i]
    inputs = {f: (None if pd.isna(r[f]) else (r[f].item() if hasattr(r[f], "item") else r[f])) for f in fields}
    inputs["purchase_date"] = r["order_purchase_timestamp"].strftime("%Y-%m-%d")
    out = pred.predict_risk(inputs, bundle)
    assert abs(out["probability"] - ps_t[i]) < 1e-9
    rows.append({"order_id": r["order_id"][:8] + "...", "route": f"{r['seller_state']} -> {r['customer_state']}", "category": r["product_category"],
                 "promised days": int(r["estimated_delivery_days"]), "items": int(r["n_items"]), "freight / price": f"{r['total_freight']:.0f} / {r['total_price']:.0f}",
                 "predicted late probability": out["probability"], "risk band": out["risk_label"],
                 "ACTUAL outcome": f"{'LATE' if r[tm.TARGET] == 1 else 'on time'} ({int(r['delivery_delay_days']):+d} days vs promise)"})
    notes_.append((out["risk_label"], out["probability_note"], out["warnings"]))
    QUOTED[f"demo_p_{out['risk_label'].lower()}"] = f"{out['probability']:.3f}"
    QUOTED[f"demo_outcome_{out['risk_label'].lower()}"] = f"{int(r['delivery_delay_days']):+d} days"
demo = pd.DataFrame(rows)
display(demo)
print("predict_risk(...) equals the notebook score for all three orders: True\n")
for lab, note, warn in notes_:
    print(f"[{lab} band example] non-blocking input warnings: {warn if warn else 'none'}")
print("\nThe probability note that accompanies every prediction (identical text for the three, apart from the band-specific sentence):")
wrap(notes_[0][1])
''')

code(r'''
# 3. What the app's form does: only the required fields are given, the rest is filled from training defaults (and reported).
minimal = {"customer_state": "SP", "seller_state": "SP", "product_category": "health_beauty", "n_items": 1,
           "total_price": 90.0, "total_freight": 15.0, "purchase_date": "2018-08-10"}
out = pred.predict_risk(minimal, bundle)
print(f"Minimal input -> probability {out['probability']:.3f}, band {out['risk_label']}, calibrated={out['calibrated']}")
print("Fields filled from training defaults:", ", ".join(out["assumptions_used"]) or "none")
wrap("Probability note: " + str(out["probability_note"]))
try:
    pred.predict_risk({**minimal, "customer_state": "XX"}, bundle)
except ValueError as exc:
    print("\nInvalid input is rejected with a readable message:", exc)
''')

md("s10_interp")

# =============================================================================
# 11. Conclusion
# =============================================================================
md("s11")

code(r'''
checks = [
    ("features equal the saved model's feature list", cols == saved["features"]["numeric"] + saved["features"]["categorical"]),
    ("window counts equal the saved split", all(len(parts[k]) == saved["split"][k]["n"] for k in parts)),
    ("dummy, logistic regression and gradient boosting reproduce the saved metrics exactly; the forest within noise", worst < 1e-9 and worst_by_model["random_forest"] < 1e-3),
    ("model choice equals the saved choice", chosen == saved["selected_model"]["name"]),
    ("risk-band cuts equal the saved cuts", bands_th == saved["risk_bands"]["thresholds"]),
    ("saved leakage audit passed", saved["leakage_audit"]["passed"]),
    ("bundle scores equal notebook scores on the test window", bundle_dev < 1e-9),
]
display(pd.DataFrame(checks, columns=["check", "passed"]).set_index("check"))
assert all(ok for _, ok in checks)
print(f"Notebook wall-clock time: {time.time() - T0:.0f}s")
QUOTED["runtime_note"] = "ok"
''')

code(r'''
# Appendix: every number quoted in the Markdown, as computed above (the build script checks the text against this table).
print("###QUOTED###" + json.dumps(QUOTED))
pd.Series(QUOTED, name="value").to_frame()
''')

# =============================================================================
# Build + execute
# =============================================================================
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}


def extract_quoted(nb_: "nbf.NotebookNode") -> dict[str, str]:
    for c in nb_.cells:
        if c.cell_type != "code":
            continue
        for o in c.get("outputs", []):
            txt = o.get("text", "")
            if MARK in txt:
                return json.loads(txt.split(MARK, 1)[1].strip().splitlines()[0])
    raise AssertionError("appendix cell output not found")


def check_narrative(nb_: "nbf.NotebookNode") -> None:
    """Every quoted number must appear in the Markdown; no markdown key may be left pending."""
    quoted = extract_quoted(nb_)
    text = "\n".join(c.source for c in nb_.cells if c.cell_type == "markdown")
    pending = [c.source[:60] for c in nb_.cells if c.cell_type == "markdown" and "pending)*" in c.source]
    missing = {k: v for k, v in quoted.items() if k != "runtime_note" and v not in text}
    assert not pending, f"pending narrative: {pending}"
    assert not missing, f"quoted values not found in the narrative (text drifted or key unused): {missing}"
    print(f"narrative check OK: {len(quoted)} computed values all appear in the Markdown")


if __name__ == "__main__":
    import time
    t0 = time.time()
    client = NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(HERE)}})
    client.execute()
    nbf.write(nb, NB_PATH)
    print(f"executed and wrote {NB_PATH} in {time.time() - t0:.1f}s")
    if "--no-check" not in sys.argv:
        check_narrative(nb)
