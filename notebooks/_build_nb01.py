"""Generate and execute notebooks/01_data_exploration.ipynb.

Usage (from anywhere):  .venv/bin/python notebooks/_build_nb01.py
The narrative text lives in notebooks/_nb01_text.py; code cells live in this file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

from _nb01_text import MD  # type: ignore  # same directory

HERE = Path(__file__).resolve().parent
NB_PATH = HERE / "01_data_exploration.ipynb"

cells: list = []


def md(key: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(MD[key].strip("\n")))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# =============================================================================
# 0. Title + setup
# =============================================================================
md("title")

code('''
import sys, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter
from scipy import stats
from sklearn.metrics import roc_auc_score

# Resolve the project root whether the notebook is run from notebooks/ or from the root.
_here = Path.cwd().resolve()
ROOT = next((p for p in [_here, *_here.parents] if (p / "src" / "config.py").is_file()), None)
if ROOT is None:
    raise RuntimeError("Run this notebook from inside the DeliveryIQ project folder.")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.preprocessing import load_raw, build_orders_table, load_processed

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

P = config.PALETTE
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


def save_show(fig, name: str) -> None:
    """Save a figure as PNG under reports/figures/ (for GitHub) and display it inline."""
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    plt.show()


def wilson(k, n, z: float = 1.96):
    """Vectorised Wilson score interval for proportions (NaN where n == 0)."""
    k = np.asarray(k, dtype=float); n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


MIN_N = config.DEFAULT_MIN_ORDERS
print(f"Project root : {ROOT}")
print(f"Figures dir  : {FIG_DIR.relative_to(ROOT)}")
print(f"Minimum sample for a 'reliable' segment: n >= {MIN_N}")
''')

# =============================================================================
# 2. Dataset understanding
# =============================================================================
md("s2_intro")

code('''
raw_dir = config.RAW_DIR
files = {**config.REQUIRED_RAW_FILES, **config.OPTIONAL_RAW_FILES}
# Plain pandas read (default dtype inference) -- this is exploration, not the pipeline.
tables = {name: pd.read_csv(raw_dir / fn) for name, fn in files.items()}
orders, items, customers, sellers, products = (tables[k] for k in
                                               ["orders", "items", "customers", "sellers", "products"])
reviews, translation = tables["reviews"], tables["category_translation"]

overview = pd.DataFrame({
    "file": [files[k] for k in tables],
    "rows": [len(t) for t in tables.values()],
    "columns": [t.shape[1] for t in tables.values()],
    "size_MB": [round((raw_dir / files[k]).stat().st_size / 1e6, 2) for k in tables],
}, index=list(tables))
overview
''')

code('''
for name in ["orders", "items"]:
    print(f"===== {name} =====")
    tables[name].info()
    print()
''')

code('''
for name in ["customers", "sellers", "products", "reviews", "category_translation"]:
    print(f"===== {name} =====")
    tables[name].info()
    print()
''')

code('''
orders.head(3).T
''')

code('''
display(items.head(3).T)
display(products.head(3).T)
display(reviews.drop(columns=["review_comment_message"]).head(3).T)
''')

md("er")

code('''
def coverage(child: pd.Series, parent: pd.Series) -> dict:
    """Share of non-null child keys that exist in the parent key set."""
    c = child.dropna()
    return {"child_rows": len(c), "found_in_parent": int(c.isin(set(parent)).sum()),
            "orphans": int((~c.isin(set(parent))).sum())}

checks = {
    "orders.customer_id -> customers.customer_id": coverage(orders["customer_id"], customers["customer_id"]),
    "items.order_id -> orders.order_id": coverage(items["order_id"], orders["order_id"]),
    "items.product_id -> products.product_id": coverage(items["product_id"], products["product_id"]),
    "items.seller_id -> sellers.seller_id": coverage(items["seller_id"], sellers["seller_id"]),
    "reviews.order_id -> orders.order_id": coverage(reviews["order_id"], orders["order_id"]),
    "products.category -> translation": coverage(products["product_category_name"],
                                                  translation["product_category_name"]),
}
pd.DataFrame(checks).T
''')

code('''
keys = pd.DataFrame({
    "table": ["orders", "customers", "sellers", "products", "items", "reviews"],
    "key": ["order_id", "customer_id", "seller_id", "product_id", "(order_id, order_item_id)", "review_id"],
    "rows": [len(orders), len(customers), len(sellers), len(products), len(items), len(reviews)],
    "unique_key_values": [orders.order_id.nunique(), customers.customer_id.nunique(),
                          sellers.seller_id.nunique(), products.product_id.nunique(),
                          len(items.drop_duplicates(["order_id", "order_item_id"])),
                          reviews.review_id.nunique()],
})
keys["key_is_unique"] = keys["rows"] == keys["unique_key_values"]
display(keys)

print(f"customer_id values        : {customers.customer_id.nunique():,}")
print(f"customer_unique_id values : {customers.customer_unique_id.nunique():,}  "
      f"(a person can have several customer_id, one per order)")
print(f"orders per customer_id    : max = {orders.groupby('customer_id').size().max()}  (so customer_id <-> order is 1:1)")
rev_per_order = reviews.groupby("order_id").size()
print(f"orders with >1 review     : {(rev_per_order > 1).sum():,}  -> reduced to the latest review per order in the pipeline")
''')

md("s2_card")

code('''
n_orders_total = orders.order_id.nunique()
per_order = items.groupby("order_id").agg(n_items=("order_item_id", "size"),
                                          n_sellers=("seller_id", "nunique"),
                                          n_products=("product_id", "nunique"))
per_order = per_order.reindex(orders.order_id).fillna(0).astype(int)   # orders without item rows -> 0

def capped_counts(s: pd.Series, cap: int) -> pd.Series:
    c = s.clip(upper=cap).value_counts().sort_index()
    c.index = [f"{i}+" if i == cap else str(i) for i in c.index]
    return c

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, col, cap, ttl in [(axes[0], "n_items", 6, "Items per order"),
                          (axes[1], "n_sellers", 3, "Distinct sellers per order")]:
    c = capped_counts(per_order[col], cap)
    bars = ax.bar(c.index, c.values, color=P["blue"], width=0.65)
    ax.set_yscale("log")
    ax.set_title(f"{ttl} (log scale)")
    ax.set_xlabel(ttl.split(" per")[0] + (" (last bar = cap or more)"))
    ax.set_ylabel("Number of orders (log)")
    for b, v in zip(bars, c.values):
        ax.annotate(f"{v:,}\\n({v / n_orders_total:.1%})", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=8, color=P["slate"])
    ax.set_ylim(10, c.max() * 6)
fig.suptitle(f"Order cardinality, all {n_orders_total:,} raw orders", x=0.01, ha="left",
             fontsize=11, color=P["slate_light"])
save_show(fig, "01_order_cardinality")

summary = pd.Series({
    "orders": n_orders_total,
    "orders without any item row": int((per_order.n_items == 0).sum()),
    "orders with >= 2 items": int((per_order.n_items >= 2).sum()),
    "orders with >= 2 distinct sellers": int((per_order.n_sellers >= 2).sum()),
    "max items in one order": int(per_order.n_items.max()),
    "max sellers in one order": int(per_order.n_sellers.max()),
    "mean items per order (orders that have items)": round(per_order.loc[per_order.n_items > 0, "n_items"].mean(), 3),
})
summary.map(lambda v: f"{v:,}" if float(v).is_integer() else f"{v:.3f}").to_frame("value")
''')

md("s2_interp")

# =============================================================================
# 3. Data quality
# =============================================================================
md("s3_intro")

code('''
miss_rows = []
for name, t in tables.items():
    m = t.isna().sum()
    for col, k in m[m > 0].items():
        miss_rows.append({"table": name, "column": col, "missing": int(k), "rows": len(t), "pct_missing": 100 * k / len(t)})
missing = pd.DataFrame(miss_rows).sort_values("pct_missing", ascending=False).reset_index(drop=True)
display(missing)

lab = missing["table"] + "." + missing["column"]
fig, ax = plt.subplots(figsize=(9, 0.38 * len(missing) + 1.2))
ax.barh(lab[::-1], missing["pct_missing"][::-1], color=P["blue"], height=0.6)
for y, (pct, k) in enumerate(zip(missing["pct_missing"][::-1], missing["missing"][::-1])):
    ax.text(pct + 1, y, f"{pct:.2f}%  (n={k:,})", va="center", fontsize=8.5, color=P["slate"])
ax.set_xlim(0, 105); ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
ax.set_xlabel("Share of rows missing"); ax.set_title("Missing values by table and column (raw files)")
ax.grid(axis="y", visible=False)
save_show(fig, "02_missing_values")
''')

code('''
dup_rows = []
key_map = {"orders": ["order_id"], "items": ["order_id", "order_item_id"], "customers": ["customer_id"],
           "sellers": ["seller_id"], "products": ["product_id"], "reviews": ["review_id", "order_id"],
           "category_translation": ["product_category_name"]}
for name, t in tables.items():
    dup_rows.append({"table": name, "rows": len(t), "exact_duplicate_rows": int(t.duplicated().sum()),
                     "duplicate_keys": int(t.duplicated(key_map[name]).sum()), "key": ", ".join(key_map[name])})
dup_df = pd.DataFrame(dup_rows)
display(dup_df)
print("Reviews: repeated order_id =", int(reviews.order_id.duplicated().sum()),
      "| repeated review_id =", int(reviews.review_id.duplicated().sum()),
      "(a review_id can be attached to several orders; handled by keeping the latest review per order)")
''')

code('''
ts_cols = config.TIMESTAMP_COLS
parse_rows = []
op = orders.copy()
for c in ts_cols:
    raw_missing = int(op[c].isna().sum())
    parsed = pd.to_datetime(op[c], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    parse_rows.append({"column": c, "raw_missing": raw_missing,
                       "unparseable_non_missing": int(parsed.isna().sum() - raw_missing),
                       "earliest": parsed.min(), "latest": parsed.max()})
    op[c] = parsed
display(pd.DataFrame(parse_rows).set_index("column"))
print("Estimated-delivery timestamps all at midnight? ->",
      bool((op.order_estimated_delivery_date.dt.normalize() == op.order_estimated_delivery_date).all()))
''')

code('''
T = op
anoms = {
    "approved before purchase": T.order_approved_at < T.order_purchase_timestamp,
    "carrier pickup before purchase": T.order_delivered_carrier_date < T.order_purchase_timestamp,
    "carrier pickup before approval": T.order_delivered_carrier_date < T.order_approved_at,
    "customer delivery before carrier pickup": T.order_delivered_customer_date < T.order_delivered_carrier_date,
    "customer delivery before purchase": T.order_delivered_customer_date < T.order_purchase_timestamp,
    "estimate before purchase": T.order_estimated_delivery_date < T.order_purchase_timestamp,
}
delivered_mask = T.order_status == "delivered"
anom_df = pd.DataFrame({
    "all orders": {k: int(v.sum()) for k, v in anoms.items()},
    "delivered orders": {k: int((v & delivered_mask).sum()) for k, v in anoms.items()},
})
display(anom_df)

# the gap size for the two anomalies that the pipeline handles explicitly
gap = (T.order_purchase_timestamp - T.order_delivered_carrier_date).dt.total_seconds() / 3600
print("Carrier-before-purchase rows: how many hours EARLIER than the purchase the carrier timestamp is:")
print(gap[gap > 0].describe(percentiles=[.5, .9]).round(1).to_string())
''')

md("s3_anom_interp")

code('''
status = orders.order_status.value_counts()
st_df = pd.DataFrame({"orders": status, "share_pct": 100 * status / status.sum()})
st_df["has_customer_delivery_date"] = orders.dropna(subset=["order_delivered_customer_date"]).order_status.value_counts().reindex(st_df.index).fillna(0).astype(int)
st_df["has_item_rows"] = orders[orders.order_id.isin(items.order_id)].order_status.value_counts().reindex(st_df.index).fillna(0).astype(int)
display(st_df)

fig, ax = plt.subplots(figsize=(8.5, 3.6))
s = status.sort_values()
ax.barh(s.index, s.values, color=[P["blue"] if i == "delivered" else P["slate_light"] for i in s.index], height=0.6)
ax.set_xscale("log")
for y, v in enumerate(s.values):
    ax.text(v * 1.08, y, f"{v:,}  ({v / status.sum():.2%})", va="center", fontsize=8.5, color=P["slate"])
ax.set_xlim(1, status.max() * 12)
ax.set_xlabel("Number of orders (log scale)"); ax.set_title("Order status distribution (all raw orders)")
ax.grid(axis="y", visible=False)
save_show(fig, "03_order_status")
''')

md("s3_exclusion")

code('''
# --- product-category missingness and zero weights -------------------------------------
cat_missing_products = products.product_category_name.isna().sum()
items_p = items.merge(products, on="product_id", how="left")
items_cat_missing = items_p.product_category_name.isna().sum()
print(f"Products with no category            : {cat_missing_products:,} of {len(products):,} ({cat_missing_products/len(products):.2%})")
print(f"Item rows whose product has no category: {items_cat_missing:,} of {len(items):,} ({items_cat_missing/len(items):.2%})")
not_in_tr = sorted(set(products.product_category_name.dropna()) - set(translation.product_category_name))
print(f"Categories absent from the translation file ({len(not_in_tr)}): {not_in_tr}")

w = products.product_weight_g
print(f"\\nProducts missing weight : {int(w.isna().sum())}   | with weight == 0 g : {int((w == 0).sum())}")
dims = ["product_length_cm", "product_height_cm", "product_width_cm"]
print("Products with a non-positive dimension:", int((products[dims] <= 0).any(axis=1).sum()))
print("Weight (g) quantiles:", products.product_weight_g.quantile([0, .01, .5, .99, 1]).round(0).to_dict())
display(products[products.product_weight_g == 0][["product_id", "product_category_name", "product_weight_g"]].assign(
    product_id=lambda d: d.product_id.str[:10] + "..."))
''')

md("s3_cleaning")

code('''
# --- independent recomputation vs the real pipeline ----------------------------------
raw = load_raw()                                  # the pipeline's own loader (all columns as str)
pipe_all, report = build_orders_table(raw)        # the pipeline's order-level table + quality report
proc = load_processed()                           # parquet written by `python -m src.preprocessing`

o = op                                            # parsed timestamps from the cells above
elig_nb = (o.order_status == "delivered") & o.order_delivered_customer_date.notna() \\
          & o.order_estimated_delivery_date.notna() & o.order_purchase_timestamp.notna() \\
          & (o.order_delivered_customer_date >= o.order_purchase_timestamp)
late_nb = (o.order_delivered_customer_date.dt.normalize() > o.order_estimated_delivery_date.dt.normalize())[elig_nb]
late_ts_nb = (o.order_delivered_customer_date > o.order_estimated_delivery_date)[elig_nb]

cmp = pd.DataFrame([
    ("raw orders", len(orders), report["n_orders_raw"], len(pipe_all)),
    ("status = delivered", int((orders.order_status == "delivered").sum()), report["order_status_counts"]["delivered"],
     int((pipe_all.order_status == "delivered").sum())),
    ("eligible (analysis set)", int(elig_nb.sum()), report["n_eligible"], int(pipe_all.analysis_eligible.sum())),
    ("late orders (calendar-date rule)", int(late_nb.sum()), report["n_late_eligible"], int(pipe_all.is_late.fillna(0).sum())),
    ("late rate (calendar-date)", round(float(late_nb.mean()), 6), round(report["late_rate_eligible"], 6),
     round(float(pipe_all.is_late.dropna().mean()), 6)),
    ("late rate (timestamp-based, reference)", round(float(late_ts_nb.mean()), 6),
     round(report["late_rate_timestamp_based_for_reference"], 6), np.nan),
    ("multi-seller orders", int((per_order.n_sellers > 1).sum()), report["multi_seller_orders"],
     int((pipe_all.n_sellers > 1).sum())),
    ("carrier-before-purchase (all orders)", int(anoms["carrier pickup before purchase"].sum()),
     report["logical_order_anomalies_all_orders"]["carrier_before_purchase"], int(pipe_all.carrier_date_invalid.sum())),
], columns=["metric", "notebook (independent)", "pipeline report (JSON)", "pipeline table (re-run now)"])
cmp["agree"] = [
    bool(np.isclose(a, b) and (np.isnan(c) or np.isclose(a, c)))
    for a, b, c in zip(cmp.iloc[:, 1], cmp.iloc[:, 2], cmp.iloc[:, 3].astype(float))
]
fmt = lambda v: "" if pd.isna(v) else (f"{v:,.0f}" if float(v).is_integer() else f"{v:.4f}")
display(cmp.assign(**{c: cmp[c].map(fmt) for c in cmp.columns[1:4]}))
assert cmp["agree"].all(), "notebook and pipeline disagree"

print("Saved parquet matches a fresh pipeline run:",
      len(proc) == int(pipe_all.analysis_eligible.sum()) and set(proc.order_id) == set(pipe_all.loc[pipe_all.analysis_eligible, "order_id"]))
print("Validation problems reported by the pipeline:", report["validation_problems"] or "none")
print("Exclusions:", report["exclusions"])
''')

md("s3_end")

# =============================================================================
# 4. Feature engineering walk-through
# =============================================================================
md("s4_intro")

code('''
d = proc.copy()
# processed categorical columns are pandas 'category' dtype -> use observed=True in groupbys
for c in d.select_dtypes("category").columns:
    d[c] = d[c].cat.remove_unused_categories()
N = len(d)
BASE_LATE = d.is_late.astype(int).mean()
print(f"Analysis set: {N:,} orders | late = {int(d.is_late.sum()):,} ({BASE_LATE:.2%}) | "
      f"purchases {d.order_purchase_timestamp.min():%Y-%m-%d} .. {d.order_purchase_timestamp.max():%Y-%m-%d}")

show_cols = ["order_purchase_timestamp", "order_delivered_carrier_date", "order_delivered_customer_date",
             "order_estimated_delivery_date", "delivery_duration_days", "estimated_delivery_days",
             "delivery_delay_days", "is_late", "processing_days"]

# Example 1: a typical early delivery (the order closest to the median duration among on-time orders)
on_time = d[d.is_late == 0]
ex_ontime = on_time.iloc[(on_time.delivery_duration_days - on_time.delivery_duration_days.median()).abs().argsort()[:1]]
# Example 2: a late order closest to the median delay among late orders
late = d[d.is_late == 1]
ex_late = late.iloc[(late.delivery_delay_days - late.delivery_delay_days.median()).abs().argsort()[:1]]
# Example 3: boundary -- delivered ON the estimated calendar date but after midnight
same_day = d[(d.order_delivered_customer_date.dt.normalize() == d.order_estimated_delivery_date)
             & (d.order_delivered_customer_date > d.order_estimated_delivery_date)]
ex_boundary = same_day.sort_values("order_id").iloc[:1]
print(f"Orders delivered on the estimated calendar date (after 00:00): {len(same_day):,} "
      f"-> all labelled is_late = {sorted(same_day.is_late.unique().tolist())}")

ex = pd.concat([ex_ontime.assign(example="typical on-time"), ex_late.assign(example="typical late"),
                ex_boundary.assign(example="boundary: delivered on the estimate date")])
ex["order_id"] = ex.order_id.str[:8] + "..."
ex.set_index("example")[["order_id"] + show_cols].T
''')

code('''
# Re-derive each engineered column by hand from the raw timestamps and compare with the pipeline output.
r = d[d.order_id.isin(pd.concat([ex_ontime, ex_late, ex_boundary]).order_id)].copy()
manual = pd.DataFrame({
    "duration_days": (r.order_delivered_customer_date - r.order_purchase_timestamp).dt.total_seconds() / 86400,
    "delay_days": (r.order_delivered_customer_date.dt.normalize() - r.order_estimated_delivery_date.dt.normalize()).dt.days,
    "promise_days": (r.order_estimated_delivery_date.dt.normalize() - r.order_purchase_timestamp.dt.normalize()).dt.days,
    "processing_days": (r.order_delivered_carrier_date - r.order_purchase_timestamp).dt.total_seconds() / 86400,
})
manual["is_late (rule: delay > 0)"] = (manual.delay_days > 0).astype(int)
manual["timestamp-based late (deliv > estimate)"] = (r.order_delivered_customer_date > r.order_estimated_delivery_date).astype(int)
manual["pipeline is_late"] = r.is_late.astype(int)
manual.index = r.order_id.str[:8] + "..."
display(manual)
assert np.allclose(manual.duration_days, r.delivery_duration_days.values)
assert (manual.delay_days.values == r.delivery_delay_days.astype(int).values).all()
assert (manual.promise_days.values == r.estimated_delivery_days.astype(int).values).all()
assert np.allclose(manual.processing_days, r.processing_days.values)
assert (manual["is_late (rule: delay > 0)"].values == manual["pipeline is_late"].values).all()
print("Manual derivations match the pipeline columns for the example orders.")
''')

md("s4_boundary")

code('''
# Late-rate definition comparison on the full analysis set
cal_late = d.is_late.astype(int)
ts_late = (d.order_delivered_customer_date > d.order_estimated_delivery_date).astype(int)
cmp_def = pd.DataFrame({
    "definition": ["calendar date: delivery date > estimated date  (used in this project)",
                   "timestamp: delivery time > estimated midnight  (reference only)"],
    "late_orders": [int(cal_late.sum()), int(ts_late.sum())],
    "n": [N, N],
    "late_rate_pct": [100 * cal_late.mean(), 100 * ts_late.mean()],
})
display(cmp_def)
diff = int(((ts_late == 1) & (cal_late == 0)).sum())
print(f"Orders late by timestamp but on time by calendar date: {diff:,} ({diff / N:.2%} of all orders)")
print("...of which delivered on the estimated calendar date itself:",
      int(((ts_late == 1) & (cal_late == 0) & (d.delivery_delay_days == 0)).sum()))
print("Share of eligible orders by delivery_delay_days == 0 :", f"{(d.delivery_delay_days == 0).mean():.2%}")
''')

md("s4_multi")

code('''
# Multi-seller example: pick the first (sorted by order_id) eligible order with 2 sellers in DIFFERENT states
# whose seller states differ from each other -- so the representative-seller rule visibly matters.
ms_ids = d.loc[d.n_sellers >= 2, "order_id"]
it = items[items.order_id.isin(ms_ids)].merge(sellers, on="seller_id").merge(
    products[["product_id", "product_category_name", "product_weight_g"]], on="product_id", how="left")
it = it.merge(translation, on="product_category_name", how="left")
per = it.groupby("order_id").agg(n_states=("seller_state", "nunique"), n_cats=("product_category_name_english", "nunique"),
                                 n_sellers=("seller_id", "nunique"))
cand = per[(per.n_states >= 2) & (per.n_cats >= 2) & (per.n_sellers == 2)].index.min()
ex_ms = it[it.order_id == cand].sort_values("order_item_id")
print(f"Multi-seller order {cand[:8]}... raw item rows (joined with sellers / products / translation):")
display(ex_ms[["order_item_id", "seller_id", "seller_state", "product_category_name_english", "price",
               "freight_value", "product_weight_g"]].assign(seller_id=lambda x: x.seller_id.str[:8] + "..."))

row = d[d.order_id == cand].iloc[0]
seller_sum = ex_ms.groupby(["seller_id", "seller_state"])["price"].sum().sort_values(ascending=False)
print("\\nSummed price per seller (rule: largest total wins; ties -> smallest seller_id):")
print(seller_sum.round(2).to_string())
top_item = ex_ms.sort_values(["price", "order_item_id"], ascending=[False, True]).iloc[0]
print(f"\\nHighest-priced item -> representative category rule gives: {top_item.product_category_name_english}")
print("\\nEngineered order-level row:")
display(row[["n_items", "n_sellers", "n_categories", "seller_state", "customer_state", "same_state", "state_pair",
             "product_category", "total_price", "total_freight", "freight_ratio", "total_weight_g",
             "total_volume_cm3", "estimated_delivery_days", "delivery_duration_days", "is_late"]].to_frame(name="value").T)
assert row.seller_state == seller_sum.index[0][1]
assert row.product_category == top_item.product_category_name_english
assert np.isclose(row.total_price, ex_ms.price.sum()) and np.isclose(row.total_freight, ex_ms.freight_value.sum())
print("Representative seller state / category, total price and freight all re-derived correctly.")
''')

code('''
# Aggregates and geography on the whole analysis set
agg_view = d[["n_items", "n_sellers", "total_price", "total_freight", "freight_ratio", "total_weight_g",
              "total_volume_cm3", "estimated_delivery_days", "processing_days"]].describe(percentiles=[.5, .95]).T
agg_view["missing"] = d[agg_view.index].isna().sum()
display(agg_view)
print(f"same_state = 1 (representative seller and customer in the same state): {d.same_state.mean():.2%} of orders")
print(f"multi-item orders: {(d.n_items > 1).mean():.2%} | multi-seller orders: {(d.n_sellers > 1).mean():.2%}")
''')

# =============================================================================
# 5. EDA
# =============================================================================
md("s5_intro")

code('''
tgt = d.is_late.map({0: "On time / early", 1: "Late"}).value_counts().reindex(["On time / early", "Late"])
fig, ax = plt.subplots(figsize=(6.5, 3.4))
bars = ax.barh(tgt.index[::-1], tgt.values[::-1], color=[P["red"], P["green"]], height=0.55)
for b, v in zip(bars, tgt.values[::-1]):
    ax.text(v + N * 0.01, b.get_y() + b.get_height() / 2, f"{v:,}  ({v / N:.2%})", va="center", color=P["slate"])
ax.set_xlim(0, N * 1.25); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
ax.set_xlabel("Number of delivered orders"); ax.set_title(f"Target: is_late (n = {N:,})")
ax.grid(axis="y", visible=False)
save_show(fig, "04_target_distribution")
print(f"Class imbalance: {tgt['On time / early'] / tgt['Late']:.1f} on-time orders per late order. "
      f"A model that always predicts 'on time' would score {1 - BASE_LATE:.2%} accuracy -> accuracy is a poor metric here.")
''')

md("s5_target_interp")

code('''
dur = d.delivery_duration_days
q = dur.quantile([.5, .9, .95, .99])
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
hi = dur.quantile(.99)
axes[0].hist(dur, bins=60, range=(0, hi), color=P["blue"], alpha=.85, edgecolor="white", linewidth=.3)
axes[0].axvline(dur.mean(), color=P["red"], lw=1.6, ls="--", label=f"mean {dur.mean():.1f} d")
axes[0].axvline(dur.median(), color=P["navy"], lw=1.6, label=f"median {dur.median():.1f} d")
axes[0].set_xlabel(f"Delivery duration, purchase to delivery (days; axis stops at p99 = {hi:.0f} d)")
axes[0].set_ylabel("Orders"); axes[0].set_title("Delivery duration"); axes[0].legend()
axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

delay = d.delivery_delay_days.astype(int)
lo, hi2 = int(delay.quantile(.005)), int(delay.quantile(.995))
axes[1].hist(delay, bins=np.arange(lo - .5, hi2 + 1.5, 1), color=P["blue"], alpha=.85, edgecolor="white", linewidth=.3)
axes[1].axvline(0.5, color=P["red"], lw=1.6, ls="--", label="late if delay >= 1 day")
axes[1].axvline(delay.median(), color=P["navy"], lw=1.6, label=f"median {delay.median():.0f} d")
axes[1].set_xlabel(f"Delay vs estimated date (days; negative = early; axis {lo} to {hi2} d)")
axes[1].set_ylabel("Orders"); axes[1].set_title("Delay relative to the promise"); axes[1].legend()
axes[1].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
save_show(fig, "05_duration_and_delay_distribution")

print(f"Orders beyond the plotted duration range: {(dur > hi).sum():,}; beyond the delay range: {((delay < lo) | (delay > hi2)).sum():,}")
print("Duration quantiles (days):", {f"p{int(k*100)}": round(v, 1) for k, v in q.items()}, "| skew =", round(dur.skew(), 2))
print(f"Delay: median {delay.median():.0f} d, mean {delay.mean():.2f} d | early (<0): {(delay < 0).mean():.1%} | "
      f"on the date (=0): {(delay == 0).mean():.1%} | late (>0): {(delay > 0).mean():.1%}")
L = d[d.is_late == 1].delivery_delay_days.astype(int)
print(f"Among the {len(L):,} late orders: median delay {L.median():.0f} d, mean {L.mean():.1f} d, p90 {L.quantile(.9):.0f} d, max {L.max()} d")
print(f"Estimated promise (checkout): mean {d.estimated_delivery_days.mean():.1f} d vs actual mean duration {dur.mean():.1f} d")
''')

md("s5_dist_interp")

code('''
mon = d.groupby("purchase_year_month", observed=True).agg(n=("is_late", "size"), late=("is_late", "sum"),
                                                        mean_dur=("delivery_duration_days", "mean")).reset_index()
mon["late_rate"] = mon.late / mon.n
mon["lo"], mon["hi"] = wilson(mon.late, mon.n)
mon["reliable"] = mon.n >= MIN_N
# Plot on a uniform calendar axis: months with no eligible orders stay as empty slots
full = pd.period_range(mon.purchase_year_month.min(), mon.purchase_year_month.max(), freq="M").astype(str)
pm = mon.assign(purchase_year_month=mon.purchase_year_month.astype(str)).set_index("purchase_year_month").reindex(full)
pm["n"] = pm.n.fillna(0)
pm["reliable"] = pm.reliable.fillna(False).astype(bool)
pm.index.name = "purchase_year_month"
x = np.arange(len(pm))
fig, (a1, a2) = plt.subplots(2, 1, figsize=(11.5, 6.4), sharex=True, gridspec_kw={"height_ratios": [1, 1.25]})
a1.bar(x, pm.n, color=np.where(pm.reliable, P["blue"], P["blue_light"]), width=.75)
a1.set_ylabel("Delivered orders"); a1.set_title("Monthly order volume (light bars: fewer than %d orders)" % MIN_N)
a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
ok = pm.reliable.values
a2.fill_between(x[ok], (pm.lo * 100)[ok], (pm.hi * 100)[ok], color=P["red"], alpha=.15, lw=0, label="95% Wilson interval")
a2.plot(x[ok], (pm.late_rate * 100)[ok], color=P["red"], lw=2, marker="o", ms=4, label="late rate")
a2.plot(x[~ok], (pm.late_rate * 100)[~ok], ls="none", marker="o", ms=5, mfc="white", mec=P["red"], label=f"n < {MIN_N} (unreliable)")
a2.axhline(BASE_LATE * 100, color=P["slate_light"], ls=":", lw=1.2, label=f"overall {BASE_LATE:.2%}")
top = pm[pm.reliable].nlargest(3, "late_rate")
for lab_, rr in top.iterrows():
    a2.annotate(f"{lab_}: {rr.late_rate:.1%} (n={int(rr.n):,})", (full.get_loc(lab_), rr.late_rate * 100),
                xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8, color=P["navy"])
a2.set_ylabel("Late rate (%)"); a2.set_title("Monthly late rate by purchase month")
a2.set_xticks(x[::2]); a2.set_xticklabels(full[::2], rotation=60, ha="right")
a2.set_xlabel("Purchase month"); a2.legend(ncol=4, loc="upper left", fontsize=8)
a2.set_ylim(0, max(pm.hi[ok].max() * 100 * 1.15, 1))
a2.text(0.995, 0.02, "points with n < %d may fall outside the axis; see the table below" % MIN_N, transform=a2.transAxes, ha="right", fontsize=7.5, color=P["slate_light"])
save_show(fig, "06_monthly_volume_late_rate")

mt = mon.assign(late_rate_pct=100 * mon.late_rate)[["purchase_year_month", "n", "late", "late_rate_pct", "mean_dur"]]
display(mt.set_index("purchase_year_month").T.round(2))
print("Months with n < %d:" % MIN_N, mon.loc[~mon.reliable, "purchase_year_month"].tolist())
print("2016 eligible orders:", int((d.purchase_year == 2016).sum()))
print("First month with n >= %d:" % MIN_N, mon.loc[mon.reliable, "purchase_year_month"].iloc[0],
      "| first month with n >= 500:", mon.loc[mon.n >= 500, "purchase_year_month"].iloc[0])
print("Lowest 3 months by late rate (n >= %d):" % MIN_N)
display(mon[mon.reliable].nsmallest(3, "late_rate")[["purchase_year_month", "n", "late", "late_rate"]].assign(
    late_rate=lambda t: (100 * t.late_rate).round(2)).reset_index(drop=True))
print("Top 5 months by late rate (n >= %d):" % MIN_N)
display(mon[mon.reliable].nlargest(5, "late_rate")[["purchase_year_month", "n", "late", "late_rate"]].assign(
    late_rate=lambda t: (100 * t.late_rate).round(2)).reset_index(drop=True))
''')

code('''
# Zoom into the two spike periods: daily view around Nov-2017, weekly view Jan-Apr 2018.
dd = d.assign(day=d.order_purchase_timestamp.dt.normalize())
w1 = dd[(dd.day >= "2017-11-01") & (dd.day <= "2017-12-10")].groupby("day").agg(n=("is_late", "size"), late=("is_late", "sum"))
w1["rate"] = w1.late / w1.n
wk = dd[(dd.day >= "2018-01-01") & (dd.day <= "2018-04-29")].copy()
wk["week"] = wk.day.dt.to_period("W").dt.start_time
w2 = wk.groupby("week").agg(n=("is_late", "size"), late=("is_late", "sum"), promise=("estimated_delivery_days", "mean"),
                            dur=("delivery_duration_days", "mean"))
w2["rate"] = w2.late / w2.n

fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.4), sharex="col")
axes[0, 0].bar(w1.index, w1.n, color=P["blue"], width=.8); axes[0, 0].set_ylabel("Orders per day")
axes[0, 0].set_title("Nov 1 - Dec 10, 2017: daily volume")
axes[1, 0].plot(w1.index, w1.rate * 100, color=P["red"], lw=2, marker="o", ms=3.5)
axes[1, 0].axhline(BASE_LATE * 100, color=P["slate_light"], ls=":"); axes[1, 0].set_ylabel("Late rate (%)")
axes[1, 0].set_title("Late rate by purchase day"); axes[1, 0].set_xlabel("Purchase date")
axes[0, 1].bar(w2.index, w2.n, color=P["blue"], width=5); axes[0, 1].set_ylabel("Orders per week")
axes[0, 1].set_title("Jan - Apr 2018: weekly volume")
axes[1, 1].plot(w2.index, w2.rate * 100, color=P["red"], lw=2, marker="o", ms=4)
axes[1, 1].axhline(BASE_LATE * 100, color=P["slate_light"], ls=":", label=f"overall {BASE_LATE:.2%}")
axes[1, 1].set_ylabel("Late rate (%)"); axes[1, 1].set_title("Late rate by purchase week (week start date)")
axes[1, 1].set_xlabel("Week starting"); axes[1, 1].legend(loc="upper right")
for a in axes.ravel():
    a.tick_params(axis="x", rotation=45)
save_show(fig, "06b_spike_periods_zoom")

peak = w1.n.idxmax()
print(f"Busiest day in the window: {peak:%Y-%m-%d} with {int(w1.n.max()):,} orders "
      f"(median of other days in window: {w1.n.drop(peak).median():.0f}); late rate that day {w1.rate[peak]:.1%}")
b4, af = w1[:"2017-11-19"], w1["2017-11-21":"2017-11-30"]
print(f"Nov-2017 late rate for purchases up to Nov 19: {b4.late.sum() / b4.n.sum():.1%} (n={int(b4.n.sum()):,}) | "
      f"Nov 21-30: {af.late.sum() / af.n.sum():.1%} (n={int(af.n.sum()):,})")
pk2 = w2.rate.idxmax()
print(f"Worst week Jan-Apr 2018: starting {pk2:%Y-%m-%d}, late rate {w2.rate[pk2]:.1%} (n={int(w2.n[pk2]):,}); "
      f"mean checkout promise that week {w2.promise[pk2]:.1f} d vs {d.estimated_delivery_days.mean():.1f} d overall")

# Selection check: what share of ALL purchased orders (any status) is in the analysis set, by purchase month?
pa = pipe_all.assign(ym=pipe_all.order_purchase_timestamp.dt.strftime("%Y-%m"))
share = pa.groupby("ym").agg(orders=("order_id", "size"), eligible=("analysis_eligible", "sum"))
share["eligible_pct"] = 100 * share.eligible / share.orders
print("\\nShare of all purchased orders that are in the analysis set, last 9 purchase months:")
display(share.tail(9).round(2).T)
''')

md("s5_monthly_interp")

code('''
st = d.groupby("customer_state", observed=True).agg(n=("is_late", "size"), late=("is_late", "sum"),
                                                   mean_dur=("delivery_duration_days", "mean"),
                                                   median_dur=("delivery_duration_days", "median"),
                                                   mean_promise=("estimated_delivery_days", "mean")).reset_index()
st["late_rate"] = st.late / st.n
st["lo"], st["hi"] = wilson(st.late, st.n)
st["reliable"] = st.n >= MIN_N
st = st.sort_values("late_rate", ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(9.5, 0.27 * len(st) + 1.6))
y = np.arange(len(st))[::-1]
cols = np.where(st.reliable, P["blue"], P["blue_light"])
ax.barh(y, st.late_rate * 100, color=cols, height=.65)
ax.errorbar(st.late_rate * 100, y, xerr=[(st.late_rate - st.lo) * 100, (st.hi - st.late_rate) * 100],
            fmt="none", ecolor=P["slate"], elinewidth=.8, capsize=2)
for yy, (_, r_) in zip(y, st.iterrows()):
    ax.text(0.995, yy, f"n={int(r_.n):,}", va="center", ha="right", fontsize=7.5, color=P["slate"],
            transform=ax.get_yaxis_transform())
ax.set_yticks(y); ax.set_yticklabels(st.customer_state)
ax.axvline(BASE_LATE * 100, color=P["red"], ls="--", lw=1.2, label=f"overall {BASE_LATE:.2%}")
ax.set_xlabel("Late rate (%) with 95% Wilson interval"); ax.set_ylabel("Customer state")
ax.set_title(f"Late rate by customer state (light bars: n < {MIN_N}, unreliable)")
ax.legend(loc="lower right"); ax.grid(axis="y", visible=False); ax.set_xlim(0, st.hi.max() * 100 * 1.12)
save_show(fig, "07_late_rate_by_state")

disp = st.assign(late_rate_pct=100 * st.late_rate, ci_low_pct=100 * st.lo, ci_high_pct=100 * st.hi)[
    ["customer_state", "n", "late", "late_rate_pct", "ci_low_pct", "ci_high_pct", "mean_dur", "median_dur", "mean_promise", "reliable"]]
print("Highest and lowest late rates among states with n >= %d:" % MIN_N)
rel = disp[disp.reliable]
display(pd.concat([rel.head(5), rel.tail(5)]).reset_index(drop=True))
print("The five largest states by number of orders:")
display(disp.nlargest(5, "n").reset_index(drop=True))
print("States below the sample threshold:", disp.loc[~disp.reliable, ["customer_state", "n"]].values.tolist())
print(f"Spread of late rate across reliable states: {rel.late_rate_pct.min():.2f}% .. {rel.late_rate_pct.max():.2f}% "
      f"({rel.late_rate_pct.max() / rel.late_rate_pct.min():.1f}x)")
chi = stats.chi2_contingency(pd.crosstab(d.customer_state, d.is_late))
n_ct = pd.crosstab(d.customer_state, d.is_late).values.sum()
cv = np.sqrt(chi.statistic / (n_ct * 1))
print(f"Chi-square, customer_state vs is_late: chi2 = {chi.statistic:.1f}, dof = {chi.dof}, p {'< 1e-300' if chi.pvalue == 0 else '= %.2e' % chi.pvalue}, Cramer's V = {cv:.3f}")
''')

md("s5_state_interp")

code('''
cat = d.groupby("product_category", observed=True).agg(n=("is_late", "size"), late=("is_late", "sum"),
                                                      mean_dur=("delivery_duration_days", "mean")).reset_index()
cat["late_rate"] = cat.late / cat.n
cat["lo"], cat["hi"] = wilson(cat.late, cat.n)
cr = cat[cat.n >= MIN_N].sort_values("late_rate", ascending=False)
print(f"{len(cat)} categories in the analysis set ({len(cr)} with n >= {MIN_N}; {(cat.n < MIN_N).sum()} below the threshold and excluded from ranking)")
k = 10
sel = pd.concat([cr.head(k), cr.tail(k)]).drop_duplicates("product_category")
sel = sel.sort_values("late_rate")
fig, ax = plt.subplots(figsize=(9.5, 0.3 * len(sel) + 1.6))
y = np.arange(len(sel))
ax.barh(y, sel.late_rate * 100, color=[P["red"] if v > BASE_LATE else P["green"] for v in sel.late_rate], height=.65)
ax.errorbar(sel.late_rate * 100, y, xerr=[(sel.late_rate - sel.lo) * 100, (sel.hi - sel.late_rate) * 100],
            fmt="none", ecolor=P["slate"], elinewidth=.8, capsize=2)
for yy, (_, r_) in zip(y, sel.iterrows()):
    ax.text(0.995, yy, f"n={int(r_.n):,}", va="center", ha="right", fontsize=7.5, color=P["slate"],
            transform=ax.get_yaxis_transform())
ax.set_yticks(y); ax.set_yticklabels(sel.product_category)
ax.axvline(BASE_LATE * 100, color=P["navy"], ls="--", lw=1.2, label=f"overall {BASE_LATE:.2%}")
ax.set_xlabel("Late rate (%) with 95% Wilson interval"); ax.set_ylabel("Representative product category")
ax.set_title(f"Late rate: {k} highest and {k} lowest categories (n >= {MIN_N})")
ax.legend(loc="lower right"); ax.grid(axis="y", visible=False); ax.set_xlim(0, sel.hi.max() * 100 * 1.12)
save_show(fig, "08_late_rate_by_category")
display(pd.concat([cr.head(5), cr.tail(5)]).assign(late_rate_pct=lambda t: 100 * t.late_rate,
        ci_low_pct=lambda t: 100 * t.lo, ci_high_pct=lambda t: 100 * t.hi)[
        ["product_category", "n", "late", "late_rate_pct", "ci_low_pct", "ci_high_pct", "mean_dur"]].reset_index(drop=True))
print("Ten largest categories by n:", cat.nlargest(10, "n")[["product_category", "n"]].assign(
    late_pct=lambda t: (100 * cat.loc[t.index, "late_rate"]).round(2)).values.tolist())
unk = cat[cat.product_category == "unknown"]
print("'unknown' category (missing in source):", unk[["n", "late"]].assign(late_rate_pct=100 * unk.late / unk.n).round(2).to_dict("records"))
sig_hi = cr[cr.lo > BASE_LATE]; sig_lo = cr[cr.hi < BASE_LATE]
print(f"Categories with n >= {MIN_N}: {len(cr)}; 95% interval overlaps the overall {BASE_LATE:.2%}: {len(cr) - len(sig_hi) - len(sig_lo)}; "
      f"entirely above: {len(sig_hi)}; entirely below: {len(sig_lo)} (about {0.05 * len(cr):.1f} exclusions expected by chance alone)")
print("Interval excludes overall rate:", [f"{r_.product_category} ({r_.late_rate:.2%}, n={int(r_.n):,})" for r_ in pd.concat([sig_hi, sig_lo]).itertuples()])
chi_c = stats.chi2_contingency(pd.crosstab(d.loc[d.product_category.isin(cr.product_category), "product_category"],
                                           d.loc[d.product_category.isin(cr.product_category), "is_late"]))
print(f"Chi-square (categories with n >= {MIN_N}) vs is_late: chi2 = {chi_c.statistic:.1f}, dof = {chi_c.dof}, p = {chi_c.pvalue:.2e}, "
      f"Cramer's V = {np.sqrt(chi_c.statistic / cr.n.sum()):.3f}")
''')

md("s5_cat_interp")

code('''
def binned(df: pd.DataFrame, col: str, q: int = 10) -> pd.DataFrame:
    """Quantile bins of ``col`` with n, mean/median duration and late rate per bin (NaNs dropped)."""
    s = df[[col, "delivery_duration_days", "is_late"]].dropna()
    b = pd.qcut(s[col], q=q, duplicates="drop")
    g = s.groupby(b, observed=True).agg(n=("is_late", "size"), late=("is_late", "sum"),
                                        mean_dur=("delivery_duration_days", "mean"),
                                        median_dur=("delivery_duration_days", "median"),
                                        lo_edge=(col, "min"), hi_edge=(col, "max")).reset_index(drop=True)
    g["late_rate"] = g.late / g.n
    return g

fig, axes = plt.subplots(2, 2, figsize=(11.5, 7))
for j, (col, lab, fmt) in enumerate([("total_weight_g", "Order weight (g), decile bins", "{:,.0f}"),
                                      ("total_freight", "Freight value (BRL), decile bins", "{:,.1f}")]):
    g = binned(d, col)
    xs = np.arange(len(g))
    labels = [fmt.format(v) for v in g.hi_edge]
    axes[0, j].plot(xs, g.mean_dur, marker="o", color=P["blue"], lw=2, label="mean")
    axes[0, j].plot(xs, g.median_dur, marker="s", color=P["navy"], lw=1.5, ls="--", label="median")
    axes[0, j].set_ylabel("Delivery duration (days)"); axes[0, j].set_title(f"Duration vs {col}")
    axes[0, j].legend(); axes[0, j].set_xticks(xs); axes[0, j].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1, j].bar(xs, g.late_rate * 100, color=P["red"], alpha=.8, width=.7)
    axes[1, j].axhline(BASE_LATE * 100, color=P["slate_light"], ls=":", label=f"overall {BASE_LATE:.2%}")
    for xx, (lr, nn) in enumerate(zip(g.late_rate, g.n)):
        axes[1, j].text(xx, lr * 100 + .1, f"n={nn:,}", ha="center", fontsize=6.5, rotation=90, va="bottom", color=P["slate"])
    axes[1, j].set_ylabel("Late rate (%)"); axes[1, j].set_xlabel(f"{lab} (label = upper bin edge)")
    axes[1, j].set_ylim(0, g.late_rate.max() * 100 * 1.35); axes[1, j].legend(loc="upper left")
    axes[1, j].set_xticks(xs); axes[1, j].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1, j].set_title(f"Late rate vs {col}")
    display(g.assign(late_rate_pct=100 * g.late_rate)[["lo_edge", "hi_edge", "n", "mean_dur", "median_dur", "late_rate_pct"]]
            .rename(columns={"lo_edge": f"{col} from", "hi_edge": "to"}).round(2))
    print(f"{col}: n with value = {d[col].notna().sum():,}  | Spearman rho vs duration = "
          f"{stats.spearmanr(d[col], d.delivery_duration_days, nan_policy='omit').statistic:.3f}, "
          f"vs is_late = {stats.spearmanr(d[col], d.is_late.astype(float), nan_policy='omit').statistic:.3f}")
save_show(fig, "09_weight_freight_vs_duration")
''')

md("s5_wf_interp")

code('''
pr = d.processing_days.dropna()
print(f"processing_days (RETROSPECTIVE: carrier pickup - purchase) available for {len(pr):,} of {N:,} orders "
      f"({N - len(pr):,} NaN: carrier date missing/invalid)")
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
hi = pr.quantile(.99)
axes[0].hist(pr, bins=60, range=(0, hi), color=P["blue"], alpha=.85, edgecolor="white", linewidth=.3)
axes[0].axvline(pr.mean(), color=P["red"], ls="--", lw=1.6, label=f"mean {pr.mean():.1f} d")
axes[0].axvline(pr.median(), color=P["navy"], lw=1.6, label=f"median {pr.median():.1f} d")
axes[0].set_xlabel(f"Processing time to carrier pickup (days; axis stops at p99 = {hi:.1f} d)")
axes[0].set_ylabel("Orders"); axes[0].set_title("Seller processing time (retrospective)"); axes[0].legend()
axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

bk = pd.cut(d.processing_days, [0, 1, 2, 3, 5, 7, 10, np.inf], right=False,
            labels=["<1 d", "1-2 d", "2-3 d", "3-5 d", "5-7 d", "7-10 d", ">=10 d"])
pb = d.groupby(bk, observed=True).agg(n=("is_late", "size"), late=("is_late", "sum")).reset_index().rename(columns={"processing_days": "bucket"})
pb["late_rate"] = pb.late / pb.n
pb["lo"], pb["hi"] = wilson(pb.late, pb.n)
axes[1].bar(pb.bucket.astype(str), pb.late_rate * 100, color=P["red"], alpha=.8, width=.65)
axes[1].errorbar(pb.bucket.astype(str), pb.late_rate * 100, yerr=[(pb.late_rate - pb.lo) * 100, (pb.hi - pb.late_rate) * 100],
                 fmt="none", ecolor=P["slate"], capsize=3)
for xx, (lr, h_, nn) in enumerate(zip(pb.late_rate, pb.hi, pb.n)):
    axes[1].text(xx, h_ * 100 + .5, f"n={nn:,}", ha="center", fontsize=8, color=P["slate"])
axes[1].axhline(BASE_LATE * 100, color=P["slate_light"], ls=":", label=f"overall {BASE_LATE:.2%}")
axes[1].set_xlabel("Processing-time bucket (RETROSPECTIVE, not known at purchase)"); axes[1].set_ylabel("Late rate (%)")
axes[1].set_title("Late rate by processing-time bucket"); axes[1].legend(loc="upper left")
axes[1].set_ylim(0, pb.hi.max() * 100 * 1.2)
save_show(fig, "10_processing_time")
display(pb.assign(late_rate_pct=(100 * pb.late_rate).round(2))[["bucket", "n", "late", "late_rate_pct"]])
print("Spearman rho(processing_days, is_late) =", round(stats.spearmanr(d.processing_days, d.is_late.astype(float), nan_policy="omit").statistic, 3))
cc = d[["processing_days", "carrier_to_customer_days", "delivery_duration_days"]].dropna()
print("Share of total duration spent after carrier pickup (mean of carrier_to_customer / duration):",
      f"{(cc.carrier_to_customer_days / cc.delivery_duration_days).mean():.1%}")
''')

md("s5_proc_interp")

code('''
ss = d.groupby("same_state").agg(n=("is_late", "size"), late=("is_late", "sum"), mean_dur=("delivery_duration_days", "mean"),
                                 median_dur=("delivery_duration_days", "median"),
                                 mean_promise=("estimated_delivery_days", "mean")).reset_index()
ss["late_rate"] = ss.late / ss.n
ss["lo"], ss["hi"] = wilson(ss.late, ss.n)
ss["group"] = ss.same_state.map({1: "Same state", 0: "Cross-state"})
ss = ss.sort_values("same_state")
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
axes[0].bar(ss.group, ss.late_rate * 100, color=[P["slate_light"], P["blue"]], width=.5)
axes[0].errorbar(ss.group, ss.late_rate * 100, yerr=[(ss.late_rate - ss.lo) * 100, (ss.hi - ss.late_rate) * 100],
                 fmt="none", ecolor=P["navy"], capsize=4)
for xx, r_ in enumerate(ss.itertuples()):
    axes[0].text(xx, r_.hi * 100 + .3, f"{r_.late_rate:.2%}\\nn={r_.n:,}", ha="center", fontsize=9)
axes[0].set_ylabel("Late rate (%)"); axes[0].set_title("Late rate: seller and customer state"); axes[0].set_ylim(0, ss.hi.max() * 100 * 1.4)
xx = np.arange(len(ss)); wdt = .26
for off, col, lab, c_ in [(-wdt, "mean_dur", "mean duration", P["blue"]), (0, "median_dur", "median duration", P["navy"]),
                          (wdt, "mean_promise", "mean checkout promise", P["amber"])]:
    axes[1].bar(xx + off, ss[col], width=wdt - .02, color=c_, label=lab)
    for x_, v_ in zip(xx + off, ss[col]):
        axes[1].text(x_, v_ + .4, f"{v_:.1f}", ha="center", fontsize=8)
axes[1].set_xticks(xx); axes[1].set_xticklabels(ss.group)
axes[1].set_ylabel("Days"); axes[1].set_title("Delivery duration vs checkout promise"); axes[1].legend(loc="upper right")
axes[1].set_ylim(0, ss.mean_promise.max() * 1.45)
save_show(fig, "11_same_vs_cross_state")
display(ss.assign(late_rate_pct=100 * ss.late_rate, ci_low_pct=100 * ss.lo, ci_high_pct=100 * ss.hi)[
    ["group", "n", "late", "late_rate_pct", "ci_low_pct", "ci_high_pct", "mean_dur", "median_dur", "mean_promise"]].round(2))
ct = pd.crosstab(d.same_state, d.is_late)
c2 = stats.chi2_contingency(ct)
print(f"Chi-square same_state vs is_late: chi2 = {c2.statistic:.1f}, p = {c2.pvalue:.2e}, Cramer's V = {np.sqrt(c2.statistic / ct.values.sum()):.3f}")

pairs = d.groupby("state_pair", observed=True).agg(n=("is_late", "size"), late=("is_late", "sum")).reset_index()
pairs["late_rate_pct"] = 100 * pairs.late / pairs.n
print(f"\\nMost common seller->customer state pairs (top 6 by n) and the highest late rates among pairs with n >= {MIN_N}:")
display(pairs.nlargest(6, "n").reset_index(drop=True))
display(pairs[pairs.n >= MIN_N].nlargest(5, "late_rate_pct").reset_index(drop=True))
''')

md("s5_ss_interp")

code('''
num_cols = ["delivery_duration_days", "delivery_delay_days", "estimated_delivery_days", "total_price", "total_freight",
            "freight_ratio", "total_weight_g", "total_volume_cm3", "n_items", "processing_days", "review_score"]
summ = d[num_cols].astype(float).describe(percentiles=[.05, .25, .5, .75, .95]).T
summ["missing"] = d[num_cols].isna().sum().values
summ["skew"] = d[num_cols].astype(float).skew()
display(summ)

# Spearman association of purchase-time numeric features with outcomes (descriptive)
feat = [c for c in config.PURCHASE_TIME_NUMERIC if c not in ("purchase_quarter",)]
rows = []
for c in feat:
    s = d[[c, "delivery_duration_days", "is_late"]].dropna().astype(float)
    rows.append({"feature": c, "n": len(s),
                 "rho_vs_duration": stats.spearmanr(s[c], s.delivery_duration_days).statistic,
                 "rho_vs_is_late": stats.spearmanr(s[c], s.is_late).statistic})
sp = pd.DataFrame(rows).sort_values("rho_vs_is_late", key=abs, ascending=False).reset_index(drop=True)
display(sp)
fig, ax = plt.subplots(figsize=(8.5, 0.34 * len(sp) + 1.4))
y = np.arange(len(sp))[::-1]
ax.barh(y + .18, sp.rho_vs_duration, height=.34, color=P["blue"], label="vs delivery duration")
ax.barh(y - .18, sp.rho_vs_is_late, height=.34, color=P["red"], label="vs is_late")
ax.set_yticks(y); ax.set_yticklabels(sp.feature); ax.axvline(0, color=P["slate_light"], lw=.8)
ax.set_xlabel("Spearman rank correlation (descriptive association, not causal)")
ax.set_title("Purchase-time numeric features: rank correlation with outcomes"); ax.legend(loc="lower right")
ax.grid(axis="y", visible=False)
save_show(fig, "12_spearman_purchase_time_features")
''')

md("s5_end")

# =============================================================================
# 6. Predictive vs descriptive feature sets
# =============================================================================
md("s6_intro")

code('''
reason = {
    "order_id": "identifier", "order_purchase_timestamp": "raw timestamp; used for the chronological split",
    "order_estimated_delivery_date": "raw timestamp; its information enters via estimated_delivery_days",
    "purchase_date": "date proxy -> time-drift shortcut", "purchase_year": "time-drift proxy",
    "purchase_year_month": "time-drift proxy",
    "order_status": "final status is only known after the order ends", "order_approved_at": "post-purchase event",
    "order_delivered_carrier_date": "post-purchase event", "order_delivered_customer_date": "defines the outcome",
    "delivery_duration_days": "outcome (derived from delivery date)", "delivery_delay_days": "the target before thresholding",
    "processing_days": "known only once the seller ships", "approval_days": "known only once payment is approved",
    "carrier_to_customer_days": "known only after delivery", "is_late": "the target itself",
    "review_score": "written after delivery", "analysis_eligible": "depends on the outcome being observed",
    "exclusion_reason": "depends on the outcome being observed",
    "carrier_date_invalid": "flag from a post-purchase timestamp", "approval_missing": "flag from a post-purchase timestamp",
}
def group_of(c: str) -> str:
    if c in config.PURCHASE_TIME_FEATURES: return "purchase-time feature (allowed)"
    if c in config.LEAKAGE_COLUMNS: return "LEAKAGE / outcome (never a model input)"
    if c in config.PURCHASE_TIME_UNUSED: return "purchase-time, not used by the model"
    if c in config.NON_FEATURE_COLUMNS: return "non-feature (key / split / drift proxy)"
    return "UNCLASSIFIED"

feature_table = pd.DataFrame({"column": list(proc.columns)})
feature_table["group"] = feature_table.column.map(group_of)
feature_table["note"] = feature_table.column.map(lambda c: reason.get(c, "known when the order is placed" if group_of(c).startswith("purchase") else ""))
assert "UNCLASSIFIED" not in set(feature_table.group), "a processed column is not classified in config.py"
assert not set(config.PURCHASE_TIME_FEATURES) & set(config.LEAKAGE_COLUMNS)
display(feature_table.groupby("group").column.agg(["count", lambda s: ", ".join(s)]).rename(
    columns={"count": "n_columns", "<lambda_0>": "columns"}))
print("Every processed column falls in exactly one group (checked).")
''')

code('''
# Single-feature ROC-AUC against is_late: a descriptive way to *see* leakage. NOT model performance.
y = d.is_late.astype(int)
def auc1(col: str) -> float:
    s = d[col].astype(float)
    m = s.notna()
    return roc_auc_score(y[m], s[m])

cand = ["delivery_delay_days", "delivery_duration_days", "carrier_to_customer_days", "processing_days", "review_score",
        "estimated_delivery_days", "total_freight", "freight_ratio", "total_weight_g", "same_state", "n_sellers", "total_price"]
leak = pd.DataFrame({"column": cand, "group": [group_of(c) for c in cand],
                     "n_used": [int(d[c].notna().sum()) for c in cand],
                     "AUC_as_score": [auc1(c) for c in cand]})
leak["strength |AUC-0.5|"] = (leak.AUC_as_score - .5).abs()
display(leak.sort_values("strength |AUC-0.5|", ascending=False).reset_index(drop=True))
''')

md("s6_interp")

# =============================================================================
# 7. Takeaways
# =============================================================================
code('''
vals = {
    "analysis_set_orders": N, "late_orders": int(d.is_late.sum()), "late_rate": BASE_LATE,
    "late_rate_timestamp_basis": float(ts_late.mean()),
    "mean_duration_d": float(d.delivery_duration_days.mean()), "median_duration_d": float(d.delivery_duration_days.median()),
    "p95_duration_d": float(d.delivery_duration_days.quantile(.95)),
    "median_delay_d": float(d.delivery_delay_days.astype(int).median()),
    "mean_promise_d": float(d.estimated_delivery_days.mean()),
    "same_state_share": float(d.same_state.mean()),
    "late_rate_same_state": float(d.loc[d.same_state == 1, "is_late"].astype(int).mean()),
    "late_rate_cross_state": float(d.loc[d.same_state == 0, "is_late"].astype(int).mean()),
    "state_late_rate_min_reliable": float(rel.late_rate_pct.min()), "state_late_rate_max_reliable": float(rel.late_rate_pct.max()),
    "highest_state": rel.iloc[0].customer_state, "lowest_state": rel.iloc[-1].customer_state,
    "highest_month": mon[mon.reliable].nlargest(1, "late_rate").purchase_year_month.iloc[0],
    "highest_month_rate": float(mon[mon.reliable].late_rate.max()),
}
pd.Series(vals).to_frame("value")
''')

md("s7")

# =============================================================================
# Build + execute
# =============================================================================
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}

def check_narrative(nb_: "nbf.NotebookNode") -> None:
    """Recompute the figures quoted in the markdown straight from the parquet; fail if the text drifts."""
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(HERE.parent))
    from src import config as cfg
    df = pd.read_parquet(cfg.PROCESSED_DIR / "orders_delivered.parquet")
    late = df.is_late.astype(int)
    text = "\n".join(c.source for c in nb_.cells if c.cell_type == "markdown")
    pct = lambda k, n: f"{100 * k / n:.2f}%"  # noqa: E731  # one rounding only, from exact counts
    need: dict[str, str] = {}
    ss_ = df.groupby(df.same_state.astype(int)).is_late.agg(lambda v: (v.astype(int).sum(), len(v)))
    need["same-state late rate"] = pct(*ss_[1])
    need["cross-state late rate"] = pct(*ss_[0])
    pdays = df.processing_days
    m710 = (pdays >= 7) & (pdays < 10)
    need["7-10 day processing late rate"] = pct(late[m710].sum(), m710.sum())
    mon_ = df.groupby("purchase_year_month", observed=True).is_late.agg(n="size", k=lambda v: int(v.astype(int).sum())).query("n >= @cfg.DEFAULT_MIN_ORDERS")
    mon_.index = mon_.index.astype(str)
    mon_ = mon_.assign(r=mon_.k / mon_.n)
    for label, sub in (("lowest month with n >= 100", mon_), ("lowest month since 2017-01", mon_[mon_.index >= "2017-01"])):
        low = sub.loc[sub.r.idxmin()]
        need[label] = pct(low.k, low.n)
    need["first month with n >= 100"] = mon_.index[0]
    cat_ = df.groupby("product_category", observed=True).is_late.agg(n="size", k=lambda v: int(v.astype(int).sum()))
    cat_ = cat_[cat_.n >= cfg.DEFAULT_MIN_ORDERS]
    z = 1.96
    p_ = cat_.k / cat_.n
    den = 1 + z**2 / cat_.n
    ctr = (p_ + z**2 / (2 * cat_.n)) / den
    hw = z * np.sqrt(p_ * (1 - p_) / cat_.n + z**2 / (4 * cat_.n**2)) / den
    base = late.mean()
    n_out = int(((ctr - hw > base) | (ctr + hw < base)).sum())
    need["categories overlapping overall rate"] = f"{len(cat_) - n_out} of {len(cat_)} categories"
    need["categories excluding overall rate"] = f"{n_out} differ detectably"
    bad = [f"{k}: expected '{v}' in markdown" for k, v in need.items() if v not in text]
    stale = [t for t in ("4.51%", "12.55%", "1.17%", "largely noise") if t in text]
    assert not bad and not stale, f"narrative drift: {bad} stale: {stale}"
    print("narrative check OK:", need)


if __name__ == "__main__":
    import time
    t0 = time.time()
    client = NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(HERE)}})
    client.execute()
    check_narrative(nb)
    nbf.write(nb, NB_PATH)
    print(f"executed and wrote {NB_PATH} in {time.time() - t0:.1f}s")
