"""Regenerate ``docs/FINDINGS.md`` from the processed data, the model metrics and the ``src`` analytics.

Every number in the generated document is computed here (or read from ``models/model_metrics.json`` /
``data/processed/data_quality_report.json``); nothing is typed by hand.  Segment statistics reuse the functions the app
uses (``src.analytics``, ``src.insights``), so the document and the dashboard cannot disagree.

Usage (from the repository root)::

    .venv/bin/python scripts/generate_findings.py                 # writes docs/FINDINGS.md
    .venv/bin/python scripts/generate_findings.py --out /tmp/f.md  # write elsewhere (e.g. to diff)

Requires ``data/processed/orders_delivered.parquet`` (``python -m src.preprocessing``) and, for the model section,
``models/model_metrics.json`` (``python -m src.train_model``).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import analytics as an  # noqa: E402
from src import config  # noqa: E402
from src import insights as ins  # noqa: E402

MIN_ORDERS = config.DEFAULT_MIN_ORDERS
ALPHA = config.DEFAULT_ALPHA
CHECKS: list[str] = []          # cross-checks between independent sources; written into the document


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #
def pct(x: float, d: int = 1) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{100 * x:.{d}f}%"


def pp(x: float, d: int = 1) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{100 * x:+.{d}f} pp"


def num(x: float, d: int = 1) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:,.{d}f}"


def i(x) -> str:
    return f"{int(x):,}"


def ci(lo: float, hi: float) -> str:
    return f"{pct(lo)} to {pct(hi)}"


def state(code: str) -> str:
    return f"{code} ({config.STATE_NAMES.get(code, code)})"


def cat(c: str) -> str:
    return str(c).replace("_", " ")


def table(headers: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = " | ".join({"l": ":--", "r": "--:"}[a] for a in align)
    out = ["| " + " | ".join(headers) + " |", "| " + sep + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def check(label: str, a, b, tol: float = 1e-9) -> None:
    ok = (a == b) if isinstance(a, (int, str)) else abs(a - b) <= tol
    CHECKS.append(f"{'OK  ' if ok else 'FAIL'} {label}: computed {a!r} vs recorded {b!r}")


def perf_rows(t: pd.DataFrame, key: str, label_fn, flag: dict[str, str] | None = None, extra: bool = False) -> list[list[str]]:
    rows = []
    for _, r in t.iterrows():
        row = [label_fn(r[key]), i(r["n_orders"]), pct(r["late_rate"]), ci(r["late_ci_low"], r["late_ci_high"])]
        if extra:
            row.append(num(r["mean_duration"]))
        if flag is not None:
            row.append(flag.get(r[key], ""))
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #
def section_scope(df: pd.DataFrame, q: dict) -> str:
    kpi = an.kpi_summary(df)
    check("eligible orders (parquet rows vs quality report n_eligible)", int(len(df)), int(q["n_eligible"]))
    check("late orders (parquet vs quality report)", int(kpi["n_late"]), int(q["n_late_eligible"]))
    rng_all, rng_el = q["date_range_all"], q["date_range_eligible"]
    status = q["order_status_counts"]
    excl = q["exclusions"]
    status_rows = [[k, i(v), pct(v / q["n_orders_all"])] for k, v in status.items()]
    monthly = an.monthly_performance(df)
    sparse = monthly[monthly["n_orders"] < MIN_ORDERS]
    sparse_txt = ", ".join(f"{r.purchase_year_month} (n={i(r.n_orders)})" for r in sparse.itertuples())
    return f"""## 1. Dataset scope

Source: the public Olist Brazilian e-commerce dataset (Kaggle), as read by `src/preprocessing.py`. All numbers below
describe **delivered orders with valid dates** unless stated otherwise.

* Orders in the raw file: **{i(q['n_orders_raw'])}** (one row per order; {i(q['raw_files']['items']['rows'])} item rows, {i(q['raw_files']['sellers']['rows'])} sellers).
* Purchase-date range of all orders: {rng_all['min'][:10]} to {rng_all['max'][:10]}.
* **Analysis set (delivered, all three dates present, delivery not before purchase): {i(q['n_eligible'])} orders**
  ({pct(q['n_eligible'] / q['n_orders_all'])} of all orders). Purchase-date range of the analysis set: {rng_el['min'][:10]} to {rng_el['max'][:10]}.
* Excluded: {i(excl['non_delivered_status'])} orders with a non-delivered status and {i(excl['delivered_missing_delivery_date'])} delivered orders with no delivery date.
  They have no outcome and are never counted as on time.
* "Late" means the calendar date of customer delivery is after the calendar date of the estimated delivery
  ({q['late_definition']}). Under a stricter timestamp-based definition the late rate would be {pct(q['late_rate_timestamp_based_for_reference'], 2)} instead of {pct(q['late_rate_eligible'], 2)}.
* {state_count_line(df)}
* Months with fewer than {MIN_ORDERS} analysed orders (too sparse to interpret): {sparse_txt or 'none'}.

Order status in the raw file:

{table(['Status', 'Orders', 'Share of all orders'], status_rows)}
"""


def state_count_line(df: pd.DataFrame) -> str:
    return (f"Customers are in {df['customer_state'].nunique()} states, sellers in {df['seller_state'].nunique()} states, "
            f"products fall into {df['product_category'].nunique()} categories.")


def section_overall(df: pd.DataFrame) -> str:
    k = an.kpi_summary(df)
    lo, hi = an.wilson_ci(k["n_late"], k["n_orders"])
    delay = df["delivery_delay_days"].astype(float)
    early = (delay < 0).mean()
    ontime_day = (delay == 0).mean()
    late_delay = delay[delay > 0]
    return f"""## 2. Overall late rate

* **{pct(k['late_rate'], 2)} of delivered orders arrived after the estimated date** ({i(k['n_late'])} of {i(k['n_orders'])}; 95% Wilson interval {ci(lo, hi)}).
* {pct(early)} arrived before the estimated date, {pct(ontime_day)} on the estimated date.
* Late orders were late by a mean of {num(k['mean_late_days'])} days (median {num(late_delay.median())}, 90th percentile {num(late_delay.quantile(0.9))}, maximum {num(late_delay.max(), 0)}).

This is the average over the whole 2016-2018 history. Section 4 shows that the rate moved a lot between periods, so the
overall figure is not a stable property of the marketplace.
"""


def section_duration(df: pd.DataFrame) -> str:
    s = an.duration_summary(df)
    dur = df["delivery_duration_days"].astype(float)
    promise = df["estimated_delivery_days"].astype(float)
    edges = [0, 7, 14, 21, 30, np.inf]
    labels = ["up to 7 days", "7-14 days", "14-21 days", "21-30 days", "more than 30 days"]
    binned = pd.cut(dur, edges, labels=labels, right=True)
    counts = binned.value_counts().reindex(labels)
    rows = [[lab, i(c), pct(c / len(dur))] for lab, c in counts.items()]
    within14, within30 = float((dur <= 14).mean()), float((dur <= 30).mean())
    skew_txt = (f"The distribution is right-skewed (skewness {num(s['skew'], 2)}; mean {num(s['mean'])} days is above the median {num(s['median'])}). "
                if s["skew"] > 0.5 and s["mean"] > s["median"] else
                f"Skewness is {num(s['skew'], 2)} (mean {num(s['mean'])}, median {num(s['median'])}). ")
    prom_txt = (f"Promises are typically longer than actual delivery times (median promise {num(promise.median())} vs median delivery {num(s['median'])} days), "
                "which is why most orders arrive early or on the estimated date." if promise.median() > s["median"] else
                f"The median promise ({num(promise.median())} days) is not longer than the median delivery time ({num(s['median'])} days).")
    return f"""## 3. Delivery-time distribution

Delivery time = purchase to customer delivery, in days (n = {i(s['n'])}).

{table(['Statistic', 'Days'], [['mean', num(s['mean'])], ['median', num(s['median'])], ['standard deviation', num(s['std'])],
                                ['90th percentile', num(s['p90'])], ['95th percentile', num(s['p95'])],
                                ['minimum', num(s['min'])], ['maximum', num(s['max'])], ['skewness', num(s['skew'], 2)]])}

{skew_txt}Share delivered within 14 days: {pct(within14)}; within 30 days: {pct(within30)}; more than 30 days: {pct(1 - within30)}.

{table(['Delivery time', 'Orders', 'Share'], rows)}

The delivery promise shown at checkout (estimated date minus purchase date) has a median of {num(promise.median())} days
(mean {num(promise.mean())}, 10th to 90th percentile {num(promise.quantile(0.1), 0)} to {num(promise.quantile(0.9), 0)} days). {prom_txt}
"""


def section_months(df: pd.DataFrame) -> str:
    monthly = an.monthly_performance(df, MIN_ORDERS, ALPHA)
    screen = ins.month_screen(monthly, ALPHA)
    flagged = screen[(screen["diff"] > 0) & (screen["p_bonferroni"] < ALPHA)]
    fl = {r.month: "flagged (Bonferroni)" for r in flagged.itertuples()}
    rows = []
    for _, r in monthly.iterrows():
        m = r["purchase_year_month"]
        rows.append([m, i(r["n_orders"]), pct(r["late_rate"]), ci(r["late_ci_low"], r["late_ci_high"]),
                     num(r["mean_duration"]), fl.get(m, "" if r["reliable"] else f"n < {MIN_ORDERS}")])
    top = flagged.sort_values("late_rate", ascending=False)
    lines = [f"* {r.month}: {pct(r.late_rate)} late (n = {i(r.n_orders)}) vs {pct(r.rest_late_rate)} in all other months; "
             f"difference {pp(r.diff)} (95% CI {pp(r.diff_ci_low)} to {pp(r.diff_ci_high)})" for r in top.itertuples()]
    med = monthly[monthly["reliable"]]["late_rate"].median()
    return f"""## 4. Late rate by month

{table(['Purchase month', 'Orders', 'Late rate', '95% Wilson interval', 'Mean delivery days', 'Note'], rows, 'lrrrrl')}

Months compared with the pooled remaining months (two-proportion test, Bonferroni-adjusted over {len(screen)} months with n >= {MIN_ORDERS}).
{len(flagged)} months stand out as markedly higher:

{chr(10).join(lines)}

The median monthly late rate (months with n >= {MIN_ORDERS}) is {pct(med)}, so these months are several times the typical level.
The rate is **not stationary** (highest month {pct(monthly[monthly['reliable']]['late_rate'].max())}, lowest reliable month {pct(monthly[monthly['reliable']]['late_rate'].min())}).
This table cannot say why. Possible explanations (for example seasonal demand peaks or carrier disruption) are hypotheses; nothing in this data tests them.
"""


def windows(df: pd.DataFrame, mm: dict | None) -> dict[str, pd.DataFrame]:
    w = (mm or {}).get("split", {}).get("windows")
    if not w:
        return {}
    ts = df["order_purchase_timestamp"]
    out = {}
    for name, (a, b) in w.items():
        mask = ts >= pd.Timestamp(a)
        if b:
            mask &= ts < pd.Timestamp(b)
        out[name] = df[mask]
    return out


def section_regions(df: pd.DataFrame, wins: dict[str, pd.DataFrame]) -> str:
    overall = an.kpi_summary(df)["late_rate"]
    perf = an.state_performance(df, MIN_ORDERS, "customer_state", ALPHA)
    screened = perf[perf["reliable"]]
    hi = an.high_delay_segments(perf, MIN_ORDERS, overall, ALPHA, "high")
    lo = an.high_delay_segments(perf, MIN_ORDERS, overall, ALPHA, "low")
    flag = {**{k: "interval above overall" for k in hi["customer_state"]}, **{k: "interval below overall" for k in lo["customer_state"]}}
    small = perf[~perf["reliable"]]
    rows = perf_rows(screened.sort_values("late_rate", ascending=False), "customer_state", state, flag, extra=True)
    small_txt = ", ".join(f"{r.customer_state} (n={i(r.n_orders)})" for r in small.itertuples())
    # stability of the top flagged states across the model windows
    stab = ""
    if wins:
        top = list(hi["customer_state"].head(5))
        hdr = ["State", "All data"] + [f"{n} window" for n in wins]
        srows = []
        for st in top:
            row = [state(st), pct(perf.set_index("customer_state").loc[st, "late_rate"])]
            for n, w in wins.items():
                p = an.state_performance(w, 0, "customer_state", ALPHA).set_index("customer_state")
                row.append(f"{pct(p.loc[st, 'late_rate'])} (n={i(p.loc[st, 'n_orders'])})" if st in p.index else "n/a")
            srows.append(row)
        wl = ", ".join(f"{n}: {w['order_purchase_timestamp'].min():%Y-%m-%d} to {w['order_purchase_timestamp'].max():%Y-%m-%d}" for n, w in wins.items())
        tw = list(wins)[-1]
        n_drop = sum(1 for st in top if (lambda p: st in p.index and p.loc[st, "late_rate"] < overall)(
            an.state_performance(wins[tw], 0, "customer_state", ALPHA).set_index("customer_state")))
        stab_note = (f"In the latest ({tw}) window, {n_drop} of these {len(top)} states have a late rate below the overall all-history rate of {pct(overall)}"
                     f" (small samples: see n). "
                     + ("The regional pattern is therefore not stable across periods." if n_drop else "The top flagged states stay above the overall rate in the latest window."))
        stab = (f"\nDo the five highest flagged states stay high in each time window? (windows used for the model split, {wl})\n\n"
                f"{table(hdr, srows)}\n\n{stab_note}\n")
    return f"""## 5. Late rate by customer state (region)

Overall late rate for reference: {pct(overall)}. Screened: the {len(screened)} states with n >= {MIN_ORDERS}
(not screened: {small_txt or 'none'}). Each state is flagged when its unadjusted 95% Wilson interval lies entirely above
(or below) the overall rate: **{len(hi)} states above, {len(lo)} below**.

{table(['Customer state', 'Orders', 'Late rate', '95% Wilson interval', 'Mean delivery days', 'Flag'], rows, 'lrrrrl')}
{stab}
**How to read this.** The flags are unadjusted: {len(screened)} states were screened at the 95% level without a multiple-comparison
correction, so some flags can arise by chance and the flagged states are *leads for investigation*, not confirmed effects. Flagged
above overall: {', '.join(hi['customer_state'].head(8))}{' ...' if len(hi) > 8 else ''}. Flagged below overall: {', '.join(lo['customer_state'].head(8))}{' ...' if len(lo) > 8 else ''}.
That is a geographic association; this data does not isolate its cause (distance, carrier network and seller mix are all entangled with state).
"""


def section_categories(df: pd.DataFrame) -> str:
    overall = an.kpi_summary(df)["late_rate"]
    perf = an.category_performance(df, MIN_ORDERS, ALPHA)
    n_unknown = int(perf.loc[perf["product_category"] == ins.UNKNOWN_CATEGORY, "n_orders"].sum())
    perf = perf[perf["product_category"] != ins.UNKNOWN_CATEGORY]      # same rule as the app: "unknown" is not a category
    screened = perf[perf["reliable"]]
    hi = an.high_delay_segments(perf, MIN_ORDERS, overall, ALPHA, "high")
    lo = an.high_delay_segments(perf, MIN_ORDERS, overall, ALPHA, "low")
    flag = {**{k: "interval above overall" for k in hi["product_category"]}, **{k: "interval below overall" for k in lo["product_category"]}}
    top = screened.sort_values("late_rate", ascending=False).head(10)
    bot = screened.sort_values("late_rate").head(5)
    slow = screened.sort_values("mean_duration", ascending=False).head(5)
    n_cat = perf["product_category"].nunique()
    st_perf = an.state_performance(df, MIN_ORDERS, "customer_state", ALPHA)
    st_perf = st_perf[st_perf["reliable"]]
    mo = an.monthly_performance(df, MIN_ORDERS, ALPHA)
    mo = mo[mo["reliable"]]
    rng = lambda t: f"{pct(t['late_rate'].min())} to {pct(t['late_rate'].max())}"  # noqa: E731
    return f"""## 6. Late rate by product category

{n_cat} named categories exist (orders with no category, {i(n_unknown)}, are left out here as in the app); **{len(screened)} have n >= {MIN_ORDERS}** and were screened. Of those, **{len(hi)} have an interval entirely above the overall
{pct(overall)} and {len(lo)} entirely below** (unadjusted).

Ten highest late rates among screened categories:

{table(['Category', 'Orders', 'Late rate', '95% Wilson interval', 'Flag'], perf_rows(top, 'product_category', cat, flag), 'lrrrl')}

Five lowest:

{table(['Category', 'Orders', 'Late rate', '95% Wilson interval', 'Flag'], perf_rows(bot, 'product_category', cat, flag), 'lrrrl')}

Longest mean delivery times (screened categories):

{table(['Category', 'Orders', 'Mean days', 'Median days'], [[cat(r.product_category), i(r.n_orders), num(r.mean_duration), num(r.median_duration)] for r in slow.itertuples()])}

**How to read this.** Late rates of screened categories span {rng(screened)} (median category has {i(screened['n_orders'].median())} orders, so intervals are wide),
compared with {rng(st_perf)} across screened states and {rng(mo)} across months. {len(screened)} categories were screened without correction: expect a few flags by chance alone. Category is also entangled with
product size, price and which sellers (and states) sell it, none of which is adjusted for here. Treat these as leads.
"""


def section_routes(df: pd.DataFrame, wins: dict[str, pd.DataFrame]) -> str:
    overall = an.kpi_summary(df)["late_rate"]
    perf = an.state_pair_performance(df, MIN_ORDERS, ALPHA)
    screened = perf[perf["reliable"]]
    hi = an.high_delay_segments(perf, MIN_ORDERS, overall, ALPHA, "high")
    lo = an.high_delay_segments(perf, MIN_ORDERS, overall, ALPHA, "low")
    ss = an.same_state_comparison(df, MIN_ORDERS, ALPHA).set_index("group")

    def rt(t: pd.DataFrame) -> list[list[str]]:
        return [[r.state_pair, i(r.n_orders), pct(r.late_rate), ci(r.late_ci_low, r.late_ci_high), pp(r.excess_vs_overall)]
                for r in t.head(8).itertuples()]

    hdr = ["Route (seller to customer state)", "Orders", "Late rate", "95% Wilson interval", "vs overall"]
    ss_all = [["all data", i(ss.loc['cross_state', 'n_orders']), pct(ss.loc['cross_state', 'late_rate']), i(ss.loc['same_state', 'n_orders']),
               pct(ss.loc['same_state', 'late_rate']), pp(ss.loc['cross_state', 'late_rate'] - ss.loc['same_state', 'late_rate'])]]
    flip = []
    for n, w in wins.items():
        s = an.same_state_comparison(w, 0, ALPHA).set_index("group")
        diff = s.loc["cross_state", "late_rate"] - s.loc["same_state", "late_rate"]
        flip.append(diff)
        ss_all.append([f"{n} window", i(s.loc['cross_state', 'n_orders']), pct(s.loc['cross_state', 'late_rate']), i(s.loc['same_state', 'n_orders']),
                       pct(s.loc['same_state', 'late_rate']), pp(diff)])
    sign_note = ""
    if flip and (min(flip) < 0 < max(flip)):
        sign_note = (f"\n**The cross-state minus same-state gap changes sign between windows ({', '.join(pp(x) for x in flip)} in the "
                     "windows in the order shown).** Over the full history cross-state orders are later, but in the most recent window "
                     "the relationship is reversed. A route or same-state pattern learned from the earlier data therefore need not hold "
                     "for later orders; this matters directly for the prediction model (section 9).\n")
    return f"""## 7. Late rate by seller-to-customer route

{len(screened)} seller-state to customer-state routes have n >= {MIN_ORDERS} and were screened; **{len(hi)} have an interval entirely above the overall
{pct(overall)} and {len(lo)} entirely below** (unadjusted; {len(hi) + len(lo)} of {len(screened)} routes flagged in total, and a route is also a bundle of customer state, seller state, distance and carrier lanes).

Highest late-rate routes flagged above overall (top 8 by excess):

{table(hdr, rt(hi), 'lrrrr')}

Lowest late-rate routes flagged below overall (top 8 by deficit):

{table(hdr, rt(lo), 'lrrrr')}

Same-state (seller and customer in the same state) versus cross-state orders:

{table(['Period', 'Cross-state orders', 'Cross-state late', 'Same-state orders', 'Same-state late', 'Cross minus same'], ss_all, 'lrrrrr')}
{sign_note}
Mean delivery time over the full history: cross-state {num(ss.loc['cross_state', 'mean_duration'])} days vs same-state {num(ss.loc['same_state', 'mean_duration'])} days.
"""


def section_associations(df: pd.DataFrame) -> str:
    items = {x.kind: x for x in ins.generate_insights(df, MIN_ORDERS, ALPHA)}
    wf, rt, pt = items["weight_freight"].metrics, items["region_test"].metrics, items["processing_time"].metrics

    def rho(d: dict) -> str:
        return f"{d['rho']:.2f} ({d['strength_label']}; 95% CI {d['ci_low']:.2f} to {d['ci_high']:.2f}, n = {i(d['n'])})"

    return f"""## 8. Other associations

* Customer state vs late delivery: chi-square = {num(rt['chi2'])}, df = {int(rt['dof'])}, Cramer's V = {rt['cramers_v']:.3f} ({rt['effect_label']} effect). Statistically
  detectable with {i(rt['n'])} orders; the effect size is labelled {rt['effect_label']}, so state on its own separates late from on-time orders only partly.
* Spearman rank correlation with delivery time: freight cost rho = {rho(wf['freight_vs_duration'])}; order weight rho = {rho(wf['weight_vs_duration'])}.
  Freight and weight are themselves correlated (rho = {wf['weight_vs_freight']['rho']:.2f}), and freight also tracks distance. These do not show that weight or freight *cause* longer delivery.
* **Retrospective, not usable for prediction:** processing time (purchase to carrier hand-over; mean {num(pt['mean_processing_days'])} days, about {pct(pt['share_of_duration'], 0)} of total delivery time).
  Orders in the slowest quarter ({num(pt['q75_processing_days'])}+ days) were late {pct(pt['slow_late_rate'])} of the time (n = {i(pt['slow_n'])}) vs {pct(pt['rest_late_rate'])} for the rest
  (n = {i(pt['rest_n'])}); difference {pp(pt['diff'])}. This is only known after the order was handed to the carrier, and part of the association is plausibly mechanical (a late hand-over leaves less time to arrive).
"""


def section_model(mm: dict | None, wins: dict[str, pd.DataFrame]) -> str:
    if not mm:
        return "## 9. Prediction model\n\n`models/model_metrics.json` was not found; run `.venv/bin/python -m src.train_model`.\n"
    split = mm["split"]
    sel = mm["selected_model"]
    name = sel["name"]
    m = mm["models"][name]
    te, va = m["test"], m["validation"]
    lift_te = te["pr_auc"] / te["positive_rate"]
    lift_va = va["pr_auc"] / va["positive_rate"]
    ci_pr = m["test_ci95"]["pr_auc"]
    if wins:
        for n_, w in wins.items():
            check(f"{n_} window: n orders (data vs model_metrics.json split)", int(len(w)), int(split[n_]["n"]))
            check(f"{n_} window: late rate", float(w["is_late"].astype(float).mean()), float(split[n_]["late_rate"]), 1e-9)
    srows = [[n_, f"{split[n_]['start']} to {split[n_]['end']}", i(split[n_]["n"]), i(split[n_]["n_late"]), pct(split[n_]["late_rate"], 2)]
             for n_ in ("train", "validation", "test")]
    mrows = []
    for k, r in mm["models"].items():
        v, t = r["validation"], r["test"]
        mrows.append([r["label"] + (" (selected)" if k == name else ""), f"{v['pr_auc']:.3f} / {t['pr_auc']:.3f}", f"{v['roc_auc']:.3f} / {t['roc_auc']:.3f}",
                      f"{v['brier']:.4f} / {t['brier']:.4f}", f"{v['log_loss']:.4f} / {t['log_loss']:.4f}",
                      f"{t['precision']:.3f} / {t['recall']:.3f} / {t['f1']:.3f}", f"{t['lift_top10pct']:.2f}x"])
    tn, fp = m["confusion_matrix_test"][0]
    fn, tp = m["confusion_matrix_test"][1]
    bands = mm["risk_bands"]
    brows = []
    for b_v, b_t in zip(bands["validation"], bands["test"]):
        brows.append([b_v["band"], f"{i(b_v['n'])}", f"{pct(b_v['late_rate'])} ({ci(b_v['late_ci_low'], b_v['late_ci_high'])})",
                      f"{i(b_t['n'])}", f"{pct(b_t['late_rate'])} ({ci(b_t['late_ci_low'], b_t['late_ci_high'])})"])
    pi = mm["permutation_importance"]
    prow = []
    te_items = {x["feature"]: x for x in pi.get("test_items", [])}
    for rank, x in enumerate(pi["items"][:8], 1):
        t = te_items.get(x["feature"])
        prow.append([f"{rank}. {x['feature']}", f"{x['importance_mean']:+.4f}", f"{t['importance_mean']:+.4f}" if t else "n/a"])
    cs = split.get("covariate_support", {}).get("test", {})
    caveat = sel["test_window_caveat"]
    strength = ("weak (close to chance)" if te["roc_auc"] < 0.6 else "modest" if te["roc_auc"] < 0.7 else "moderate")
    base_vs = ("does NOT beat" if caveat["brier_not_better_than_baseline"] else "beats")
    ss = caveat["by_same_state"]
    warns = "\n".join(f"* {w}" for w in sel["test_window_warnings"]) or "* none"
    leak = mm["leakage_audit"]

    def band_story(rows: list[dict], label: str) -> str:
        rates = [r["late_rate"] for r in rows]
        mono = all(a < b for a, b in zip(rates, rates[1:]))
        ov = [f"{a['band']}-{b['band']}" for a, b in zip(rows, rows[1:]) if a["late_ci_high"] >= b["late_ci_low"]]
        return (f"On {label} the observed late rate {'rises' if mono else 'does not rise'} monotonically from {rows[0]['band']} to {rows[-1]['band']} "
                f"({' / '.join(pct(r) for r in rates)}); " + (f"adjacent bands with overlapping 95% intervals: {', '.join(ov)}." if ov else "no adjacent bands overlap."))

    band_txt = (band_story(bands["validation"], "validation") + " " + band_story(bands["test"], "test") +
                f" The High-to-Low late-rate ratio is {bands['validation'][-1]['late_rate'] / bands['validation'][0]['late_rate']:.1f}x on validation "
                f"and {bands['test'][-1]['late_rate'] / bands['test'][0]['late_rate']:.1f}x on test.")
    return f"""## 9. Prediction model

**Task.** Predict, from information known at purchase time only, whether an order will arrive after its estimated date
(target: {mm['target']}). Features: {len(mm['features']['numeric'])} numeric and {len(mm['features']['categorical'])} categorical purchase-time columns
(customer and seller state, product category, same-state flag, order size, price, freight, weight and volume, the checkout delivery promise, day of week).
Outcome-side columns (processing time, review score, status, delivery dates) are excluded; the leakage audit in the metrics file
{'passed' if leak['passed'] else 'FAILED'} ({len(leak['checks'])} checks).

**Split (chronological, no order appears twice).**

{table(['Partition', 'Purchase dates', 'Orders', 'Late', 'Late rate'], srows, 'lrrrr')}

The late rate differs a lot between partitions ({pct(split['train']['late_rate'])} train, {pct(split['validation']['late_rate'])} validation, {pct(split['test']['late_rate'])} test), so the
partitions are different regimes. Orders purchased before 2017-01-01 are excluded from modelling (very few orders per month).

**All candidate models** (validation / test; validation figures are tuned and therefore optimistic, test is the untouched later window;
precision / recall / F1 are at each model's own validation-chosen threshold):

{table(['Model', 'PR-AUC', 'ROC-AUC', 'Brier', 'Log-loss', 'Precision / recall / F1 (test)', 'Lift in top 10% (test)'], mrows, 'lrrrrrr')}

PR-AUC of an uninformative model equals the late rate ({pct(te['positive_rate'], 2)} on the test window); ROC-AUC of an uninformative model is 0.5. The dummy row is that baseline.

**Selected model: {m['label']}** ({'isotonic-calibrated' if m['calibrated'] else 'uncalibrated'}), chosen on validation PR-AUC as the simplest model within one bootstrap standard deviation of the best.

* Validation (tuned, optimistic): ROC-AUC {va['roc_auc']:.3f}, PR-AUC {va['pr_auc']:.3f}, a {lift_va:.1f}x lift over the {pct(va['positive_rate'], 2)} base rate.
* **Test (out of sample, later window): ROC-AUC {te['roc_auc']:.3f} (95% bootstrap interval {m['test_ci95']['roc_auc'][0]:.3f} to {m['test_ci95']['roc_auc'][1]:.3f}),
  PR-AUC {te['pr_auc']:.3f} (interval {ci_pr[0]:.3f} to {ci_pr[1]:.3f}) against a no-skill {te['positive_rate']:.3f}: a {lift_te:.1f}x lift.**
* At the selected threshold ({sel['threshold']:.3f}) the test window gives: {i(tp)} late orders caught and {i(fn)} missed (recall {pct(te['recall'])}), {i(fp)} on-time orders flagged
  (precision {pct(te['precision'])}), and {pct(te['predicted_positive_rate'])} of all orders flagged. Accuracy ({pct(te['accuracy'])}) is not informative: always predicting "on time" scores {pct(1 - te['positive_rate'])}.
* Probability quality on test: Brier {te['brier']:.4f} against {caveat['brier_baseline']:.4f} for always quoting the base rate; log-loss {te['log_loss']:.4f} against
  {caveat['log_loss_baseline']:.4f}. The model {base_vs} the constant baseline on Brier score. Mean predicted risk is {pct(caveat['mean_predicted'])} against an observed {pct(caveat['observed_late_rate'])}.
* Calibration changed test PR-AUC from {m['test_pr_auc_uncalibrated']:.3f} (uncalibrated) to {te['pr_auc']:.3f}.

**Plain assessment.** On the later test window the model's ranking ability is **{strength}**: ROC-AUC {te['roc_auc']:.3f} (0.5 is chance) and a {lift_te:.1f}x PR-AUC lift
(validation looked much better at {lift_va:.1f}x). Its probabilities are not better calibrated than the base rate, and performance degraded from validation to test.
It should be treated as a coarse screening aid with a modest signal, not as a reliable forecaster, and the stated probabilities should not be read as precise.
Warnings recorded by the training script:

{warns}

**Risk bands** (cut points from validation scores; observed late rate by band):

{table(['Band', 'Validation orders', 'Validation late rate (95% CI)', 'Test orders', 'Test late rate (95% CI)'], brows, 'lrrrr')}

{band_txt}

**Where the training-to-test mismatch shows up: the same-state relationship reversed.** On the test window the same-state orders were late {pct(ss['same_state']['observed_late_rate'])} of the time
(n = {i(ss['same_state']['n'])}) and other-state orders {pct(ss['other_state']['observed_late_rate'])} (n = {i(ss['other_state']['n'])}), but the model had mean predicted risk {pct(ss['same_state']['mean_predicted'])} for same-state and
{pct(ss['other_state']['mean_predicted'])} for other-state orders, consistent with the model having learned from the training window, where the pattern ran the other way. In addition {pct(cs.get('share_outside', float('nan')))} of test orders
({i(cs.get('n_outside', 0))}) have a checkout promise outside the range seen in training (late rate {pct(cs.get('late_rate_outside', float('nan')))} for those vs {pct(cs.get('late_rate_inside', float('nan')))} for the rest). These are descriptive
observations of drift; they are not shown to be the cause of the weaker test performance, and the data does not show what changed.

**What the model leans on** (permutation importance: drop in PR-AUC when a column is shuffled; predictive association, not cause; the validation column is tuned, the test column is diagnostic):

{table(['Feature (validation rank)', 'Validation', 'Test'], prow, 'lrr')}

Model fitting time recorded in the metrics file: {num(mm['training_seconds'], 0)} s; trained {mm['generated_at'][:10]} with scikit-learn {mm['library_versions']['scikit-learn']}.
"""


def section_caveats(q: dict, mm: dict | None) -> str:
    an_ = q["logical_order_anomalies_delivered_orders"]
    eq = q["eligible_quality"]
    pq = q["product_quality"]
    split_notes = (mm or {}).get("split", {}).get("notes", [])
    cens = next((n for n in split_notes if n.startswith("Selection bias")), None)
    return f"""## 10. Data caveats

From `data/processed/data_quality_report.json` (generated {q['generated_at_utc'][:10]}):

* **Only delivered orders are labelled.** {i(q['exclusions']['non_delivered_status'])} orders in a non-delivered status (shipped, canceled, unavailable, invoiced, processing, created, approved) and
  {i(q['exclusions']['delivered_missing_delivery_date'])} delivered orders without a delivery date are not in any rate above. Orders still in transit at the end of the data may be disproportionately late, so recent months can be understated.{(' ' + cens) if cens else ''}
* **Timestamp anomalies among delivered orders:** {i(an_['carrier_before_purchase'])} carrier hand-over dates earlier than purchase (processing time set to missing), {i(an_['customer_delivery_before_carrier'])} customer deliveries
  dated before the carrier hand-over, {i(an_['carrier_before_approval'])} hand-overs before payment approval. None affect the late label (which uses purchase, estimate and delivery dates only).
* **Product attributes:** {i(eq['category_missing'])} analysed orders have no product category (grouped as "unknown"); {i(pq['zero_weight_products'])} products have a zero weight and {i(pq['products_missing_weight'])} have none
  ({i(eq['weight_missing'])} analysed orders affected). Non-positive weights or dimensions are treated as missing.
* **Multi-item orders:** {i(q['multi_item_orders'])} orders have several items, {i(q['multi_seller_orders'])} several sellers and {i(q['multi_category_orders'])} several categories. Each order is assigned one
  representative seller (largest summed price) and category (highest-priced item), so seller-state, route and category results describe that representative, not the whole basket.
* **Two categories were missing from the official translation file** ({', '.join(q['category_translation']['categories_missing_from_translation'])}) and were mapped by a documented supplement.
* **Lateness definition is calendar-date based**: arriving any time on the estimated date counts as on time. The timestamp-based alternative gives {pct(q['late_rate_timestamp_based_for_reference'], 2)}.
* **One marketplace, 2016-2018, no causal design.** No carrier, weather, holiday, stock or seller-capacity data is available, and customers and sellers cannot be compared on a like-for-like basis.
"""


# --------------------------------------------------------------------------- #
def build() -> str:
    df = pd.read_parquet(config.ORDERS_DELIVERED_PATH)
    q = json.loads(config.QUALITY_REPORT_PATH.read_text())
    mm = json.loads(config.MODEL_METRICS_PATH.read_text()) if config.MODEL_METRICS_PATH.is_file() else None
    wins = windows(df, mm)
    parts = [
        section_scope(df, q), section_overall(df), section_duration(df), section_months(df), section_regions(df, wins),
        section_categories(df), section_routes(df, wins), section_associations(df), section_model(mm, wins), section_caveats(q, mm),
    ]
    head = f"""# DeliveryIQ: findings summary

*Generated by `scripts/generate_findings.py` on {date.today().isoformat()} from `data/processed/orders_delivered.parquet`,
`data/processed/data_quality_report.json` and `models/model_metrics.json`. Do not edit by hand: every number below is computed by the script.
Regenerate with `.venv/bin/python scripts/generate_findings.py`.*

## Read this first

* This is a **descriptive** analysis of historical Olist orders (2016-2018). Every finding is an **association**. Nothing here shows what *causes* a late delivery.
* Segment screens (state, category, route) test many segments at once at the 95% level **without adjusting for multiple comparisons** (the month screen is Bonferroni-adjusted).
  A flagged segment is a **lead to investigate**, not a confirmed effect, and several flags would be expected by chance alone.
* Late rates were **not stable over time** (section 4), and some relationships changed sign or weakened between time windows (sections 5 to 9). Figures averaged over the whole history can mislead about any one period.
* The prediction model's out-of-sample (later-window) performance is **modest**; see section 9 for the numbers before relying on any score.
* "Late" = delivered on a calendar date after the estimated delivery date. "Orders" means delivered orders with valid dates unless stated.

"""
    tail = "\n## 11. Reproducibility checks\n\nIndependent sources compared by the script (a `FAIL` here means the processed data and the recorded reports are out of sync):\n\n" + \
           "\n".join(f"* `{c}`" for c in CHECKS) + "\n"
    return head + "\n".join(parts) + tail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "FINDINGS.md")
    args = ap.parse_args()
    if not config.ORDERS_DELIVERED_PATH.is_file():
        print(f"Processed data not found at {config.ORDERS_DELIVERED_PATH}. Run: .venv/bin/python -m src.preprocessing", file=sys.stderr)
        return 1
    text = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out} ({len(text.splitlines())} lines)")
    for c in CHECKS:
        print(c)
    return 1 if any(c.startswith("FAIL") for c in CHECKS) else 0


if __name__ == "__main__":
    raise SystemExit(main())
