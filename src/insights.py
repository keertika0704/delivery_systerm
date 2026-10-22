"""Computed insights, recommendations and exportable reports for DeliveryIQ.

Nothing here is hard-coded: every finding is derived from the (filtered) frame passed in, so
insights change when the filters change.  Wording is deliberately associational ("associated
with", "investigate", "review") -- never a root-cause claim.

Severity rules
--------------
* ``info``  -- descriptive / no reliable difference found.
* ``watch`` -- a statistically reliable difference exists (Wilson interval of a segment excludes
  the overall rate, or a Bonferroni-adjusted test is significant).
* ``alert`` -- reliable AND the worst segment's late rate is at least ``ALERT_RATIO`` times
  the comparison rate.

Wording: in the insight text "reliably higher / lower" means the segment's unadjusted Wilson
interval excludes the overall rate; this is distinct from the ``reliable`` column of the
analytics tables, which only means ``n_orders >= min_orders``.

Segments are only ranked when ``n_orders >= min_orders``; when no segment of a kind qualifies,
a single ``insufficient_data`` insight says so instead of ranking tiny samples.  Segment
screens use unadjusted Wilson intervals at level ``1 - alpha`` (95% by default; many segments are
inspected, so treat a flagged segment as a lead to investigate, not a confirmed effect); the
month screen is Bonferroni-adjusted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import analytics as an
from . import statistical_analysis as sa
from .config import DEFAULT_ALPHA, DEFAULT_MIN_ORDERS, STATE_NAMES

__all__ = ["Insight", "generate_insights", "recommendations", "summary_report", "regional_report",
           "ALERT_RATIO"]

ALERT_RATIO = 2.0
TOP_K = 3
UNKNOWN_CATEGORY = "unknown"


@dataclass
class Insight:
    """One computed finding: ``metrics`` hold the raw numbers, ``n`` the orders it rests on."""

    kind: str
    title: str
    text: str
    metrics: dict = field(default_factory=dict)
    n: int = 0
    severity: str = "info"  # "info" | "watch" | "alert"


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def _pct(x: float, digits: int = 1) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.{digits}f}%"


def _pp(x: float) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:+.1f} pp"


def _num(x: float, digits: int = 1) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:,.{digits}f}"


def _p(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "p = n/a"
    if p == 0:
        return "p < 0.001"
    return f"p = {p:.2g}" if p < 0.001 else f"p = {p:.3f}"


def _state_label(code: str) -> str:
    name = STATE_NAMES.get(str(code))
    return f"{code} ({name})" if name else str(code)


def _lvl(alpha: float) -> str:
    """Confidence level text for ``alpha`` (``0.05`` -> ``95%``)."""
    return f"{100 * (1 - alpha):g}%"


def _lead_note(alpha: float, n_screened: int) -> str:
    """Multiple-comparison caveat attached to every unadjusted segment screen."""
    return (f"Unadjusted {_lvl(alpha)} intervals across {n_screened:,} segments screened; treat "
            "flagged segments as leads, not confirmed effects.")


def _assoc_summary(s_w: dict, s_f: dict) -> str:
    """Closing sentence for the weight / freight insight, worded from the computed Spearman results."""
    def _clause(r: dict, name: str) -> str:
        if r["strength_label"] == "negligible":
            return f"{name} shows a negligible rank association"
        return f"{name} shows a {r['strength_label']} {r['direction']} rank association"
    if s_w["strength_label"] == "negligible" and s_f["strength_label"] == "negligible":
        return "Neither weight nor freight cost shows more than a negligible rank association with delivery time."
    return (_clause(s_w, "Order weight") + " and " + _clause(s_f, "freight cost")
            + " with delivery time.")


def _seg_text(row: pd.Series, key: str, label_fn=str, alpha: float = DEFAULT_ALPHA) -> str:
    return (f"{label_fn(row[key])}: {_pct(row['late_rate'])} late (n = {int(row['n_orders']):,}; "
            f"{_lvl(alpha)} CI {_pct(row['late_ci_low'])} to {_pct(row['late_ci_high'])})")


def _records(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    return df[[c for c in cols if c in df.columns]].to_dict("records")


SEG_COLS = ["n_orders", "n_late", "late_rate", "late_ci_low", "late_ci_high", "mean_duration"]


# --------------------------------------------------------------------------- #
# Insight generation
# --------------------------------------------------------------------------- #
def _month_screen(monthly: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Each reliable month vs all other months (pooled): z-test + Newcombe CI, Bonferroni p."""
    rel = monthly[monthly["reliable"]]
    n_tot, k_tot = int(monthly["n_orders"].sum()), int(monthly["n_late"].sum())
    rows = []
    for _, r in rel.iterrows():
        k1, n1 = int(r["n_late"]), int(r["n_orders"])
        k2, n2 = k_tot - k1, n_tot - n1
        if n2 <= 0:
            continue
        z = sa.two_proportion_ztest(k1, n1, k2, n2)
        ci = sa.proportion_diff_ci(k1, n1, k2, n2, alpha)
        rows.append({"month": r["purchase_year_month"], "n_orders": n1, "n_late": k1,
                     "late_rate": k1 / n1, "rest_n": n2, "rest_late_rate": k2 / n2,
                     "diff": ci["diff"], "diff_ci_low": ci["ci_low"], "diff_ci_high": ci["ci_high"],
                     "p_value": z["p_value"]})
    out = pd.DataFrame(rows, columns=["month", "n_orders", "n_late", "late_rate", "rest_n",
                                      "rest_late_rate", "diff", "diff_ci_low", "diff_ci_high",
                                      "p_value"])
    out["p_bonferroni"] = np.minimum(1.0, out["p_value"] * max(len(out), 1))
    return out


def generate_insights(df: pd.DataFrame, min_orders: int = DEFAULT_MIN_ORDERS,
                      alpha: float = DEFAULT_ALPHA) -> list[Insight]:
    """Build the list of computed insights for a (filtered) delivered frame.

    Order: overview, regions, statistical regional test, months, categories, routes,
    same-state vs cross-state, weight / freight association, processing time (retrospective),
    and finally one ``insufficient_data`` insight if any section lacked enough orders.  An
    empty frame gives a single ``no_data`` insight.
    """
    k = an.kpi_summary(df)
    n = k["n_orders"]
    if n == 0:
        return [Insight("no_data", "No data for the current filters",
                        "No delivered orders match the current filters, so nothing can be "
                        "computed. Widen the date range or clear some filters.", {"n_orders": 0}, 0)]
    overall = k["late_rate"]
    lo, hi = an.wilson_ci(k["n_late"], n, alpha)
    out: list[Insight] = [Insight(
        "overview", "Overall delivery performance",
        f"{n:,} delivered orders: {_pct(overall)} arrived after the estimated date "
        f"({_lvl(alpha)} CI {_pct(lo)} to {_pct(hi)}; n_late = {k['n_late']:,}). Mean delivery time is "
        f"{_num(k['mean_duration'])} days (median {_num(k['median_duration'])}); late orders were "
        f"on average {_num(k['mean_late_days'])} days late.",
        {**k, "late_ci_low": lo, "late_ci_high": hi, "alpha": alpha}, n)]

    if n < min_orders:
        out.append(Insight(
            "insufficient_data", "Insufficient data to draw conclusions",
            f"Only {n:,} orders match the current filters, fewer than the minimum of "
            f"{min_orders:,} needed to compare regions, months, categories or routes. Rankings "
            "and tests are skipped rather than reported on a tiny sample.",
            {"n_orders": n, "min_orders": min_orders, "sections": ["all comparisons"]}, n))
        return out

    skipped: list[str] = []

    # ---- regions ---------------------------------------------------------- #
    states = an.state_performance(df, min_orders, alpha=alpha)
    if not states["reliable"].any():
        skipped.append("regional comparison")
    else:
        high = an.high_delay_segments(states, min_orders, overall, alpha, "high")
        low = an.high_delay_segments(states, min_orders, overall, alpha, "low")
        n_rel = int(states["reliable"].sum())
        if len(high):
            top = high.head(5)
            sev = "alert" if high["rate_ratio"].iloc[0] >= ALERT_RATIO else "watch"
            out.append(Insight(
                "region_high", f"{len(high)} region(s) with a reliably higher late rate",
                f"Compared with the overall {_pct(overall)}, these customer states have a late rate "
                f"whose {_lvl(alpha)} interval lies entirely above it: "
                + "; ".join(_seg_text(r, "customer_state", _state_label, alpha) for _, r in top.iterrows())
                + f". Screened {n_rel} regions with n >= {min_orders:,}. " + _lead_note(alpha, n_rel),
                {"segments": _records(top, ["customer_state", *SEG_COLS, "excess_vs_overall", "rate_ratio"]),
                 "n_flagged": int(len(high)), "n_screened": n_rel, "overall_late_rate": overall},
                int(high["n_orders"].sum()), sev))
        else:
            out.append(Insight(
                "region_high", "No region reliably above the overall late rate",
                f"None of the {n_rel} regions with n >= {min_orders:,} has a late-rate interval "
                f"entirely above the overall {_pct(overall)} ({_lvl(alpha)} interval, unadjusted).",
                {"n_flagged": 0, "n_screened": n_rel, "overall_late_rate": overall},
                int(states.loc[states["reliable"], "n_orders"].sum())))
        if len(low):
            top = low.head(5)
            out.append(Insight(
                "region_low", f"{len(low)} region(s) with a reliably lower late rate",
                f"These customer states have a late rate whose {_lvl(alpha)} interval lies entirely below the "
                f"overall {_pct(overall)}: "
                + "; ".join(_seg_text(r, "customer_state", _state_label, alpha) for _, r in top.iterrows())
                + f". Screened {n_rel} regions with n >= {min_orders:,}. " + _lead_note(alpha, n_rel),
                {"segments": _records(top, ["customer_state", *SEG_COLS, "excess_vs_overall", "rate_ratio"]),
                 "n_flagged": int(len(low)), "n_screened": n_rel, "overall_late_rate": overall},
                int(low["n_orders"].sum())))
        chi = sa.chi_square_test(df, "customer_state", alpha=alpha)
        if np.isfinite(chi["chi2"]):
            relr = states[states["reliable"]]
            worst, best = relr.iloc[0], relr.iloc[-1]
            sig = chi["significant"] and chi["effect_label"] != "negligible"
            out.append(Insight(
                "region_test", "Regional differences: statistical test",
                f"A chi-square test of customer state vs late delivery (n = {chi['n']:,}) gives "
                f"chi2 = {chi['chi2']:.1f}, df = {chi['dof']}, {_p(chi['p_value'])}, Cramer's V = "
                f"{chi['cramers_v']:.3f} ({chi['effect_label']} effect). Among regions with "
                f"n >= {min_orders:,} the late rate ranges from {_pct(best['late_rate'])} "
                f"({_state_label(best['customer_state'])}, n = {int(best['n_orders']):,}) to "
                f"{_pct(worst['late_rate'])} ({_state_label(worst['customer_state'])}, "
                f"n = {int(worst['n_orders']):,}). This is an association, not a cause.",
                {"chi2": chi["chi2"], "dof": chi["dof"], "p_value": chi["p_value"],
                 "cramers_v": chi["cramers_v"], "effect_label": chi["effect_label"],
                 "n": chi["n"], "min_late_rate": float(best["late_rate"]),
                 "max_late_rate": float(worst["late_rate"])},
                chi["n"], "watch" if sig else "info"))

    # ---- months ----------------------------------------------------------- #
    monthly = an.monthly_performance(df, min_orders, alpha)
    if int(monthly["reliable"].sum()) < 2:
        skipped.append("month-by-month comparison")
    else:
        scr = _month_screen(monthly, alpha)
        flagged = scr[(scr["diff"] > 0) & (scr["p_bonferroni"] < alpha)].sort_values(
            "diff", ascending=False, kind="mergesort")
        if len(flagged):
            top = flagged.head(TOP_K)
            sev = "alert" if (top["late_rate"] / top["rest_late_rate"].replace(0, np.nan)
                              ).max() >= ALERT_RATIO else "watch"
            parts = [f"{r['month']}: {_pct(r['late_rate'])} late (n = {int(r['n_orders']):,}) vs "
                     f"{_pct(r['rest_late_rate'])} in all other months (n = {int(r['rest_n']):,}); "
                     f"difference {_pp(r['diff'])}, {_lvl(alpha)} CI {_pp(r['diff_ci_low'])} to "
                     f"{_pp(r['diff_ci_high'])}, Bonferroni-adjusted {_p(r['p_bonferroni'])}"
                     for _, r in top.iterrows()]
            out.append(Insight(
                "month_high", f"{len(flagged)} month(s) with a markedly higher late rate",
                "Each month was compared with the pooled remaining months (two-proportion test, "
                f"Bonferroni-adjusted over {len(scr)} months): " + "; ".join(parts) + ".",
                {"months": top.to_dict("records"), "n_flagged": int(len(flagged)),
                 "n_tested": int(len(scr))}, int(flagged["n_orders"].sum()), sev))
        else:
            out.append(Insight(
                "month_high", "No month stands out after multiple-testing adjustment",
                f"None of the {len(scr)} months with n >= {min_orders:,} has a late rate significantly "
                "above the other months (Bonferroni-adjusted).",
                {"n_flagged": 0, "n_tested": int(len(scr))},
                int(monthly.loc[monthly["reliable"], "n_orders"].sum())))
        relm = monthly[monthly["reliable"]].sort_values(["late_rate", "n_orders"], ascending=[False, False],
                                                       kind="mergesort")
        w, b = relm.iloc[0], relm.iloc[-1]
        out.append(Insight(
            "month_extremes", "Worst and best months by late rate",
            f"Highest late rate: {w['purchase_year_month']} at {_pct(w['late_rate'])} "
            f"(n = {int(w['n_orders']):,}, {_lvl(alpha)} CI {_pct(w['late_ci_low'])} to {_pct(w['late_ci_high'])}). "
            f"Lowest: {b['purchase_year_month']} at {_pct(b['late_rate'])} "
            f"(n = {int(b['n_orders']):,}, {_lvl(alpha)} CI {_pct(b['late_ci_low'])} to {_pct(b['late_ci_high'])}). "
            f"Only months with n >= {min_orders:,} are ranked.",
            {"worst_month": w["purchase_year_month"], "worst_late_rate": float(w["late_rate"]),
             "worst_n": int(w["n_orders"]), "best_month": b["purchase_year_month"],
             "best_late_rate": float(b["late_rate"]), "best_n": int(b["n_orders"])},
            int(relm["n_orders"].sum())))

    # ---- categories ------------------------------------------------------- #
    cats = an.category_performance(df, min_orders, alpha)
    cats = cats[cats["product_category"] != UNKNOWN_CATEGORY]
    relc = cats[cats["reliable"]]
    if relc.empty:
        skipped.append("product-category ranking")
    else:
        dur = relc.sort_values("mean_duration", ascending=False, kind="mergesort").head(TOP_K)
        out.append(Insight(
            "category_duration", "Categories with the longest delivery times",
            "Longest mean delivery time (categories with n >= "
            f"{min_orders:,}): " + "; ".join(
                f"{r['product_category']}: {_num(r['mean_duration'])} days (median "
                f"{_num(r['median_duration'])}, n = {int(r['n_orders']):,})" for _, r in dur.iterrows())
            + f". Overall mean is {_num(k['mean_duration'])} days.",
            {"segments": _records(dur, ["product_category", "n_orders", "mean_duration",
                                        "median_duration"]), "overall_mean_duration": k["mean_duration"]},
            int(dur["n_orders"].sum())))
        flag = an.high_delay_segments(cats, min_orders, overall, alpha, "high")
        flagged_set = set(flag["product_category"])
        dly = relc.sort_values("late_rate", ascending=False, kind="mergesort").head(TOP_K)
        out.append(Insight(
            "category_delay", "Categories with the highest late rates",
            "Highest late rate among categories with n >= " f"{min_orders:,}: " + "; ".join(
                _seg_text(r, "product_category", alpha=alpha) + (" [reliably above overall]"
                                                                 if r["product_category"] in flagged_set else "")
                for _, r in dly.iterrows()) + f". Overall late rate is {_pct(overall)}. "
            + _lead_note(alpha, len(relc)),
            {"segments": _records(dly, ["product_category", *SEG_COLS]),
             "flagged": [c for c in dly["product_category"] if c in flagged_set],
             "n_reliably_above": len(flagged_set), "n_screened": int(len(relc)),
             "overall_late_rate": overall},
            int(dly["n_orders"].sum()), "watch" if flagged_set & set(dly["product_category"]) else "info"))
        cost = an.category_cost_comparison(df, min_orders)
        cost = cost[cost["reliable"] & (cost["product_category"] != UNKNOWN_CATEGORY)].head(TOP_K)
        if len(cost):
            out.append(Insight(
                "category_freight", "Categories with the highest freight cost",
                "Highest mean freight per order: " + "; ".join(
                    f"{r['product_category']}: {_num(r['mean_freight'], 2)} (freight/price "
                    f"{_num(r['mean_freight_ratio'], 2)}, n = {int(r['n_orders']):,})"
                    for _, r in cost.iterrows())
                + f". Overall mean freight per order is {_num(k['total_freight'] / n, 2)}.",
                {"segments": _records(cost, ["product_category", "n_orders", "mean_freight",
                                             "mean_freight_ratio"]),
                 "overall_mean_freight": k["total_freight"] / n}, int(cost["n_orders"].sum())))

    # ---- seller -> customer routes ---------------------------------------- #
    pairs = an.state_pair_performance(df, min_orders, alpha)
    if not pairs["reliable"].any():
        skipped.append("seller-to-customer route comparison")
    else:
        n_pairs = int(pairs["reliable"].sum())
        for direction, kind, title, phr, sev_ok in (
                ("high", "pair_high", "Routes with a reliably higher late rate", "above", True),
                ("low", "pair_low", "Routes with a reliably lower late rate", "below", False)):
            seg = an.high_delay_segments(pairs, min_orders, overall, alpha, direction)
            if seg.empty:
                continue
            top = seg.head(TOP_K)
            sev = ("alert" if sev_ok and top["rate_ratio"].iloc[0] >= ALERT_RATIO
                   else "watch" if sev_ok else "info")
            out.append(Insight(
                kind, title,
                f"Seller-state to customer-state routes whose {_lvl(alpha)} interval lies entirely {phr} the "
                f"overall {_pct(overall)} ({len(seg)} of {n_pairs:,} routes with n >= {min_orders:,} flagged): "
                + "; ".join(_seg_text(r, "state_pair", alpha=alpha) for _, r in top.iterrows())
                + ". " + _lead_note(alpha, n_pairs),
                {"segments": _records(top, ["state_pair", *SEG_COLS, "excess_vs_overall", "rate_ratio"]),
                 "n_flagged": int(len(seg)), "n_screened": n_pairs, "overall_late_rate": overall},
                int(seg["n_orders"].sum()), sev))

    # ---- same-state vs cross-state ---------------------------------------- #
    ss = an.same_state_comparison(df, min_orders, alpha).set_index("group")
    if {"same_state", "cross_state"} <= set(ss.index) and ss["reliable"].all():
        a, c = ss.loc["cross_state"], ss.loc["same_state"]
        d = sa.proportion_diff_ci(a["n_late"], a["n_orders"], c["n_late"], c["n_orders"], alpha)
        z = sa.two_proportion_ztest(a["n_late"], a["n_orders"], c["n_late"], c["n_orders"])
        out.append(Insight(
            "same_state", "Cross-state vs same-state orders",
            f"Cross-state orders: {_pct(a['late_rate'])} late (n = {int(a['n_orders']):,}), mean "
            f"{_num(a['mean_duration'])} days. Same-state orders: {_pct(c['late_rate'])} late "
            f"(n = {int(c['n_orders']):,}), mean {_num(c['mean_duration'])} days. Late-rate difference "
            f"{_pp(d['diff'])}, {_lvl(alpha)} CI {_pp(d['ci_low'])} to {_pp(d['ci_high'])}, {_p(z['p_value'])}.",
            {"cross_late_rate": float(a["late_rate"]), "same_late_rate": float(c["late_rate"]),
             "cross_n": int(a["n_orders"]), "same_n": int(c["n_orders"]), "diff": d["diff"],
             "diff_ci_low": d["ci_low"], "diff_ci_high": d["ci_high"], "p_value": z["p_value"],
             "cross_mean_duration": float(a["mean_duration"]),
             "same_mean_duration": float(c["mean_duration"])},
            int(a["n_orders"] + c["n_orders"]),
            "watch" if np.isfinite(d["ci_low"]) and d["ci_low"] > 0 else "info"))
    else:
        skipped.append("same-state vs cross-state comparison")

    # ---- weight / freight association -------------------------------------- #
    s_w = sa.spearman_test(df, "total_weight_g", "delivery_duration_days", alpha)
    s_f = sa.spearman_test(df, "total_freight", "delivery_duration_days", alpha)
    s_wf = sa.spearman_test(df, "total_weight_g", "total_freight", alpha)
    if min(s_w["n"], s_f["n"]) >= min_orders and np.isfinite(s_w["rho"]) and np.isfinite(s_f["rho"]):
        def _s(r: dict, a_: str, b_: str) -> str:
            ci = (f"approx. {_lvl(alpha)} CI {r['ci_low']:.2f} to {r['ci_high']:.2f}, "
                  if np.isfinite(r["ci_low"]) and np.isfinite(r["ci_high"]) else "")
            return (f"{a_} vs {b_}: rho = {r['rho']:.2f} ({r['strength_label']}, {r['direction']}; "
                    f"{ci}{_p(r['p_value'])}, n = {r['n']:,})")
        text = "Spearman rank correlations: " + "; ".join([
            _s(s_w, "order weight", "delivery time"), _s(s_f, "freight cost", "delivery time")])
        if np.isfinite(s_wf["rho"]):
            text += "; " + _s(s_wf, "order weight", "freight cost")
        out.append(Insight(
            "weight_freight", "Weight and freight cost vs delivery time",
            text + ". " + _assoc_summary(s_w, s_f) + " This does not show that weight or freight "
            "cost causes the difference.",
            {"weight_vs_duration": {k_: s_w[k_] for k_ in ("rho", "p_value", "n", "ci_low", "ci_high",
                                                          "strength_label")},
             "freight_vs_duration": {k_: s_f[k_] for k_ in ("rho", "p_value", "n", "ci_low", "ci_high",
                                                           "strength_label")},
             "weight_vs_freight": {k_: s_wf[k_] for k_ in ("rho", "p_value", "n", "ci_low", "ci_high",
                                                          "strength_label")}},
            int(min(s_w["n"], s_f["n"])),
            "watch" if max(abs(s_w["rho"]), abs(s_f["rho"])) >= 0.3 and (s_w["significant"] or s_f["significant"])
            else "info"))
    else:
        skipped.append("weight / freight association")

    # ---- processing time (retrospective) ------------------------------------ #
    ps = an.processing_summary(df)
    lab = an._labeled(df)  # same labelled rows as processing_summary / kpi_summary
    proc = lab["processing_days"].astype("float64") if "processing_days" in lab.columns else pd.Series(dtype=float)
    ok = proc.notna().to_numpy() if len(proc) else np.zeros(0, bool)
    if ps["n"] >= min_orders and ps["n"] > 0:
        q75 = float(np.quantile(proc[ok], 0.75))
        late = lab["is_late"].astype("float64").to_numpy()[ok]
        slow = proc.to_numpy()[ok] > q75
        n1, n2 = int(slow.sum()), int((~slow).sum())
        if n1 > 0 and n2 > 0:
            k1, k2 = int(late[slow].sum()), int(late[~slow].sum())
            d = sa.proportion_diff_ci(k1, n1, k2, n2, alpha)
            out.append(Insight(
                "processing_time", "Seller processing time and lateness (retrospective)",
                "Retrospective analysis: the carrier hand-over date is only known after the order, "
                f"so it cannot be used at purchase time. Processing (purchase to carrier hand-over) "
                f"averages {_num(ps['mean'])} days (median {_num(ps['median'])}), about "
                f"{_pct(ps['share_of_duration'], 0)} of total delivery time. Orders handed over after "
                f"{_num(q75)} days (slowest quarter) were late {_pct(d['p1'])} of the time "
                f"(n = {n1:,}) vs {_pct(d['p2'])} for the rest (n = {n2:,}); difference "
                f"{_pp(d['diff'])}, {_lvl(alpha)} CI {_pp(d['ci_low'])} to {_pp(d['ci_high'])}.",
                {"retrospective": True, "q75_processing_days": q75, "slow_n": n1, "slow_late_rate": d["p1"],
                 "rest_n": n2, "rest_late_rate": d["p2"], "diff": d["diff"],
                 "diff_ci_low": d["ci_low"], "diff_ci_high": d["ci_high"],
                 "mean_processing_days": ps["mean"], "median_processing_days": ps["median"],
                 "share_of_duration": ps["share_of_duration"]},
                n1 + n2, "watch" if np.isfinite(d["ci_low"]) and d["ci_low"] > 0 else "info"))
        else:
            skipped.append("processing-time comparison")
    else:
        skipped.append("processing-time comparison")

    if skipped:
        out.append(Insight(
            "insufficient_data", "Insufficient data for some comparisons",
            f"With {n:,} orders and a minimum of {min_orders:,} orders per segment, there is not "
            "enough data to report: " + "; ".join(skipped) + ". Widen the filters or collect more "
            "orders before drawing conclusions about these.",
            {"sections": skipped, "min_orders": min_orders, "n_orders": n}, n))
    return out


# --------------------------------------------------------------------------- #
# Recommendations
# --------------------------------------------------------------------------- #
def _names(insight: Insight, key: str, field_: str, label_fn=str, k: int = TOP_K) -> str:
    segs = insight.metrics.get(key, [])[:k]
    return ", ".join(label_fn(s[field_]) for s in segs)


def recommendations(insights: list[Insight]) -> list[str]:
    """Cautious, action-oriented suggestions derived from the insights (no root-cause claims)."""
    recs: list[str] = []
    kinds = {i.kind: i for i in insights}
    if "no_data" in kinds:
        return ["Widen the date range or clear filters; there are no delivered orders to analyse."]
    ins = kinds.get("region_high")
    if ins and ins.metrics.get("n_flagged"):
        top = ins.metrics["segments"][0]
        recs.append(
            f"Investigate delivery performance for orders to {_names(ins, 'segments', 'customer_state', _state_label)} "
            f"(e.g. {_state_label(top['customer_state'])}: {_pct(top['late_rate'])} late, n = {int(top['n_orders']):,}, "
            f"vs {_pct(ins.metrics['overall_late_rate'])} overall; unadjusted screen of "
            f"{ins.metrics.get('n_screened', 0):,} regions, so treat these as leads). Review carrier performance on "
            "these lanes and whether the delivery estimates shown at checkout are realistic.")
    ins = kinds.get("month_high")
    if ins and ins.metrics.get("n_flagged"):
        months = ", ".join(m["month"] for m in ins.metrics["months"])
        recs.append(f"Review capacity and delivery-promise planning for periods similar to {months}; late rates were "
                    "higher then, but the data cannot say why (demand peaks, holidays and other factors overlap).")
    ins = kinds.get("pair_high")
    if ins:
        recs.append(f"Review the seller-to-customer routes {_names(ins, 'segments', 'state_pair')}: their late-rate "
                    "intervals lie above the overall rate (unadjusted screen of "
                    f"{ins.metrics.get('n_screened', 0):,} routes, so treat them as leads), so check whether "
                    "estimates or hand-over practices on these lanes need adjusting.")
    ins = kinds.get("category_delay")
    if ins and ins.severity == "watch" and ins.metrics.get("flagged"):
        recs.append("Look into categories with reliably higher late rates "
                    f"({', '.join(map(str, ins.metrics['flagged'][:TOP_K]))}; unadjusted screen of "
                    f"{ins.metrics.get('n_screened', 0):,} categories, so treat these as leads) "
                    "to see whether they share sellers, routes or handling needs.")
    ins = kinds.get("category_duration")
    if ins:
        recs.append(f"Consider checking whether the delivery estimates for slower categories "
                    f"({_names(ins, 'segments', 'product_category')}) reflect their typical delivery times.")
    ins = kinds.get("same_state")
    if ins and ins.severity == "watch":
        recs.append("Review whether cross-state shipments need more conservative delivery estimates: they are late "
                    f"more often than same-state orders ({_pct(ins.metrics['cross_late_rate'])} vs "
                    f"{_pct(ins.metrics['same_late_rate'])}).")
    ins = kinds.get("processing_time")
    if ins and ins.severity == "watch":
        recs.append("Review seller hand-over times (retrospective finding): slower processing is associated with more "
                    "late orders. Monitoring it is possible after purchase, but it is not available for purchase-time "
                    "prediction.")
    ins = kinds.get("insufficient_data")
    if ins:
        recs.append("Collect more data or widen the filters before drawing conclusions about: "
                    + "; ".join(ins.metrics.get("sections", [])) + ".")
    if recs:
        recs.append("Treat all findings as associations in observational data; validate them with further data or a "
                    "controlled comparison before changing operations.")
    return recs


# --------------------------------------------------------------------------- #
# Markdown summary report
# --------------------------------------------------------------------------- #
def _norm_key(k: object) -> str:
    return "".join(ch for ch in str(k).lower() if ch.isalnum())


def _find(d: dict, candidates: list[str]):
    """Value of the first key of ``d`` whose normalised name is in ``candidates``."""
    norm = {_norm_key(k): v for k, v in d.items()}
    for c in candidates:
        if c in norm:
            return norm[c]
    return None


MAX_TEXT_CHARS = 2500  # long metrics-file text is cut at a sentence boundary beyond this


def _fmt_value(v, depth: int = 0) -> str:
    """Compact text for a metrics-file value: scalars as-is, dicts reduced to their scalar entries."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.3f}"
    if isinstance(v, dict):
        if depth == 0 and isinstance(v.get("definition"), str):
            return v["definition"]
        items = [f"{k}: {_fmt_value(x, depth + 1)}" for k, x in v.items()
                 if isinstance(x, (str, bool, int, float, np.number, type(None)))
                 or (isinstance(x, dict) and depth < 1)
                 or (isinstance(x, (list, tuple)) and not any(isinstance(y, (dict, list)) for y in x))]
        return "; ".join(items)
    if v is None:
        return "open"
    if isinstance(v, (list, tuple)):
        return " to ".join(_fmt_value(x, depth + 1) for x in v) if len(v) == 2 else ", ".join(
            _fmt_value(x, depth + 1) for x in v[:8] if not isinstance(x, (dict, list)))
    text = str(v)
    if len(text) <= MAX_TEXT_CHARS:
        return text
    cut = text[:MAX_TEXT_CHARS]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))  # keep whole sentences
    return cut[:end + 1] if end > 0 else cut.rsplit(" ", 1)[0] + " ..."


METRIC_LABELS = [
    ("Accuracy", ["accuracy"]), ("Precision", ["precision"]), ("Recall", ["recall"]),
    ("F1", ["f1", "f1score"]), ("ROC-AUC", ["rocauc", "auc"]),
    ("PR-AUC", ["prauc", "averageprecision"]), ("Brier score", ["brier", "brierscore"]),
]


def _model_section(mm: dict | None) -> list[str]:
    """Markdown lines for the optional model-performance section (tolerates missing keys)."""
    if not mm or not isinstance(mm, dict):
        return []
    lines = ["## Prediction model (purchase-time features only)", ""]
    name = _find(mm, ["selectedmodel", "modelname", "bestmodel", "selected"])
    selected_info = name if isinstance(name, dict) else {}
    if isinstance(name, dict):
        name = _find(name, ["name", "model"])
    models = _find(mm, ["models", "metrics", "permodel", "modelmetrics"])
    rows, shown = [], False
    if isinstance(models, dict):
        for mname, sub in models.items():
            if not isinstance(sub, dict):
                continue
            src = _find(sub, ["test"]) if isinstance(_find(sub, ["test"]), dict) else sub
            vals = {lab: _find(src, c) for lab, c in METRIC_LABELS}
            vals = {k: v for k, v in vals.items() if isinstance(v, (int, float, np.number))}
            if vals:
                rows.append((str(mname), vals))
    if name:
        lines.append(f"Selected model: **{name}**.")
    if rows:
        labels = [lab for lab, _ in METRIC_LABELS if any(lab in v for _, v in rows)]
        lines += ["", "| Model | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
        for mname, vals in rows:
            star = " (selected)" if name and mname == str(name) else ""
            lines.append(f"| {mname}{star} | " + " | ".join(
                f"{vals[l]:.3f}" if l in vals else "n/a" for l in labels) + " |")
        lines.append("")
        lines.append("Metrics are reported on the held-out test split when the file provides one.")
        shown = True
    split = _find(mm, ["split", "splits"])
    if isinstance(split, dict) and isinstance(_find(split, ["windows"]), dict):
        split = _find(split, ["windows"])
    sources = (("Selection rationale", _find(mm, ["selectionrationale", "rationale"])
                or _find(selected_info, ["rationale"])),
               ("Risk bands", _find(mm, ["riskbands", "riskbanddefinition", "riskthresholds"])),
               ("Leakage audit", _find(mm, ["leakageaudit", "leakageauditresult"])),
               ("Split windows", split if split is not None else _find(mm, ["splitwindows", "windows"])))
    for label, v in sources:
        if v not in (None, "", {}, []):
            lines += ["", f"- **{label}:** {_fmt_value(v)}"]
            shown = True
    if not shown and not name:
        lines.append("Model metrics were provided but contained no recognised fields.")
    lines.append("")
    return lines


def _report_alpha(insights: list[Insight]) -> float:
    """``alpha`` the insights were built with (stored on the overview insight), else the default."""
    for i in insights:
        a = i.metrics.get("alpha") if i.kind == "overview" else None
        if isinstance(a, (int, float)) and 0 < a < 1:
            return float(a)
    return DEFAULT_ALPHA


def summary_report(df: pd.DataFrame, filters_description: str, insights: list[Insight],
                   model_metrics: dict | None = None, alpha: float | None = None) -> str:
    """Markdown report: filters, KPIs, top insights (with n), recommendations, model section
    (optional) and methodology caveats.  Safe on empty frames and on ``model_metrics=None``.

    ``alpha`` sets the confidence level printed and used for the key-figures interval; ``None``
    takes it from the insights (so the report never mixes levels), else ``DEFAULT_ALPHA``."""
    alpha = _report_alpha(insights) if alpha is None else alpha
    k = an.kpi_summary(df)
    n = k["n_orders"]
    lines = ["# DeliveryIQ - Delivery Performance Summary", "",
             f"**Filters applied:** {filters_description or 'none (all delivered orders)'}", ""]
    if n:
        pdates = df["purchase_date"].dropna() if "purchase_date" in df.columns else pd.Series(dtype="datetime64[ns]")
        if len(pdates):
            lines += [f"**Purchase dates covered:** {pdates.min():%Y-%m-%d} to {pdates.max():%Y-%m-%d}", ""]
    lines += ["## Key figures", ""]
    if n == 0:
        lines += ["No delivered orders match the current filters (n = 0).", ""]
    else:
        lo, hi = an.wilson_ci(k["n_late"], n, alpha)
        lines += [
            "| Metric | Value |", "|---|---|",
            f"| Delivered orders (n) | {n:,} |",
            f"| Late rate | {_pct(k['late_rate'])} ({_lvl(alpha)} CI {_pct(lo)} to {_pct(hi)}; {k['n_late']:,} late orders) |",
            f"| On-time rate | {_pct(k['on_time_rate'])} |",
            f"| Mean / median delivery time | {_num(k['mean_duration'])} / {_num(k['median_duration'])} days |",
            f"| Mean days late (late orders only) | {_num(k['mean_late_days'])} |",
            f"| Total freight | {_num(k['total_freight'], 2)} |",
            f"| Customer states | {k['n_states']} |", ""]
    lines += ["## Key findings", ""]
    shown = [i for i in insights if i.kind != "overview"]
    if not shown:
        lines += ["No findings available.", ""]
    for i in shown:
        sev = {"alert": "ALERT", "watch": "WATCH"}.get(i.severity, "INFO")
        lines += [f"### {i.title} [{sev}]", "", i.text, "", f"*Based on n = {i.n:,} orders.*", ""]
    lines += ["## Recommendations", ""]
    recs = recommendations(insights)
    lines += [f"{j}. {r}" for j, r in enumerate(recs, 1)] if recs else ["No recommendations for the current selection."]
    lines.append("")
    lines += _model_section(model_metrics)
    lines += [
        "## Methodology and caveats", "",
        "- Analysis set: orders with status *delivered* and valid purchase, estimate and delivery timestamps.",
        "- *Late* means the delivery calendar date is after the estimated delivery date; arriving any time "
        "on the promised date counts as on time.",
        f"- Rates carry {_lvl(alpha)} Wilson confidence intervals; segments with fewer orders than the "
        "minimum are not ranked.",
        "- Many segments are screened at once, so a flagged segment is a lead to investigate, not a confirmed effect.",
        "- All findings are associations in observational data and do not show causes; statistical "
        "significance is not practical importance, especially with large samples.",
        "- Processing-time findings are retrospective (the hand-over date is unknown at purchase) and are "
        "not used for prediction.", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Regional report (CSV download)
# --------------------------------------------------------------------------- #
def regional_report(df: pd.DataFrame, min_orders: int = DEFAULT_MIN_ORDERS,
                    alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    """Per customer-state table for CSV export, sorted by late rate (descending).

    Columns: ``customer_state, state_name, n_orders, n_late, late_rate, late_ci_low,
    late_ci_high, on_time_rate, mean_duration, median_duration, mean_late_days, mean_freight,
    mean_weight_g, reliable, overall_late_rate, diff_vs_overall_pp, flag``.  Two different
    ideas are kept apart: ``reliable`` only means ``n_orders >= min_orders`` (enough orders to
    rank), while ``flag`` says whether the unadjusted ``1 - alpha`` Wilson interval excludes the
    overall rate.  ``flag`` is one of ``Insufficient data (n < min_orders)``,
    ``Above overall (interval excludes it)``, ``Below overall (interval excludes it)`` or
    ``Interval includes overall``.
    """
    perf = an.state_performance(df, min_orders, alpha=alpha)
    cols = ["customer_state", "state_name", "n_orders", "n_late", "late_rate", "late_ci_low",
            "late_ci_high", "on_time_rate", "mean_duration", "median_duration", "mean_late_days",
            "mean_freight", "mean_weight_g", "reliable", "overall_late_rate", "diff_vs_overall_pp",
            "flag"]
    if perf.empty:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    overall = an.kpi_summary(df)["late_rate"]
    perf = perf.sort_values("late_rate", ascending=False, kind="mergesort").reset_index(drop=True)
    perf["state_name"] = perf["customer_state"].map(STATE_NAMES).fillna(perf["customer_state"])
    perf["overall_late_rate"] = overall
    perf["diff_vs_overall_pp"] = (perf["late_rate"] - overall) * 100
    flag = np.select(
        [~perf["reliable"], perf["late_ci_low"] > overall, perf["late_ci_high"] < overall],
        [f"Insufficient data (n < {min_orders})", "Above overall (interval excludes it)", "Below overall (interval excludes it)"],
        default="Interval includes overall")
    perf["flag"] = flag
    return perf[cols]
