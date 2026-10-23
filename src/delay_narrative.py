"""Interpretation text for the Delay Analysis page (pure Python / pandas, no Streamlit).

Every sentence is an f-string over numbers computed by ``src.analytics`` /
``src.statistical_analysis`` / ``src.insights``; nothing is hard-coded.  The functions return plain
strings (``**bold**`` allowed) or ``None`` when there is nothing meaningful to say, so the page can
show the text next to the matching chart.  The language keeps three things apart:

* *observed* -- what the filtered orders show (counts, rates, intervals, always with ``n``);
* *statistical association* -- what a formal test says (never causal);
* *suggested action* -- what an operations manager could investigate next.
"""
from __future__ import annotations

import re
from typing import Mapping

import numpy as np
import pandas as pd

from . import analytics as an
from . import insights as ins
from .config import DEFAULT_ALPHA

UNKNOWN_CATEGORY = "unknown"


# --------------------------------------------------------------------------- #
# Tiny formatters (kept here so this module stays Streamlit-free)
# --------------------------------------------------------------------------- #
def _ok(x) -> bool:
    try:
        return x is not None and bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def pct(x, d: int = 1) -> str:
    return f"{100 * float(x):.{d}f}%" if _ok(x) else "n/a"


def pp(x, d: int = 1) -> str:
    return f"{100 * float(x):+.{d}f} pp" if _ok(x) else "n/a"


def num(x, d: int = 1) -> str:
    return f"{float(x):,.{d}f}" if _ok(x) else "n/a"


def n_(x) -> str:
    return f"n = {int(round(float(x))):,}" if _ok(x) else "n = n/a"


def pval(p) -> str:
    if not _ok(p):
        return "n/a"
    p = float(p)
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def level(alpha: float) -> str:
    return f"{int(round(100 * (1 - alpha)))}%"


def cat_name(c) -> str:
    s = str(c).replace("_", " ").strip()
    return s[:1].upper() + s[1:] if s else s


def month_name(ym) -> str:
    try:
        return pd.Timestamp(str(ym) + "-01").strftime("%b %Y")
    except (ValueError, TypeError):
        return str(ym)


def stat_text(test, x) -> str:
    """Test statistic as readable text for the 'all tests' table: thousands separators, no significant-figure clipping.

    ``%.4g`` turned U = 1,709,892,180 into 710000000 and H = 24,168 into 24170, so large statistics are shown in full
    (rank statistics U and H as whole numbers, chi-square with one decimal, Spearman rho with three).
    """
    if not _ok(x):
        return "n/a"
    t = str(test).lower()
    d = 3 if t == "spearman" else 1 if "chi" in t else 0
    return f"{float(x):,.{d}f}"


_MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                "November", "December"]
_ROUTE_LABELS = {"0": "Different states", "1": "Same state"}


def level_label(col, level) -> str:
    """Plain label for one level of a chi-square / group variable ('1' of same_state -> 'Same state', '3' of purchase_month -> 'March')."""
    s = str(level)
    if col == "same_state":
        return _ROUTE_LABELS.get(s, s)
    if col == "purchase_month":
        try:
            return _MONTH_NAMES[int(float(s)) - 1]
        except (ValueError, IndexError):
            return s
    if col == "product_category":
        return cat_name(s)
    return s


def relabel_reading(text: str, col, top_level) -> str:
    """Replace the quoted raw level in plain_language's 'departs most from independence is 'X'' sentence by its label."""
    raw, lab = f"is '{top_level}' (", f"is '{level_label(col, top_level)}' ("
    return str(text).replace(raw, lab, 1)


def plural(n, singular: str, plural_form: str | None = None) -> str:
    """``'1 month'`` / ``'3 months'`` (``plural_form`` for irregular nouns such as category / categories)."""
    k = int(n)
    return f"{k:,} {singular if k == 1 else (plural_form or singular + 's')}"


def and_list(items) -> str:
    """``'2017'``, ``'2017 and 2018'``, ``'2016, 2017 and 2018'``."""
    items = [str(i) for i in items]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


#: Raw column names -> plain words (used for anything that reaches the screen).
VARIABLE_NAMES = {
    "estimated_delivery_days": "promised delivery days", "delivery_duration_days": "delivery time",
    "product_category": "product category", "customer_state": "customer state", "total_weight_g": "order weight",
    "total_freight": "freight charge", "freight_ratio": "freight-to-price ratio", "purchase_month": "purchase month",
    "same_state": "same-state route", "is_late": "late delivery",
}
TEST_NAMES = {"chi-square": "Chi-square test", "spearman": "Spearman correlation", "mann_whitney": "Mann-Whitney U test",
              "kruskal": "Kruskal-Wallis test", "kruskal_or_mannwhitney": "Kruskal-Wallis / Mann-Whitney U test"}
_VAR_RE = re.compile("|".join(sorted(VARIABLE_NAMES, key=len, reverse=True)))


def friendly(text) -> str:
    """Replace raw column names in ``text`` by plain words (``'customer_state x is_late'`` -> ``'customer state x late delivery'``)."""
    return _VAR_RE.sub(lambda m: VARIABLE_NAMES[m.group(0)], str(text))


def _join(items: list[str], limit: int = 4) -> str:
    items = list(items)
    if len(items) > limit:
        return ", ".join(items[:limit]) + f" and {len(items) - limit} more"
    return ", ".join(items)


# --------------------------------------------------------------------------- #
# Tab 1: trends
# --------------------------------------------------------------------------- #
def trend_observed(monthly: pd.DataFrame, min_orders: int) -> str | None:
    """Range of monthly late rates over the months with at least ``min_orders`` orders."""
    if monthly is None or monthly.empty:
        return None
    n_all, k_all = int(monthly["n_orders"].sum()), int(monthly["n_late"].sum())
    rel = monthly[monthly["reliable"]]
    if rel.empty:
        return (f"No purchase month has {min_orders:,} or more orders in this view ({n_(n_all)} orders in "
                f"{plural(len(monthly), 'month')}, pooled late rate {pct(k_all / n_all if n_all else np.nan)}), so month-to-month "
                "rates are too noisy to compare. Widen the filters or lower the minimum.")
    lo = rel.sort_values(["late_rate", "n_orders"], ascending=[True, False], kind="mergesort").iloc[0]
    hi = rel.sort_values(["late_rate", "n_orders"], ascending=[False, False], kind="mergesort").iloc[0]
    last = rel.iloc[-1]
    pooled = k_all / n_all if n_all else float("nan")
    side = "above" if last["late_rate"] > pooled else "below" if last["late_rate"] < pooled else "equal to"
    s = (f"Across {plural(len(rel), 'month')} with at least {min_orders:,} orders ({month_name(rel.iloc[0]['purchase_year_month'])} to "
         f"{month_name(last['purchase_year_month'])}) the monthly late rate ranged from **{pct(lo['late_rate'])}** in "
         f"{month_name(lo['purchase_year_month'])} ({n_(lo['n_orders'])}) to **{pct(hi['late_rate'])}** in "
         f"{month_name(hi['purchase_year_month'])} ({n_(hi['n_orders'])})")
    if _ok(lo["late_rate"]) and lo["late_rate"] > 0:
        s += f", a {hi['late_rate'] / lo['late_rate']:.1f}-fold spread"
    s += (f". The latest of these months, {month_name(last['purchase_year_month'])}, was {pct(last['late_rate'])} "
          f"({n_(last['n_orders'])}), {side} the pooled rate of {pct(pooled)} ({k_all:,} late of {n_all:,} orders).")
    return s


def sparse_caveat(monthly: pd.DataFrame, min_orders: int) -> str | None:
    """Warning about months with fewer than ``min_orders`` orders (typically the earliest period)."""
    if monthly is None or monthly.empty:
        return None
    sp = monthly[~monthly["reliable"]]
    if sp.empty:
        return None
    labels = [f"{month_name(r['purchase_year_month'])} ({n_(r['n_orders'])})" for _, r in sp.iterrows()]
    years = sorted({str(m)[:4] for m in sp["purchase_year_month"]})
    return (f"{len(sp)} month(s) have fewer than {min_orders:,} orders: {_join(labels, 6)} "
            f"(together {int(sp['n_orders'].sum()):,} orders, all in {and_list(years)}). One order changes such a rate by many "
            "percentage points, so these months are drawn hollow or hatched, left out of the rate lines and not tested. "
            "Do not read a trend into them.")


def month_screen_frames(monthly: pd.DataFrame, alpha: float = DEFAULT_ALPHA) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(screen, flagged)``: every reliable month vs the rest, and the ones significantly higher (Bonferroni)."""
    cols = ["month", "n_orders", "n_late", "late_rate", "rest_n", "rest_late_rate", "diff", "diff_ci_low",
            "diff_ci_high", "p_value", "p_bonferroni"]
    if monthly is None or monthly.empty or int(monthly["reliable"].sum()) < 2:
        empty = pd.DataFrame(columns=cols)
        return empty, empty
    scr = ins.month_screen(monthly, alpha)
    flagged = scr[(scr["diff"] > 0) & (scr["p_bonferroni"] < alpha)].sort_values(
        "diff", ascending=False, kind="mergesort")
    return scr, flagged


def flagged_text(screen: pd.DataFrame, flagged: pd.DataFrame, alpha: float = DEFAULT_ALPHA) -> str:
    """Statistical-association text for the month screen."""
    if screen is None or screen.empty:
        return ("The month-by-month significance screen needs at least two months with enough orders; "
                "this view does not have them.")
    t = len(screen)
    base = (f"Each of the {t} reliable months was compared with all other months pooled (two-proportion z-test, "
            f"Bonferroni-adjusted over {t} months, {level(alpha)} confidence).")
    if flagged.empty:
        return base + " **No month has a late rate significantly above the others.**"
    parts = [f"{month_name(r['month'])}: {pct(r['late_rate'])} late ({n_(r['n_orders'])}) vs {pct(r['rest_late_rate'])} "
             f"in other months ({n_(r['rest_n'])}); difference {pp(r['diff'])} ({level(alpha)} CI {pp(r['diff_ci_low'])} to "
             f"{pp(r['diff_ci_high'])}, adjusted p {pval(r['p_bonferroni'])})" for _, r in flagged.head(4).iterrows()]
    more = f" ({len(flagged) - 4} more in the table below)" if len(flagged) > 4 else ""
    return (base + f" **{len(flagged)} month(s) are significantly higher**{more}: " + "; ".join(parts) +
            ". This shows these months differ; it does not show why.")


def trend_action(flagged: pd.DataFrame, screen_ran: bool = True) -> str:
    """Suggested next step for an operations manager (hypotheses to check, not conclusions).

    ``screen_ran`` is False when the month screen could not run (fewer than two months with enough orders):
    then nothing can be said about problem months, and the text must not claim that none exist.
    """
    if not screen_ran:
        return ("This view is too small to test for problem months, so no conclusion about specific periods is possible. "
                "Widen the date range or clear some filters, then re-run the screen.")
    if flagged is None or flagged.empty:
        return ("No month stands out statistically, so there is no period-specific problem to chase in this view. "
                "Keep an eye on the weekly trend and re-run the screen when the filters change.")
    months = _join([month_name(m) for m in flagged["month"]], 4)
    return (f"Look at what was different in {months}: order volume versus carrier capacity, the delivery promise shown at "
            "checkout (estimated delivery days), seller hand-over times, and any regional disruption. The data here "
            "cannot separate these explanations, so compare them against operational records.")


# --------------------------------------------------------------------------- #
# Tab 2: duration
# --------------------------------------------------------------------------- #
def duration_observed(summary: Mapping) -> str | None:
    if not summary or not summary.get("n"):
        return None
    mean, med = summary.get("mean"), summary.get("median")
    s = (f"Half of the orders arrived within **{num(med)} days** of purchase (median), while the mean is "
         f"{num(mean)} days ({n_(summary['n'])}).")
    if _ok(mean) and _ok(med) and mean > med:
        s += (f" The mean sits {num(mean - med)} days above the median because a minority of very slow orders stretches "
              f"the right tail (skewness {num(summary.get('skew'), 1)}): 10% of orders took {num(summary.get('p90'))} days or "
              f"more, 5% took {num(summary.get('p95'))} days or more, and the slowest took {num(summary.get('max'))} days.")
    return s


def _ranked(perf: pd.DataFrame, key: str, metric: str, min_orders: int, exclude_unknown: bool) -> pd.DataFrame:
    if perf is None or perf.empty or metric not in perf.columns:
        return pd.DataFrame()
    d = perf[perf["n_orders"] >= min_orders]
    if exclude_unknown and key == "product_category":
        d = d[d[key] != UNKNOWN_CATEGORY]
    return d[d[metric].notna()].sort_values([metric, "n_orders"], ascending=[False, False], kind="mergesort")


def segment_duration_text(perf: pd.DataFrame, key: str, summary: Mapping, min_orders: int,
                          metric: str = "mean_duration") -> str | None:
    """Spread of mean / median delivery time across states or categories with enough orders."""
    word = "median" if metric == "median_duration" else "mean"
    noun = "categories" if key == "product_category" else "states"
    d = _ranked(perf, key, metric, min_orders, True)
    if d.empty:
        return (f"No {'category' if key == 'product_category' else 'state'} reaches {min_orders:,} orders in this view, "
                f"so {noun} are not compared. Lower the minimum or widen the filters.")
    if len(d) < 2:        # a spread needs at least two segments
        return None
    hi, lo = d.iloc[0], d.iloc[-1]
    label = (lambda v: cat_name(v)) if key == "product_category" else (lambda v: str(v))
    s = (f"Among {len(d)} {noun} with at least {min_orders:,} orders, {word} delivery time runs from "
         f"**{num(lo[metric])} days** ({label(lo[key])}, {n_(lo['n_orders'])}) to **{num(hi[metric])} days** "
         f"({label(hi[key])}, {n_(hi['n_orders'])})")
    if _ok(lo[metric]) and lo[metric] > 0:
        s += f", a factor of {hi[metric] / lo[metric]:.1f}"
    ref = summary.get("median" if metric == "median_duration" else "mean") if summary else np.nan
    if _ok(ref):
        s += f"; across all orders in view the {word} is {num(ref)} days"
    return s + "."


# --------------------------------------------------------------------------- #
# Tab 3: weight & freight
# --------------------------------------------------------------------------- #
def binned_observed(bins: pd.DataFrame, what: str, unit: str, min_orders: int, overall_rate: float) -> str | None:
    """Late rate and delivery time in the lowest vs highest bin of ``what`` ('weight' / 'freight')."""
    if bins is None or bins.empty:
        return None
    rel = bins[bins["n_orders"] >= min_orders]
    small = len(bins) - len(rel)
    if len(rel) < 2:
        return (f"Fewer than two {what} groups have at least {min_orders:,} orders in this view, so no comparison is shown. "
                "Lower the minimum or widen the filters.")
    first, last = rel.iloc[0], rel.iloc[-1]
    top = rel.sort_values(["late_rate", "n_orders"], ascending=[False, False], kind="mergesort").iloc[0]
    lab = (lambda b: f"{b} {unit}".strip())
    low_word, high_word = ("lightest", "heaviest") if what == "weight" else ("cheapest", "most expensive")
    s = (f"The {low_word} group ({lab(first['bin'])}, {n_(first['n_orders'])}) has a late rate of **{pct(first['late_rate'])}** "
         f"and a mean delivery time of {num(first['mean_duration'])} days; the {high_word} group ({lab(last['bin'])}, "
         f"{n_(last['n_orders'])}) has **{pct(last['late_rate'])}** and {num(last['mean_duration'])} days. "
         f"The highest late rate is in {lab(top['bin'])} ({pct(top['late_rate'])}, {n_(top['n_orders'])}) "
         f"versus {pct(overall_rate)} overall.")
    if small:
        s += f" {small} group(s) with fewer than {min_orders:,} orders are shown hatched and not interpreted."
    return s


def spearman_sentence(res: Mapping, x_label: str, y_label: str, alpha: float = DEFAULT_ALPHA) -> str:
    """One-sentence statement of a Spearman result (CI only when the test reports one)."""
    if not res or not _ok(res.get("rho")):
        note = (res or {}).get("note") or "too few usable orders"
        return f"Spearman correlation of {x_label} with {y_label} could not be computed ({note})."
    ci = (f", {level(alpha)} CI {num(res['ci_low'], 2)} to {num(res['ci_high'], 2)}"
          if _ok(res.get("ci_low")) and _ok(res.get("ci_high")) else "")
    return (f"Spearman rank correlation of {x_label} with {y_label}: rho = **{num(res['rho'], 2)}**{ci} "
            f"({n_(res.get('n'))}; {res.get('strength_label', '')}, {res.get('direction', '')}).")


def confounding_note(rho_wf: Mapping | None, what_other: str) -> str:
    """Why weight and freight cannot be read as separate causes."""
    if rho_wf and _ok(rho_wf.get("rho")):
        link = (f"In this view weight and freight cost are themselves correlated (Spearman rho = {num(rho_wf['rho'], 2)}, "
                f"{n_(rho_wf.get('n'))}), ")
    else:
        link = "Weight and freight cost tend to move together, "
    return (link + f"and both also depend on how far the parcel travels and on the product type, so the {what_other} pattern "
            "is an association that mixes several factors. The route-type filter above (same-state vs cross-state) is a "
            "first way to look at the distance effect, not a full adjustment.")


# --------------------------------------------------------------------------- #
# Tab 4: categories
# --------------------------------------------------------------------------- #
def category_observed(perf: pd.DataFrame, overall_rate: float, min_orders: int,
                      alpha: float = DEFAULT_ALPHA) -> tuple[str | None, str | None, pd.DataFrame, pd.DataFrame]:
    """``(observed_text, association_text, flagged_high, flagged_low)`` for the category ranking."""
    empty = pd.DataFrame()
    if perf is None or perf.empty:
        return None, None, empty, empty
    p = perf[perf["product_category"] != UNKNOWN_CATEGORY]
    rel = p[p["n_orders"] >= min_orders]
    if rel.empty:
        return (f"No named category has {min_orders:,} or more orders in this view ({plural(len(p), 'category', 'categories')} shown in the table "
                "are all small samples), so none is ranked. Lower the minimum or widen the filters.", None, empty, empty)
    if len(rel) == 1:     # nothing to compare: a "range" from a category to itself would be degenerate prose
        only = rel.iloc[0]
        return (f"Only one category ({cat_name(only['product_category'])}) has at least {min_orders:,} orders in this view "
                f"(late rate {pct(only['late_rate'])}, {n_(only['n_orders'])}, against {pct(overall_rate)} overall), so "
                "categories cannot be compared. Widen the filters or lower the minimum to compare categories.", None, empty, empty)
    top = rel.sort_values(["late_rate", "n_orders"], ascending=[False, False], kind="mergesort").iloc[0]
    bot = rel.sort_values(["late_rate", "n_orders"], ascending=[True, False], kind="mergesort").iloc[0]
    longest = rel.sort_values("mean_duration", ascending=False, kind="mergesort").iloc[0]
    costly = rel.sort_values("mean_freight", ascending=False, kind="mergesort").iloc[0]
    obs = (f"Among {plural(len(rel), 'category', 'categories')} with at least {min_orders:,} orders, the late rate ranges from "
           f"**{pct(bot['late_rate'])}** ({cat_name(bot['product_category'])}, {n_(bot['n_orders'])}) to "
           f"**{pct(top['late_rate'])}** ({cat_name(top['product_category'])}, {n_(top['n_orders'])}), against {pct(overall_rate)} "
           f"overall. The longest mean delivery time is {num(longest['mean_duration'])} days "
           f"({cat_name(longest['product_category'])}, {n_(longest['n_orders'])}); the highest mean freight is "
           f"R$ {num(costly['mean_freight'], 2)} per order ({cat_name(costly['product_category'])}, {n_(costly['n_orders'])}).")
    hi = an.high_delay_segments(rel, min_orders, overall_rate, alpha, "high")
    lo = an.high_delay_segments(rel, min_orders, overall_rate, alpha, "low")
    # every flagged category is counted: the first four are named, the rest are summarised as "and N more"
    names = lambda d: _join([f"{cat_name(r['product_category'])} ({pct(r['late_rate'])}, {n_(r['n_orders'])})"
                             for _, r in d.iterrows()], 4)
    chance = len(rel) * alpha / 2
    ncat = plural(len(rel), "category", "categories")
    if hi.empty:
        assoc = (f"{'The category' if len(rel) == 1 else f'None of the {ncat}'} does not have a {level(alpha)} Wilson interval "
                 f"entirely above the overall rate of {pct(overall_rate)}, so no category is reliably worse than average.")
    else:
        assoc = (f"{len(hi)} of {ncat} {'has' if len(rel) == 1 else 'have'} a {level(alpha)} Wilson interval entirely above the "
                 f"overall rate: {names(hi)}. The intervals are not adjusted for screening {ncat}; even if no category truly "
                 f"differed, about {chance:.1f} would be expected to be flagged high by chance, so treat borderline ones as leads.")
    if not lo.empty:
        assoc += f" Reliably below average ({len(lo)}): {names(lo)}."
    return obs, assoc, hi, lo


def category_action(hi: pd.DataFrame, ranked: bool = True, n_compared: int | None = None) -> str:
    """Suggested step for the category tab; ``ranked`` is False when no category had enough orders to be compared.

    ``n_compared`` (optional) is the number of named categories with enough orders; exactly one means there is nothing to compare.
    """
    if n_compared == 1:
        return ("Only one category in this view, so categories cannot be compared and no category-level action follows. "
                "Widen the filters to compare categories.")
    if not ranked:
        return ("This view has no category with enough orders to compare, so no category-level conclusion is possible. "
                "Widen the filters or lower the minimum number of orders.")
    if hi is None or hi.empty:
        return ("No category is reliably above average, so category-specific action is not supported by this view. "
                "Check regional and seasonal patterns first.")
    names = _join([cat_name(c) for c in hi["product_category"]], 4)
    return (f"For {names}, check which sellers and routes supply these orders, how long hand-over to the carrier takes and whether "
            "the delivery promise shown at checkout is realistic for them. Category effects here are confounded with seller "
            "location, weight and region, so confirm with a like-for-like comparison before changing anything.")


# --------------------------------------------------------------------------- #
# Tab 5: processing time (retrospective)
# --------------------------------------------------------------------------- #
def processing_observed(psum: Mapping, buckets: pd.DataFrame, min_orders: int) -> str | None:
    """Retrospective description of hand-over time and how late rates vary with it."""
    if not psum or not psum.get("n"):
        return None
    s = (f"Orders reached the carrier a median of **{num(psum['median'])} days** after purchase (mean {num(psum['mean'])}; 90% within "
         f"{num(psum['p90'])} days; {n_(psum['n'])}). "
         f"{int(psum.get('n_missing', 0)):,} orders have no usable hand-over date and are left out. ")
    if _ok(psum.get("share_of_duration")):
        s += (f"Hand-over accounts for {pct(psum['share_of_duration'], 0)} of the total delivery time, and the carrier leg "
              f"averages {num(psum.get('mean_carrier_to_customer_days'))} days. ")
    rel = buckets[buckets["n_orders"] >= min_orders] if buckets is not None and not buckets.empty else pd.DataFrame()
    if len(rel) >= 2:
        a, b = rel.iloc[0], rel.iloc[-1]
        mono = bool((rel["late_rate"].diff().dropna() >= 0).all())
        s += (f"The late rate is **{pct(a['late_rate'])}** when hand-over took {a['bin']} days ({n_(a['n_orders'])}) and "
              f"**{pct(b['late_rate'])}** when it took {b['bin']} days ({n_(b['n_orders'])}); it "
              f"{'rises at every step' if mono else 'does not rise at every step'} across the {len(rel)} groups with at least "
              f"{min_orders:,} orders.")
    return s.strip()


def processing_action() -> str:
    return ("Hand-over time is something sellers control, so it is a candidate operational metric (for example a seller "
            "hand-over service level and an alert when it is exceeded), not a purchase-time predictor. Part of the link to late "
            "delivery is arithmetic: time spent before hand-over is time not available to the carrier within the promised date.")


# --------------------------------------------------------------------------- #
# Tab 6: tests
# --------------------------------------------------------------------------- #
def tests_headline(results: Mapping, min_orders: int | None = None) -> str | None:
    """Summary of the whole test battery: how many are significant vs how many have a non-negligible effect.

    ``min_orders``: when the view has fewer orders than this, the results are described as unstable and the
    effect-size labels are not listed as findings.
    """
    s = results.get("summary") if results else None
    if s is None or s.empty:
        return None
    n_view = results.get("n_orders")
    few = bool(min_orders) and _ok(n_view) and float(n_view) < min_orders
    ran = s[s["p_value"].notna()]
    if ran.empty:
        return "None of the tests could be run on this view (too few orders or too little variation)."
    sig = ran[ran["significant_holm"].astype(bool)]
    labels = ran["effect_label"].astype(str).str.lower()
    notable = ran[~labels.isin(["negligible", "nan", ""]) & ran["significant_holm"].astype(bool)]
    names = _join([f"{friendly(r['variables'])} ({r['effect_label']})" for _, r in notable.iterrows()], 5)
    few_note = (f" This view has fewer orders than the minimum of {min_orders:,}, so these results are unstable and "
                "indicative only; widen the filters before reading anything into them.") if few else ""
    n_tests = plural(len(ran), "test")
    if sig.empty:
        return (f"None of the {n_tests} that could be run is statistically significant after "
                f"Holm adjustment ({n_(n_view)}); absence of significance is not proof of no effect, especially with few "
                "orders." + few_note)
    if few:
        return (f"{len(sig)} of {n_tests} {'is' if len(sig) == 1 else 'are'} statistically significant after Holm adjustment "
                f"({n_(n_view)})." + few_note)
    s = (f"{len(sig)} of {n_tests} {'is' if len(sig) == 1 else 'are'} statistically significant after Holm adjustment "
         f"({n_(n_view)}), but only {len(notable)} of them {'has' if len(notable) == 1 else 'have'} an effect size labelled "
         "small or larger")
    s += (f": {names}. " if len(notable) else ". ")
    if _ok(n_view) and float(n_view) >= 10 * (min_orders or 100):
        return s + ("With this many orders, tiny and practically irrelevant differences reach p < 0.05, so read the effect "
                    "size, not the p-value alone.")
    return s + "Read the effect size, not the p-value alone."
