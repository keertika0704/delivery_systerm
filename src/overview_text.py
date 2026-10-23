"""Plain-text interpretation builders for the Overview page (pure pandas / numpy, no Streamlit).

Each ``*_text`` function takes tables that ``src.analytics`` already computed and returns an
:class:`Interpretation`: an ``observed`` sentence block (always built from the real numbers, with
``n``) and an optional ``action`` (a *suggested next step*, phrased as something to investigate,
never as a proven cause).  Nothing here is hardcoded: every figure in the text comes from the
arguments.  All functions are safe on empty / tiny input and never raise or divide by zero.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from . import analytics as an
from .config import DEFAULT_MIN_ORDERS, STATE_NAMES

__all__ = ["Interpretation", "status_text", "duration_text", "monthly_text", "state_text",
           "headline_insights", "teaser", "insight_label", "insight_caveat", "STATISTICAL_KINDS"]

#: Insight kinds that rest on a hypothesis test / correlation (shown as "Statistical association").
STATISTICAL_KINDS = frozenset({"region_test", "same_state", "weight_freight", "processing_time", "month_high"})
_SEVERITY_RANK = {"alert": 0, "watch": 1, "info": 2}
_HIDDEN_KINDS = frozenset({"overview", "no_data"})
#: A month counts as well populated for a first-to-last comparison at this share of the median charted month.
_WELL_POPULATED = 0.25


def _pl(n: int, noun: str) -> str:
    """'1 month' / '3 months'."""
    return f"{n:,} {noun}" if n == 1 else f"{n:,} {noun}s"


def teaser(text: str, limit: int = 230) -> str:
    """Shorten ``text`` for a card without cutting through a statistic.

    Whole sentences are kept while they fit in ``limit`` characters; if even the first sentence is too long it is cut
    at the last ``; `` (a list item boundary), else at the last space, and ends with an ellipsis.  A text that already
    fits is returned unchanged.
    """
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    sentences, start = [], 0
    for m in re.finditer(r"[.!?](?=\s+[A-Z])", text):      # a sentence ends at . ! ? followed by a capital letter
        sentences.append(text[start:m.end()])
        start = m.end() + 1
    sentences.append(text[start:])
    out = ""
    for sent in sentences:
        cand = f"{out} {sent}".strip()
        if len(cand) > limit:
            break
        out = cand
    if out:
        return out if out != text else text
    cut = text[:limit]
    depth, item_end, open_at = 0, -1, []
    for idx, ch in enumerate(cut):          # last "; " list-item boundary outside any parenthesis
        if ch == "(":
            depth += 1
            open_at.append(idx)
        elif ch == ")" and depth:
            depth -= 1
            open_at.pop()
        elif ch == ";" and depth == 0 and cut[idx + 1:idx + 2] == " ":
            item_end = idx
    if item_end > limit * 0.4:
        cut = cut[:item_end]
    else:                                    # no clean boundary: stop at a space and never leave a bracket open
        cut = cut.rsplit(" ", 1)[0]
        if open_at and open_at[0] < len(cut):
            cut = cut[:open_at[0]]
    return cut.rstrip(" ,;:(") + " ..."


@dataclass(frozen=True)
class Interpretation:
    """``observed`` = what the data show; ``action`` = a suggested next step (or None)."""

    observed: str
    action: str | None = None


# --------------------------------------------------------------------------- #
# Formatting (no streamlit here, so a tiny local copy)
# --------------------------------------------------------------------------- #
def _ok(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _pct(x, d: int = 1) -> str:
    return f"{100 * float(x):.{d}f}%" if _ok(x) else "n/a"


def _pp(x, d: int = 1) -> str:
    return f"{100 * float(x):+.{d}f} pp" if _ok(x) else "n/a"


def _days(x, d: int = 1) -> str:
    if not _ok(x):
        return "n/a"
    txt = f"{float(x):.{d}f}"
    return f"{txt} {'day' if txt == '1' else 'days'}"


def _n(x) -> str:
    return f"n = {int(x):,}" if _ok(x) else "n = n/a"


def _state(code) -> str:
    name = STATE_NAMES.get(str(code))
    return f"{code} ({name})" if name else str(code)


def _month(ym) -> str:
    try:
        return pd.Period(str(ym), "M").strftime("%b %Y")
    except (ValueError, TypeError):
        return str(ym)


# --------------------------------------------------------------------------- #
# Chart interpretations
# --------------------------------------------------------------------------- #
def status_text(k: dict, k_all: dict | None = None, filters_active: bool = False,
                min_orders: int = DEFAULT_MIN_ORDERS) -> Interpretation:
    """Interpret the on-time / late split.  ``k`` = ``analytics.kpi_summary`` of the selection."""
    n = int(k.get("n_orders", 0) or 0)
    if n == 0:
        return Interpretation("No delivered orders match the current filters.")
    n_late = int(k.get("n_late", 0))
    n_on = n - n_late
    lo, hi = an.wilson_ci(n_late, n)
    obs = (f"**{n_on:,}** of {n:,} delivered orders ({_pct(k['on_time_rate'])}) arrived on or before the "
           f"estimated delivery date; **{n_late:,}** ({_pct(k['late_rate'])}; 95% interval "
           f"{_pct(lo)} to {_pct(hi)}) arrived after it.")
    if n_late == 0:
        obs += " No order in this selection was late."
    if filters_active and k_all and int(k_all.get("n_orders", 0)) > 0 and _ok(k_all.get("late_rate")):
        diff = float(k["late_rate"]) - float(k_all["late_rate"])
        rel = "about the same as" if abs(diff) < 0.0005 else ("above" if diff > 0 else "below")
        obs += (f" Across all {int(k_all['n_orders']):,} orders the late rate is {_pct(k_all['late_rate'])}, so this "
                f"selection is {rel} the overall level" + ("" if rel.startswith("about") else f" ({_pp(diff)})") + ".")
    if n < min_orders:
        obs += f" Small sample (n = {n:,} < {min_orders:,}): treat the rates as indicative only."
    action = None
    if n_late and n >= min_orders:
        action = (f"Look at where the {n_late:,} late orders cluster (by month, state or route) before changing "
                  "anything network-wide; the charts below and the Delay Analysis page break this down.")
    return Interpretation(obs, action)


def duration_text(d: dict, min_orders: int = DEFAULT_MIN_ORDERS) -> Interpretation:
    """Interpret the delivery-time distribution.  ``d`` = ``analytics.duration_summary``."""
    n = int(d.get("n", 0) or 0)
    if n == 0 or not _ok(d.get("mean")):
        return Interpretation("No delivery times are available for the current filters.")
    obs = (f"Half of the orders arrived within **{_days(d['median'])}** of purchase and 90% within "
           f"**{_days(d['p90'])}**; the mean is {_days(d['mean'])} (n = {n:,}; fastest {_days(d['min'], 0)}, "
           f"slowest {_days(d['max'], 0)}).")
    if _ok(d.get("median")) and float(d["mean"]) > 1.05 * float(d["median"]):
        obs += (f" The mean is above the median, which is what a right-skewed distribution looks like: a minority "
                f"of slow deliveries stretches the tail (95th percentile {_days(d['p95'])}).")
    action = None
    if n >= min_orders:
        action = (f"The slowest 10% of orders (more than {_days(d['p90'])}) are the natural group to examine first "
                  "for carrier or route problems; compare them with the typical order on the Delay Analysis page.")
    else:
        obs += f" Small sample (n = {n:,} < {min_orders:,}): percentiles are indicative only."
    return Interpretation(obs, action)


def monthly_text(monthly: pd.DataFrame | None, min_orders: int = DEFAULT_MIN_ORDERS) -> Interpretation:
    """Interpret the monthly late-rate series.  ``monthly`` = ``analytics.monthly_performance`` output."""
    if monthly is None or len(monthly) == 0:
        return Interpretation("No months to show for the current filters.")
    m = monthly.copy()
    m["n_orders"] = pd.to_numeric(m["n_orders"], errors="coerce").fillna(0)
    n_months = len(m)
    rel = m[(m["n_orders"] >= min_orders) & m["late_rate"].notna()].sort_values("purchase_year_month").reset_index(drop=True)
    n_small = n_months - len(rel)
    one = n_small == 1
    tail = (f" {_pl(n_small, 'month')} with fewer than {min_orders:,} orders {'is' if one else 'are'} not charted as "
            f"{'a rate' if one else 'rates'} because {'its rate' if one else 'their rates'} would be too noisy."
            if n_small else "")
    if rel.empty:
        return Interpretation(f"No month in this selection has {min_orders:,} or more orders "
                              f"({_pl(n_months, 'month')}, {int(m['n_orders'].sum()):,} orders in total), so no reliable "
                              "monthly rate or trend can be described. Widen the date range to see one." )
    if len(rel) == 1:
        only = rel.iloc[0]
        return Interpretation(f"Only one month ({_month(only['purchase_year_month'])}) has at least {min_orders:,} orders: "
                              f"{_pct(only['late_rate'])} late ({_n(only['n_orders'])}). No trend can be described from a "
                              "single month." + tail)
    hi_row = rel.loc[rel["late_rate"].idxmax()]
    lo_row = rel.loc[rel["late_rate"].idxmin()]
    obs = (f"Across the {len(rel)} months with at least {min_orders:,} orders, the monthly late rate ranged from "
           f"**{_pct(lo_row['late_rate'])}** ({_month(lo_row['purchase_year_month'])}, {_n(lo_row['n_orders'])}) to "
           f"**{_pct(hi_row['late_rate'])}** ({_month(hi_row['purchase_year_month'])}, {_n(hi_row['n_orders'])}).")
    # first-to-last comparison only between well-populated months: the ramp-up months of a young marketplace
    # (a few hundred orders) would make an arbitrary endpoint pair.
    typical = float(rel["n_orders"].median())
    well = rel[rel["n_orders"] >= _WELL_POPULATED * typical]
    if len(well) >= 2:
        first, last = well.iloc[0], well.iloc[-1]
        obs += (f" Between the first ({_month(first['purchase_year_month'])}, {_pct(first['late_rate'])}, "
                f"{_n(first['n_orders'])}) and the last ({_month(last['purchase_year_month'])}, "
                f"{_pct(last['late_rate'])}, {_n(last['n_orders'])}) well-populated months the change was "
                f"{_pp(float(last['late_rate']) - float(first['late_rate']))} (two end points, not a trend).")
    obs += tail
    spread = float(hi_row["late_rate"]) - float(lo_row["late_rate"])
    if spread < 0.01:
        obs += " The rate was fairly stable (less than 1 percentage point between best and worst month)."
    action = None
    if spread >= 0.01:
        # compare the peak with its neighbours (up to two charted months either side), not with the whole
        # series: order volume grew a lot over the period, so the all-months median is mostly growth
        pos = int(rel.index.get_loc(hi_row.name))
        nb = pd.concat([rel.iloc[max(0, pos - 2):pos], rel.iloc[pos + 1:pos + 3]])
        ref = float(nb["n_orders"].median()) if len(nb) else 0.0
        if ref > 0:
            ratio = float(hi_row["n_orders"]) / ref
            where = f"the median of its {len(nb)} neighbouring month{'s' if len(nb) != 1 else ''}"
            vol = (f"its volume ({int(hi_row['n_orders']):,} orders) was {ratio:.1f}x {where}, so strain on "
                   "capacity is one hypothesis worth checking" if ratio >= 1.25 else
                   f"its volume ({int(hi_row['n_orders']):,} orders) was ordinary ({ratio:.1f}x {where}), so "
                   "volume alone does not obviously explain it")
            vol = f": {vol}"
        else:
            vol = ""
        action = (f"Investigate what was different in {_month(hi_row['purchase_year_month'])}{vol}. "
                  "Other candidates are carrier performance, seasonal promotions and how generous the promised "
                  "delivery dates were. This is observational, not proof of a cause.")
    return Interpretation(obs, action)


def state_text(state_perf: pd.DataFrame | None, min_orders: int = DEFAULT_MIN_ORDERS,
               overall_late_rate: float | None = None) -> Interpretation:
    """Interpret the late rate by customer state.  ``state_perf`` = ``analytics.state_performance`` output."""
    if state_perf is None or len(state_perf) == 0:
        return Interpretation("No states to show for the current filters.")
    sp = state_perf.copy()
    sp["n_orders"] = pd.to_numeric(sp["n_orders"], errors="coerce").fillna(0)
    ok = sp[(sp["n_orders"] >= min_orders) & sp["late_rate"].notna()].sort_values("late_rate", ascending=False)
    small = sp[sp["n_orders"] < min_orders]
    n_small = len(small)
    one = n_small == 1
    small_txt = (f" {_pl(n_small, 'state')} with fewer than {min_orders:,} orders ({int(small['n_orders'].sum()):,} orders "
                 f"in total) {'is' if one else 'are'} shown greyed out because {'its rate is' if one else 'their rates are'} "
                 "unreliable." if n_small else "")
    if ok.empty:
        return Interpretation(f"No state has {min_orders:,} or more orders in this selection, so state rates are "
                              "too noisy to rank. Lower the minimum-orders control or widen the filters." + small_txt)
    top, bot = ok.iloc[0], ok.iloc[-1]
    if len(ok) == 1:
        return Interpretation(f"Only {_state(top['customer_state'])} has at least {min_orders:,} orders: "
                              f"{_pct(top['late_rate'])} late ({_n(top['n_orders'])}). No comparison between "
                              "states is possible." + small_txt)
    gap = float(top["late_rate"]) - float(bot["late_rate"])
    ratio = (f", {float(top['late_rate']) / float(bot['late_rate']):.1f}x the lowest" if float(bot["late_rate"]) > 0 else "")
    top3 = "; ".join(f"{r['customer_state']} {_pct(r['late_rate'])} ({_n(r['n_orders'])})" for _, r in ok.head(3).iterrows())
    obs = (f"Among the {len(ok)} states with at least {min_orders:,} orders, the late rate ranges from "
           f"**{_pct(bot['late_rate'])}** ({_state(bot['customer_state'])}, {_n(bot['n_orders'])}) to "
           f"**{_pct(top['late_rate'])}** ({_state(top['customer_state'])}, {_n(top['n_orders'])}){ratio}, a gap of "
           f"{_pp(gap)}. Highest three: {top3}." + small_txt)
    if _ok(overall_late_rate):
        n_above = int((ok["late_rate"] > float(overall_late_rate)).sum())
        obs += (f" {n_above} of these {len(ok)} states {'has' if n_above == 1 else 'have'} a point estimate above the overall "
                f"{_pct(overall_late_rate)}.")
    action = (f"Start with {_state(top['customer_state'])}"
              + (f" and {_state(ok.iloc[1]['customer_state'])}" if len(ok) > 2 else "")
              + ": check whether the slow routes are the long seller-to-customer corridors by opening Regional "
                "Intelligence. Differences between states are associations; distance, carrier mix and seller "
                "location are all possible contributors.")
    return Interpretation(obs, action)


# --------------------------------------------------------------------------- #
# Headline findings
# --------------------------------------------------------------------------- #
def headline_insights(insights: Sequence, k: int = 3) -> list:
    """Top ``k`` insights for the headline strip: alerts first, then watch, then info (stable order).

    The ``overview`` insight (the KPI cards already show it) and ``no_data`` are skipped; an
    ``insufficient_data`` notice is shown only when nothing else is available.
    """
    cand = [i for i in insights if getattr(i, "kind", None) not in _HIDDEN_KINDS]
    real = [i for i in cand if i.kind != "insufficient_data"]
    pool = real or cand
    pool = sorted(pool, key=lambda i: _SEVERITY_RANK.get(getattr(i, "severity", "info"), 3))
    return pool[:max(0, int(k))]


def insight_label(insight) -> str:
    """'Statistical association' for test / correlation based insights, else 'Observed finding'."""
    return "Statistical association" if getattr(insight, "kind", None) in STATISTICAL_KINDS else "Observed finding"


def insight_caveat(insight) -> str:
    """Fixed one-line caution shown on every headline card (the card text itself may be clipped)."""
    if getattr(insight, "kind", None) in STATISTICAL_KINDS:
        return "Statistical association, not proof of a cause. Full text with intervals below."
    return "A lead to check, not a confirmed effect or a cause. Full text with intervals below."
