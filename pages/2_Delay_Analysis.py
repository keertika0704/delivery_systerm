"""Delay Analysis: when, where and for which kinds of orders do deliveries run late?

Six tabs (trends, delivery duration, weight & freight, product categories, processing time,
statistical tests).  Every number comes from ``src.analytics`` / ``src.statistical_analysis`` /
``src.insights``; the interpretation sentences come from ``src.delay_narrative`` (f-strings over the
computed values).  Data arrives only through ``ui.get_filtered_orders()``.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:      # lets the page run standalone (e.g. streamlit.testing AppTest)
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import analytics as an  # noqa: E402
from src import charts, config, ui  # noqa: E402
from src import delay_narrative as nar  # noqa: E402
from src import statistical_analysis as sa  # noqa: E402

PAL = config.PALETTE
ALPHA = config.DEFAULT_ALPHA
MIN_LATE_FOR_MEAN = 30     # a mean over fewer late orders is flagged "indicative only" (usual rule of thumb for a mean)

# --------------------------------------------------------------------------- #
# Page-local styling (labelled finding blocks and test cards)
# --------------------------------------------------------------------------- #
_PAGE_CSS = """
<style>
.da-note{display:flex; gap:.85rem; align-items:flex-start; background:#fff; border:1px solid var(--border);
  border-left:4px solid var(--nc); border-radius:12px; padding:.8rem 1rem; margin:.5rem 0 .75rem;
  box-shadow:var(--shadow);}
.da-note .tag{flex:0 0 auto; min-width:9.2rem; font-size:.7rem; font-weight:800; letter-spacing:.07em;
  text-transform:uppercase; color:var(--nc); padding-top:.18rem;}
.da-note .txt{color:#334155; font-size:.93rem; line-height:1.6; min-width:0;}
.da-note.observed{--nc:#2563EB;} .da-note.assoc{--nc:#7C3AED;} .da-note.action{--nc:#15803D;}
.da-note.caveat{--nc:#B45309; background:#FFFBEB; border-color:#FDE68A; border-left-color:#B45309;}
@media (max-width:640px){ .da-note{flex-direction:column; gap:.2rem;} .da-note .tag{min-width:0;} }
.da-card{background:#fff; border:1px solid var(--border); border-radius:var(--radius); padding:1rem 1.1rem .9rem;
  box-shadow:var(--shadow); margin:.2rem 0 .4rem; min-width:0;}
.da-card .hd{display:flex; justify-content:space-between; align-items:flex-start; gap:.6rem;}
.da-card .ttl{font-weight:700; color:var(--navy); font-size:.98rem; line-height:1.3;}
.da-card .kind{font-size:.68rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:var(--slate-light);}
.da-chip{display:inline-block; font-size:.72rem; font-weight:700; padding:.12rem .55rem; border-radius:999px;
  white-space:nowrap; background:#F1F5F9; color:#475569;}
.da-chip.small{background:#DBEAFE; color:#1E40AF;} .da-chip.medium{background:#BFDBFE; color:#1E3A8A;}
.da-chip.large{background:#1E3A8A; color:#fff;} .da-chip.skip{background:#F1F5F9; color:#64748B;}
.da-card .big{font-size:1.7rem; font-weight:800; color:var(--navy); letter-spacing:-.02em; margin:.45rem 0 .05rem;
  font-variant-numeric:tabular-nums;}
.da-card .big small{font-size:.8rem; font-weight:600; color:var(--slate-light); letter-spacing:0; margin-left:.3rem;}
.da-card .sub{color:var(--slate-light); font-size:.8rem; line-height:1.45;}
.da-card .stats{min-height:3.3em; color:#334155; font-size:.82rem; margin-top:.5rem; padding-top:.5rem; border-top:1px solid var(--border);
  line-height:1.6; font-variant-numeric:tabular-nums;}
</style>
"""
st.html(_PAGE_CSS)


def note(kind: str, text: str | None) -> None:
    """Labelled interpretation block: observed | assoc | action | caveat."""
    if not text:
        return
    tags = {"observed": "Observed finding", "assoc": "Statistical association",
            "action": "Suggested action", "caveat": "Data caveat"}
    body = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", ui.esc(text))
    st.html(f'<div class="da-note {kind}" role="note"><div class="tag">{tags[kind]}</div>'
            f'<div class="txt">{body}</div></div>')


def show(fig, key: str) -> None:
    st.plotly_chart(fig, width="stretch", config=charts.PLOTLY_CONFIG, key=key)


_TAB_WIDGET_KEYS = ("da_trend_gran", "da_trend_window", "da_dur_stat", "da_dur_order", "da_dur_topn", "da_wf_route", "da_wf_bins",
                    "da_wf_scatter", "da_cat_sort", "da_cat_reliable", "da_cat_unknown", "da_cat_topn", "da_cat_fmetric")


def empty_tab() -> None:
    # Streamlit forgets the value of a widget that is not drawn in a run.  The tab controls are only drawn when
    # there are orders, so re-assign them here and they survive a filter combination that returns nothing.
    for k in _TAB_WIDGET_KEYS:
        if k in st.session_state:
            st.session_state[k] = st.session_state[k]
    ui.empty_state("No orders match the current filters",
                   "Widen the date range or clear the state / category filters in the sidebar.")


def cat_label(c) -> str:
    return ui.category_label(c)


# --------------------------------------------------------------------------- #
# Cached computations (keyed on the filter signature; the frame itself is not hashed)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _trend(_df, fkey, freq, min_orders):
    return an.delay_trend(_df, freq, min_orders, ALPHA)


@st.cache_data(show_spinner=False)
def _monthly(_df, fkey, min_orders):
    m = an.monthly_performance(_df, min_orders, ALPHA)
    scr, flagged = nar.month_screen_frames(m, ALPHA)
    return m, scr, flagged


@st.cache_data(show_spinner=False)
def _segments(_df, fkey, min_orders):
    """Category / state / category-cost tables used by several tabs."""
    return (an.category_performance(_df, min_orders, ALPHA), an.state_performance(_df, min_orders, "customer_state", ALPHA),
            an.category_cost_comparison(_df, min_orders))


@st.cache_data(show_spinner=False)
def _tests(_df, fkey):
    res = sa.run_all_tests(_df, alpha=ALPHA)
    cis = {col: an.cramers_v_bootstrap_ci(r["table"], ALPHA) for col, r in res["chi_square"].items()
           if isinstance(r.get("table"), pd.DataFrame)}
    return res, cis


def tab_trends(df, fkey, min_orders):
    if df.empty:
        return empty_tab()
    ui.section_header("Late orders over time",
                      "Number and share of late deliveries by purchase date. Rates are shown only for periods with enough orders.",
                      icon="trend")
    c1, c2 = st.columns([3, 2])
    with c1:
        gran = st.radio("Time granularity", ["Month", "Week", "Quarter"], horizontal=True, key="da_trend_gran",
                        help="Group orders by the month, week or quarter of purchase.")
    with c2:
        window = st.slider("Rolling average (periods)", 1, 8, 1, key="da_trend_window",
                           help="1 = no smoothing. Larger values pool consecutive periods to show the underlying direction.")
    freq = {"Month": "M", "Week": "W", "Quarter": "Q"}[gran]
    trend = _trend(df, fkey, freq, min_orders)
    show(charts.delay_trend_chart(trend, window=window, min_orders=min_orders, freq_label=gran.lower()), "da_trend_chart")
    monthly, screen, flagged = _monthly(df, fkey, min_orders)
    note("observed", nar.trend_observed(monthly, min_orders))
    note("caveat", nar.sparse_caveat(monthly, min_orders))

    ui.section_header("Months significantly above the rest",
                      "Every month with enough orders is compared with all other months pooled; "
                      "p-values are Bonferroni-adjusted for the number of months tested.", icon="alert")
    if not screen.empty:
        show(charts.flagged_months_chart(monthly, screen, min_orders=min_orders, alpha=ALPHA), "da_flagged_months_chart")
    note("assoc", nar.flagged_text(screen, flagged, ALPHA))
    if not flagged.empty:
        t = pd.DataFrame({
            "Month": [nar.month_name(m) for m in flagged["month"]],
            "Orders (n)": flagged["n_orders"].astype(int), "Late orders": flagged["n_late"].astype(int),
            "Late rate (%)": 100 * flagged["late_rate"], "Other months (%)": 100 * flagged["rest_late_rate"],
            "Difference (pp)": 100 * flagged["diff"], "CI low (pp)": 100 * flagged["diff_ci_low"],
            "CI high (pp)": 100 * flagged["diff_ci_high"], "Adjusted p": [nar.pval(p) for p in flagged["p_bonferroni"]],
        })
        st.dataframe(t, hide_index=True, width="stretch", column_config={
            "Orders (n)": st.column_config.NumberColumn(format="%d"), "Late orders": st.column_config.NumberColumn(format="%d"),
            "Late rate (%)": st.column_config.NumberColumn(format="%.1f"),
            "Other months (%)": st.column_config.NumberColumn(format="%.1f"),
            "Difference (pp)": st.column_config.NumberColumn(format="%+.1f"),
            "CI low (pp)": st.column_config.NumberColumn(format="%+.1f", help="95% CI of the difference (Newcombe)"),
            "CI high (pp)": st.column_config.NumberColumn(format="%+.1f")})
    note("action", nar.trend_action(flagged, screen_ran=not screen.empty))


def tab_duration(df, fkey, min_orders):
    if df.empty:
        return empty_tab()
    s = an.duration_summary(df)
    ui.section_header("How long do deliveries take?",
                      "Days from purchase to the customer receiving the order (delivered orders only).", icon="clock")
    ui.kpi_row([
        dict(label="Mean", value=ui.fmt_days(s["mean"]), sublabel=ui.fmt_n(s["n"]), icon="clock", accent="blue"),
        dict(label="Median", value=ui.fmt_days(s["median"]), sublabel="Half of orders arrive sooner", icon="clock", accent="blue"),
        dict(label="90th percentile", value=ui.fmt_days(s["p90"]), sublabel=f"1 in 10 takes longer · 95th pct. {ui.fmt_days(s['p95'])}",
             icon="trend", accent="amber"),
        dict(label="Slowest order", value=ui.fmt_days(s["max"]), sublabel=f"Std. deviation {ui.fmt_days(s['std'])}", icon="alert", accent="red"),
    ])
    show(charts.duration_histogram(df), "da_dur_hist")
    note("observed", nar.duration_observed(s))

    cat_perf, state_perf, _ = _segments(df, fkey, min_orders)
    ui.section_header("Product categories", f"Only categories with at least {min_orders:,} orders are ranked (n is shown on each bar).",
                      icon="layers")
    c1, c2, c3 = st.columns(3)
    with c1:
        stat = st.radio("Statistic", ["Mean", "Median"], horizontal=True, key="da_dur_stat")
    with c2:
        order = st.radio("Show first", ["Longest", "Shortest"], horizontal=True, key="da_dur_order")
    with c3:
        top_n = st.slider("Categories shown", 5, 30, 15, key="da_dur_topn")
    metric = "median_duration" if stat == "Median" else "mean_duration"
    named = cat_perf[cat_perf["product_category"] != "unknown"] if not cat_perf.empty else cat_perf
    show(charts.category_duration_bar(named, top_n=top_n, min_orders=min_orders, metric=metric,
                                      ascending=(order == "Shortest")), "da_dur_cat_chart")
    note("observed", nar.segment_duration_text(cat_perf, "product_category", s, min_orders, metric))

    n_states = 0 if state_perf is None or state_perf.empty else len(state_perf)
    ui.section_header("Customer states",
                      f"Variation across the destination state (all {nar.plural(n_states, 'state')} with orders in this view are shown; "
                      "small samples are hatched).", icon="map-pin")
    show(charts.state_duration_bar(state_perf, top_n=max(n_states, 1), min_orders=min_orders, metric=metric,
                                   ascending=(order == "Shortest"), include_small=True, max_small=max(n_states, 1)),
         "da_dur_state_chart")
    state_text = nar.segment_duration_text(state_perf, "customer_state", s, min_orders, metric)
    note("observed", state_text)
    kw = sa.kruskal_or_mannwhitney(df, "delivery_duration_days", "customer_state", min_group_size=30, alpha=ALPHA)
    if np.isfinite(kw.get("p_value", np.nan)):
        note("assoc", f"A Kruskal-Wallis test across {kw['n_groups']} states (each with at least 30 orders) finds that delivery time "
                      f"differs between states (H = {nar.num(kw['statistic'], 0)}, p {nar.pval(kw['p_value'])}, {nar.n_(kw['n'])}). "
                      f"Effect size epsilon-squared = {nar.num(kw['effect_size'], 2)} ({kw['effect_label']}). The states are also at "
                      "different distances from the sellers, so the test describes a difference, not its cause.")
    if state_text and n_states > 1:       # the advice only makes sense when states are actually being compared
        note("action", "Compare the slowest states with the distance to the nearest sellers: where the gap is larger than distance "
                       "explains, check carrier coverage and last-mile arrangements. The Regional Intelligence page shows the "
                       "state-by-state detail.")


def tab_weight_freight(df, fkey, min_orders):
    if df.empty:
        return empty_tab()
    ui.callout("info", "These are associations, not causes",
               "Heavier parcels also tend to cost more to ship, travel further and belong to different product types, so weight "
               "and freight cost are tangled up with distance and region. Use the route-type filter to look at part of that.")
    c1, c2, c3 = st.columns(3)
    with c1:
        route = st.selectbox("Route type", ["All orders", "Same state (seller and customer)", "Different states"], key="da_wf_route",
                             help="Same state = the representative seller and the customer are in the same state.")
    with c2:
        binmode = st.radio("Grouping", ["Fixed ranges", "Equal-count groups"], horizontal=True, key="da_wf_bins",
                           help="Fixed ranges use round-number edges; equal-count groups put the same number of orders in each bin.")
    with c3:
        scatter_on = st.checkbox("Overlay a random sample of orders", value=False, key="da_wf_scatter",
                                 help="Draws up to 5,000 sampled orders (fixed seed) behind the binned means.")
    d = df
    if route.startswith("Same"):
        d = df[df["same_state"] == 1]
    elif route.startswith("Different"):
        d = df[df["same_state"] == 0]
    if d.empty:
        return ui.empty_state("No orders of this route type in the current view",
                              "Choose \"All orders\" or widen the filters in the sidebar.")
    quant = binmode.startswith("Equal")
    overall = an.kpi_summary(d)["late_rate"]
    st.caption(f"{ui.fmt_n(len(d))} orders · overall late rate {ui.fmt_pct(overall)} · "
               f"{int(d['total_weight_g'].isna().sum()):,} orders without a usable weight are left out of the weight charts.")

    rho_w_d = sa.spearman_test(d, "total_weight_g", "delivery_duration_days", ALPHA)
    rho_w_l = sa.spearman_test(d, "total_weight_g", "is_late", ALPHA)
    rho_f_d = sa.spearman_test(d, "total_freight", "delivery_duration_days", ALPHA)
    rho_f_l = sa.spearman_test(d, "freight_ratio", "is_late", ALPHA)
    rho_wf = sa.spearman_test(d, "total_weight_g", "total_freight", ALPHA)

    ui.section_header("Product weight", "Total order weight in grams, compared with delivery time and late rate.", icon="package")
    wb = an.weight_bins(d, quantile=quant, bins=6, min_orders=min_orders, alpha=ALPHA)
    sc = an.scatter_sample(d, "total_weight_g", "delivery_duration_days") if scatter_on else None
    a, b = st.columns(2)
    with a:
        show(charts.weight_vs_duration(wb, sc, min_orders=min_orders), "da_w_dur")
    with b:
        show(charts.weight_late_rate_bars(wb, overall_rate=overall, min_orders=min_orders), "da_w_late")
    note("observed", nar.binned_observed(wb, "weight", "g", min_orders, overall))
    note("assoc", nar.spearman_sentence(rho_w_d, "order weight", "delivery time", ALPHA) + " "
         + nar.spearman_sentence(rho_w_l, "order weight", "being late (0/1)", ALPHA))

    ui.section_header("Freight charges", "Shipping cost charged for the order (R$), compared with delivery time and late rate.",
                      icon="dollar")
    fb = an.freight_bins(d, quantile=quant, bins=6, min_orders=min_orders, alpha=ALPHA)
    sc2 = an.scatter_sample(d, "total_freight", "delivery_duration_days") if scatter_on else None
    a, b = st.columns(2)
    with a:
        show(charts.freight_vs_duration(fb, sc2, min_orders=min_orders), "da_f_dur")
    with b:
        show(charts.freight_late_rate_bars(fb, overall_rate=overall, min_orders=min_orders), "da_f_late")
    note("observed", nar.binned_observed(fb, "freight", "R$", min_orders, overall))
    note("assoc", nar.spearman_sentence(rho_f_d, "freight charge", "delivery time", ALPHA) + " "
         + nar.spearman_sentence(rho_f_l, "freight-to-price ratio", "being late (0/1)", ALPHA))
    note("caveat", nar.confounding_note(rho_wf, "weight and freight"))
    note("action", "Treat expensive or heavy shipments as a segment to monitor rather than a cause: check whether their delays "
                   "come from long routes (compare the same-state and cross-state views) or from carrier handling of bulky "
                   "parcels, using carrier records.")


def tab_categories(df, fkey, min_orders):
    if df.empty:
        return empty_tab()
    cat_perf, _, cost = _segments(df, fkey, min_orders)
    overall = an.kpi_summary(df)["late_rate"]
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        sort_by = st.selectbox("Sort table by", ["Late rate (highest first)", "Orders (most first)", "Mean delivery time (longest first)",
                                                 "Mean freight (highest first)", "Category name"], key="da_cat_sort")
    with c2:
        only_reliable = st.toggle(f"Only n ≥ {min_orders:,}", value=True, key="da_cat_reliable",
                                  help="Hide categories with fewer orders than the minimum set above the tabs.")
    with c3:
        incl_unknown = st.toggle("Include 'unknown'", value=False, key="da_cat_unknown",
                                 help="Orders whose product category is missing in the source data.")
    with c4:
        top_n = st.slider("Bars shown", 5, 30, 12, key="da_cat_topn")

    ui.section_header("Category comparison table",
                      "Late rate with its 95% Wilson interval, delivery time and shipping cost per category. Click a column header to sort.",
                      icon="layers")
    tbl = cat_perf.merge(cost[["product_category", "mean_freight_ratio"]], on="product_category", how="left") if not cat_perf.empty else cat_perf
    if not incl_unknown and not tbl.empty:
        tbl = tbl[tbl["product_category"] != "unknown"]
    n_hidden = int((tbl["n_orders"] < min_orders).sum()) if not tbl.empty else 0
    if only_reliable and not tbl.empty:
        tbl = tbl[tbl["n_orders"] >= min_orders]
    if tbl.empty:
        ui.empty_state(f"No category has at least {min_orders:,} orders in this view",
                       "Lower the minimum number of orders above, switch off \"Only n ≥ ...\" or widen the filters.")
    else:
        key, asc = {"Late rate (highest first)": ("late_rate", False), "Orders (most first)": ("n_orders", False),
                    "Mean delivery time (longest first)": ("mean_duration", False),
                    "Mean freight (highest first)": ("mean_freight", False), "Category name": ("product_category", True)}[sort_by]
        tbl = tbl.sort_values([key, "n_orders"], ascending=[asc, False], kind="mergesort")
        show_df = pd.DataFrame({
            "Category": tbl["product_category"].map(cat_label), "Orders (n)": tbl["n_orders"].astype(int),
            "Late orders": tbl["n_late"].astype(int), "Late rate (%)": 100 * tbl["late_rate"],
            "CI low (%)": 100 * tbl["late_ci_low"], "CI high (%)": 100 * tbl["late_ci_high"],
            "vs overall (pp)": 100 * (tbl["late_rate"] - overall),
            "Mean days": tbl["mean_duration"], "Median days": tbl["median_duration"],
            "Mean freight (R$)": tbl["mean_freight"], "Freight / price (%)": 100 * tbl["mean_freight_ratio"],
            "Reliable": tbl["n_orders"] >= min_orders,
        })
        st.dataframe(show_df, hide_index=True, width="stretch", height=min(560, 76 + 35 * len(show_df)), column_config={
            "Category": st.column_config.TextColumn(width="medium"),
            "Orders (n)": st.column_config.NumberColumn(format="%d", width="small"),
            "Late orders": st.column_config.NumberColumn(format="%d", width="small"),
            "Late rate (%)": st.column_config.NumberColumn(format="%.1f", help="Observed share of orders delivered after the estimated date"),
            "CI low (%)": st.column_config.NumberColumn(format="%.1f", help="95% Wilson interval, lower bound"),
            "CI high (%)": st.column_config.NumberColumn(format="%.1f", help="95% Wilson interval, upper bound"),
            "vs overall (pp)": st.column_config.NumberColumn(format="%+.1f", help="Late rate minus the overall late rate in this view"),
            "Mean days": st.column_config.NumberColumn(format="%.1f"), "Median days": st.column_config.NumberColumn(format="%.1f"),
            "Mean freight (R$)": st.column_config.NumberColumn(format="%.2f"),
            "Freight / price (%)": st.column_config.NumberColumn(format="%.1f", help="Mean of freight divided by item price"),
            "Reliable": st.column_config.CheckboxColumn(help=f"True when the category has at least {min_orders:,} orders"),
        })
        cap = f"{nar.plural(len(show_df), 'category', 'categories')} shown"
        if only_reliable and n_hidden:
            cap += f" · {n_hidden} with fewer than {min_orders:,} orders hidden"
        st.caption(cap + f" · overall late rate {ui.fmt_pct(overall)} ({ui.fmt_n(len(df))})")
        st.download_button("Download table (CSV)", show_df.to_csv(index=False).encode("utf-8"), "category_delay_table.csv",
                           "text/csv", key="da_cat_download", icon=":material/download:")

    obs, assoc, hi, lo = nar.category_observed(cat_perf, overall, min_orders, ALPHA)
    note("observed", obs)
    note("assoc", assoc)

    named = cat_perf[cat_perf["product_category"] != "unknown"] if not cat_perf.empty else cat_perf
    ui.section_header("Charts", "Highest values among categories with enough orders; whiskers are 95% Wilson intervals.", icon="bar-chart")
    a, b = st.columns(2)
    with a:
        show(charts.category_late_rate_bar(named, top_n=top_n, min_orders=min_orders, overall_rate=overall), "da_cat_late")
    with b:
        show(charts.category_duration_bar(named, top_n=top_n, min_orders=min_orders), "da_cat_dur")
    fmetric = st.radio("Shipping cost measure", ["Mean freight per order (R$)", "Freight as % of item price"], horizontal=True,
                       key="da_cat_fmetric")
    named_cost = cost[cost["product_category"] != "unknown"] if not cost.empty else cost
    show(charts.category_freight_bar(named_cost, top_n=top_n, min_orders=min_orders,
                                     metric="mean_freight_ratio" if fmetric.startswith("Freight as") else "mean_freight"),
         "da_cat_freight")
    note("caveat", "Category is mixed up with seller location, product weight and customer region, so a high late rate for a "
                   "category is not evidence that the category itself causes delays.")
    ranked = not cat_perf.empty and bool(((cat_perf["product_category"] != "unknown") & (cat_perf["n_orders"] >= min_orders)).any())
    note("action", nar.category_action(hi, ranked=ranked))


def tab_processing(df, fkey, min_orders):
    ui.callout("watch", "Retrospective analysis",
               "Carrier handoff time is NOT known at purchase and is not a model input. This tab describes what happened after the "
               "fact; it cannot be used to predict lateness when an order is placed.")
    if df.empty:
        return empty_tab()
    ps = an.processing_summary(df)
    overall = an.kpi_summary(df)["late_rate"]
    ui.section_header("Time from purchase to carrier handoff", "Days between the purchase and the seller handing the parcel to the carrier.",
                      icon="truck")
    ui.kpi_row([
        dict(label="Median handoff", value=ui.fmt_days(ps["median"]), sublabel=ui.fmt_n(ps["n"]), icon="truck", accent="blue"),
        dict(label="Mean handoff", value=ui.fmt_days(ps["mean"]), sublabel=f"90th percentile {ui.fmt_days(ps['p90'])}", icon="clock", accent="blue"),
        dict(label="Share of delivery time", value=ui.fmt_pct(ps["share_of_duration"], 0), sublabel="Handoff days / total days", icon="percent", accent="amber"),
        dict(label="Carrier to customer", value=ui.fmt_days(ps["mean_carrier_to_customer_days"]),
             sublabel=f"Mean, after handoff · {ui.fmt_int(ps['n_missing'])} orders without a usable handoff date left out", icon="truck", accent="slate"),
    ])
    a, b = st.columns(2)
    buckets = an.late_by_processing_bucket(df, min_orders=min_orders, alpha=ALPHA)
    with a:
        show(charts.processing_histogram(df), "da_proc_hist")
    with b:
        show(charts.late_rate_by_processing_bucket(buckets, overall_rate=overall, min_orders=min_orders), "da_proc_bucket")
    note("observed", nar.processing_observed(ps, buckets, min_orders))
    note("caveat", "The link is partly built in: time spent before handoff is time the carrier no longer has before the promised "
                   "date. It also says nothing about why some sellers are slower.")
    note("action", nar.processing_action())


_CHI_TITLES = {"customer_state": "Customer state", "product_category": "Product category",
               "same_state": "Seller and customer in the same state", "purchase_month": "Purchase month (pooled across years)"}
_SP_TITLES = {"total_weight_g~delivery_duration_days": "Order weight and delivery time",
              "total_freight~delivery_duration_days": "Freight charge and delivery time",
              "total_weight_g~total_freight": "Order weight and freight charge",
              "freight_ratio~is_late": "Freight-to-price ratio and being late",
              "total_weight_g~is_late": "Order weight and being late",
              "estimated_delivery_days~is_late": "Promised delivery days and being late"}
_GROUP_TITLES = {"delivery_duration_days~same_state": "Delivery time, same-state vs cross-state",
                 "delivery_duration_days~customer_state": "Delivery time across customer states"}


def _chip(label, few: bool = False) -> str:
    if few:       # too few orders for an effect-size label to mean anything
        return '<span class="da-chip skip">indicative only</span>'
    lab = str(label)
    cls = {"negligible": "", "small": "small", "weak": "small", "medium": "medium", "moderate": "medium"}.get(lab.lower(), "large")
    return f'<span class="da-chip {cls}">{ui.esc(lab)} effect</span>'


def _p_line(res) -> str:
    p = res.get("p_value", np.nan)
    ph = res.get("p_holm", np.nan)
    s = f"p {nar.pval(p)}"
    if np.isfinite(ph):
        s += f" · Holm-adjusted p {nar.pval(ph)}"
    return s


def _card(title: str, kind: str, big: str, sub: str, stats: str, chip: str) -> None:
    st.html(f'<div class="da-card"><div class="hd"><div><div class="kind">{ui.esc(kind)}</div><div class="ttl">{ui.esc(title)}</div></div>'
            f'{chip}</div><div class="big">{big}</div><div class="sub">{ui.esc(sub)}</div><div class="stats">{ui.esc(stats)}</div></div>')


def _unrun_card(title: str, kind: str, res, n_view: int | None = None) -> None:
    """Card for a test that could not run; ``n_view`` is the number of orders in the view (the test's own n is 0)."""
    n = n_view if n_view is not None else (res or {}).get("n", 0)
    _card(title, kind, "Not run", (res or {}).get("note") or "Too few orders or too little variation in this view.",
          f"{nar.n_(n)} orders in this view", '<span class="da-chip skip">not available</span>')


def _chi_card(col, res, ci, few: bool = False, n_view: int | None = None):
    title = _CHI_TITLES.get(col, nar.friendly(col))
    if not np.isfinite(res.get("chi2", np.nan)):
        return _unrun_card(title, "Chi-square test vs late delivery", res, n_view)
    v, lo, hi = ci if ci else (np.nan, np.nan, np.nan)
    sub = (f"Cramér's V; bias-corrected {nar.num(v, 3)}, {int(100 * (1 - ALPHA))}% bootstrap CI {nar.num(lo, 3)} to {nar.num(hi, 3)}"
           if np.isfinite(v) else "Cramér's V")
    stats = (f"chi-square = {nar.num(res['chi2'], 1)} · df = {res['dof']} · {_p_line(res)} · {nar.n_(res['n'])}")
    _card(title, "Chi-square test vs late delivery", f"V = {nar.num(res['cramers_v'], 3)}", sub, stats, _chip(res["effect_label"], few))


def _sp_card(key, res, few: bool = False, n_view: int | None = None):
    title = _SP_TITLES.get(key, nar.friendly(key.replace("~", " and ")))
    if not np.isfinite(res.get("rho", np.nan)):
        return _unrun_card(title, "Spearman rank correlation", res, n_view)
    if np.isfinite(res.get("ci_low", np.nan)):
        sub = f"{int(100 * (1 - ALPHA))}% CI {nar.num(res['ci_low'], 3)} to {nar.num(res['ci_high'], 3)} (Fisher-z, approximate)"
    else:
        sub = "No CI reported: one variable is binary (late = 1)"
    _card(title, "Spearman rank correlation", f"rho = {nar.num(res['rho'], 3)}",
          f"{sub}; {res.get('direction', '')}", f"{_p_line(res)} · {nar.n_(res['n'])}", _chip(res["strength_label"], few))


def _grp_card(key, res, few: bool = False, n_view: int | None = None):
    title = _GROUP_TITLES.get(key, nar.friendly(key.replace("~", " across ")))
    if not np.isfinite(res.get("statistic", np.nan)):
        return _unrun_card(title, "Group comparison", res, n_view)
    if res.get("test") == "kruskal":
        kind, stat, eff = "Kruskal-Wallis test", f"H = {nar.num(res['statistic'], 0)}", "epsilon-squared"
    else:
        kind, stat, eff = "Mann-Whitney U test", f"U = {nar.num(res['statistic'], 0)}", "rank-biserial correlation"
    _card(title, kind, f"{eff} = {nar.num(res['effect_size'], 3)}", f"{res.get('n_groups', '')} groups compared; no CI reported",
          f"{stat} · {_p_line(res)} · {nar.n_(res['n'])}", _chip(res["effect_label"], few))


def _reading(res, extra: pd.DataFrame | None = None, extra_title: str = "") -> None:
    with st.expander("Plain-language reading"):
        st.write(nar.friendly(sa.plain_language(res)))
        if res.get("caveat"):
            st.caption(res["caveat"])
        if extra is not None and not extra.empty:
            st.caption(extra_title)
            st.dataframe(extra, hide_index=True, width="stretch")


def tab_tests(df, fkey, min_orders):
    note("caveat", "Statistical significance is not practical importance, and neither is causation. With tens of thousands of orders, "
                   "even tiny differences are 'significant' (p < 0.05), so judge each result by its effect size and confidence interval. "
                   "All tests describe associations in historical observational data.")
    if df.empty:
        return empty_tab()
    res, cis = _tests(df, fkey)
    few = len(df) < min_orders
    n_view = len(df)
    if few:
        ui.callout("watch", "Too few orders to test reliably",
                   f"Only {ui.fmt_n(n_view)} in this view, fewer than the minimum of {min_orders:,} set above. With so few orders "
                   "p-values and effect sizes are unstable, so the results below are indicative only (effect-size labels are "
                   "not shown). Widen the filters or lower the minimum to test more reliably.")
    note("observed", nar.tests_headline(res, min_orders))
    summ = res["summary"].copy()
    if not summ.empty:
        ui.section_header("All tests at a glance", "Holm adjustment controls the chance of false positives across all tests run together.",
                          icon="percent")
        tab = pd.DataFrame({
            "Test": [nar.TEST_NAMES.get(t, t) for t in summ["test"]], "Variables": [nar.friendly(v) for v in summ["variables"]],
            "Statistic": summ["statistic"],
            "p-value": [nar.pval(p) for p in summ["p_value"]], "Holm-adjusted p": [nar.pval(p) for p in summ["p_holm"]],
            "Effect size": summ["effect_size"], "Effect": ["indicative only" if few else e for e in summ["effect_label"].fillna("not run").astype(str)], "n": summ["n"].astype(int),
            "Significant after Holm": summ["significant_holm"].astype(bool),
        })
        st.dataframe(tab, hide_index=True, width="stretch", height=38 + 35 * len(tab), column_config={
            "Test": st.column_config.TextColumn(width="medium"), "Variables": st.column_config.TextColumn(width="large"),
            "Statistic": st.column_config.NumberColumn(format="%.4g"), "Effect size": st.column_config.NumberColumn(format="%.3f"),
            "n": st.column_config.NumberColumn(format="%d"),
            "Significant after Holm": st.column_config.CheckboxColumn(help="Holm-adjusted p < 0.05")})

    ui.section_header("Is delivery status related to a category? Chi-square tests",
                      "Tests whether the late rate differs between the levels of each variable; effect size is Cramér's V "
                      "(about 0.1 small, 0.3 medium, 0.5 large for a 2x2 table).", icon="bar-chart")
    items = list(res["chi_square"].items())
    for i in range(0, len(items), 2):
        cols = st.columns(2)
        for col_ui, (name, r) in zip(cols, items[i:i + 2]):
            with col_ui:
                _chi_card(name, r, cis.get(name), few, n_view)
                ls = r.get("level_summary")
                extra = None
                if isinstance(ls, pd.DataFrame) and not ls.empty:
                    extra = ls.head(6).assign(late_rate=lambda x: (100 * x["late_rate"]).round(1)).rename(columns={
                        "level": "Level", "n": "Orders (n)", "n_late": "Late orders", "late_rate": "Late rate (%)",
                        "adj_residual": "Adjusted residual"})
                _reading(r, extra, "Levels that depart most from independence (adjusted standardized residual, absolute value > 2 is notable):")

    ui.section_header("Do two measures move together? Spearman correlations",
                      "Rank correlation between pairs of variables; |rho| below 0.1 is negligible, 0.1 to 0.3 weak, 0.3 to 0.5 moderate.",
                      icon="trend")
    items = list(res["spearman"].items())
    for i in range(0, len(items), 2):
        cols = st.columns(2)
        for col_ui, (name, r) in zip(cols, items[i:i + 2]):
            with col_ui:
                _sp_card(name, r, few, n_view)
                _reading(r)

    ui.section_header("Do delivery times differ between groups?", "Rank-based comparison of delivery time across groups.", icon="clock")
    items = list(res["group_comparison"].items())
    cols = st.columns(2)
    for col_ui, (name, r) in zip(cols, items[:2]):
        with col_ui:
            _grp_card(name, r, few, n_view)
            gs = r.get("group_summary")
            extra = None
            if isinstance(gs, pd.DataFrame) and not gs.empty:
                extra = gs.sort_values("n", ascending=False).head(8).round(2).rename(columns={
                    "group": "Group", "n": "Orders (n)", "median": "Median days", "mean": "Mean days"})
            _reading(r, extra, "Largest groups (for the same-state route, 1 = seller and customer in the same state, 0 = different states):")


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
df_all, df, flt = ui.get_filtered_orders()
ui.page_header("Delay Analysis",
               "Understand when, where and for which kinds of orders deliveries arrive after the promised date. Every figure is "
               "computed from the filtered delivered orders; findings are associations in historical 2016-2018 Olist data, not causes.",
               icon="bar-chart", eyebrow="DeliveryIQ")
ui.filter_banner(flt)
fkey = (str(flt["date_range"]), tuple(flt["states"]), tuple(flt["categories"]), len(df),
        int(df["is_late"].sum()) if len(df) else 0)

kpi_slot = st.container()            # filled after the minimum-orders control below (the flags depend on it)

c1, c2 = st.columns([2, 3])
with c1:
    min_orders = st.slider("Minimum orders for a reliable segment", 10, 1000, config.DEFAULT_MIN_ORDERS, step=10, key="da_min_orders",
                           help="Months, categories, states and bins with fewer orders are flagged as small samples and left out of "
                                "rankings and comparisons. The Statistical tests tab always uses every order in view, but marks the "
                                "results as indicative only when the view has fewer orders than this.")
with c2:
    st.caption("Applies to the trend, duration, weight, category and processing tabs. Rates are always shown next to the number of "
               "orders (n); small samples are hatched in charts and marked in tables.")

if not df.empty:
    k = an.kpi_summary(df)
    lo, hi = an.wilson_ci(k["n_late"], k["n_orders"], ALPHA)
    s = an.duration_summary(df)
    few_late = k["n_late"] < MIN_LATE_FOR_MEAN
    with kpi_slot:
        if k["n_orders"] < min_orders:
            ui.callout("watch", "Small sample",
                       f"Only {ui.fmt_n(k['n_orders'])} in this view, fewer than the minimum of {min_orders:,}. The rates and averages "
                       "below are indicative only: one order moves them by many percentage points. Widen the filters for a "
                       "reliable reading.")
        ui.kpi_row([
            dict(label="Orders analysed", value=ui.fmt_int(k["n_orders"]), sublabel=f"of {ui.fmt_int(flt['n_total'])} delivered orders",
                 icon="package", accent="blue"),
            dict(label="Late rate", value=ui.fmt_pct(k["late_rate"]), sublabel=f"{ui.fmt_int(k['n_late'])} late · 95% CI "
                 f"{ui.fmt_pct(lo)} to {ui.fmt_pct(hi)}", icon="alert", accent="red"),
            dict(label="Mean delivery time", value=ui.fmt_days(k["mean_duration"]),
                 sublabel=f"Median {ui.fmt_days(k['median_duration'])} · 90th pct. {ui.fmt_days(s['p90'])}", icon="clock", accent="blue"),
            dict(label="Mean days late", value=ui.fmt_days(k["mean_late_days"]),
                 sublabel=f"Among {ui.fmt_int(k['n_late'])} late orders only" + (" · indicative only" if few_late else ""),
                 icon="calendar", accent="slate"),
        ])

tabs = st.tabs([":material/trending_up: Delay trends", ":material/schedule: Delivery duration", ":material/scale: Weight & freight",
                ":material/category: Product categories", ":material/local_shipping: Processing time",
                ":material/functions: Statistical tests"])
for tab, fn in zip(tabs, (tab_trends, tab_duration, tab_weight_freight, tab_categories, tab_processing, tab_tests)):
    with tab:
        fn(df, fkey, min_orders)
