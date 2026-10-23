"""Regional Intelligence: geographic differences in shipment performance.

Data comes ONLY from ``ui.get_filtered_orders()`` and every statistic is computed by
``src.analytics`` / ``src.statistical_analysis`` / ``src.insights``; this page only lays the
results out and words them.  Sections: map + regional table (+ CSV export), state comparison,
high-delay region screening and seller -> customer routes.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:      # lets the page run standalone (e.g. streamlit.testing AppTest)
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import analytics as an  # noqa: E402
from src import charts, config, insights, ui  # noqa: E402
from src import statistical_analysis as sa  # noqa: E402

ALPHA = config.DEFAULT_ALPHA
CI_LEVEL = f"{round(100 * (1 - ALPHA))}%"
MAP_METRICS = {"Late rate": "late_rate", "Mean delivery time": "mean_duration", "Order volume": "n_orders"}
ROUTE_METRICS = {"Late rate": "late_rate", "Order volume": "n_orders", "Mean delivery time": "mean_duration"}
HEAT_METRICS = {"Late rate": "late_rate", "Mean delivery time": "mean_duration", "Order volume": "n_orders"}


# --------------------------------------------------------------------------- #
# Cached computations (keyed on the filter state + threshold, not on the 96k-row frame)
# --------------------------------------------------------------------------- #
def _data_key(flt: dict, n_rows: int) -> tuple:
    return (tuple(str(d) for d in flt["date_range"]), tuple(flt["states"]), tuple(flt["categories"]),
            int(n_rows), ui._mtime(config.ORDERS_DELIVERED_PATH))


@st.cache_data(show_spinner=False)
def _regional_bundle(_df: pd.DataFrame, key: tuple, min_orders: int) -> dict:
    """State table, regional report, overall KPIs, high-delay screen and Holm check for one view."""
    kpi = an.kpi_summary(_df)
    perf = an.state_performance(_df, min_orders, alpha=ALPHA)
    report = insights.regional_report(_df, min_orders, alpha=ALPHA)
    high = an.high_delay_segments(perf, min_orders, kpi["late_rate"], alpha=ALPHA)
    rel = perf[perf["reliable"]].reset_index(drop=True)
    holm = pd.DataFrame(columns=["customer_state", "p_value", "p_holm"])
    if len(rel):
        k_tot, n_tot = int(kpi["n_late"]), int(kpi["n_orders"])
        pvals = [sa.two_proportion_ztest(int(r.n_late), int(r.n_orders), k_tot - int(r.n_late),
                                         n_tot - int(r.n_orders))["p_value"] for r in rel.itertuples()]
        holm = pd.DataFrame({"customer_state": rel["customer_state"], "p_value": pvals,
                             "p_holm": sa.holm_adjust(pvals)})
    return {"kpi": kpi, "perf": perf, "report": report, "high": high, "holm": holm,
            "n_screened": int(len(rel))}


@st.cache_data(show_spinner=False)
def _route_bundle(_df: pd.DataFrame, key: tuple, min_orders: int) -> dict:
    """Seller -> customer pair table and same/cross-state split for one view."""
    return {"pairs": an.state_pair_performance(_df, min_orders, alpha=ALPHA),
            "same": an.same_state_comparison(_df, min_orders, alpha=ALPHA),
            "multi_seller_share": float((pd.to_numeric(_df["n_sellers"], errors="coerce") > 1).mean())
            if len(_df) else float("nan")}


@st.cache_data(show_spinner=False)
def _heat_matrices(_df: pd.DataFrame, key: tuple, min_orders: int, top_n: int) -> dict:
    """Seller state x customer state matrices (``top_n`` = 0 keeps every state)."""
    return an.state_matrix(_df, min_orders, top_n=top_n or None)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _plot(fig, key: str) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, config=charts.PLOTLY_CONFIG, key=key)


def _p_text(p) -> str:
    """Readable p-value: 'p < 0.001' for tiny values, three significant digits otherwise."""
    try:
        v = float(p)
    except (TypeError, ValueError):
        return "p undefined"
    if not np.isfinite(v):
        return "p undefined"
    return "p < 0.001" if v < 0.001 else f"p = {v:.3g}"


def _plural(n: int, one: str, many: str | None = None) -> str:
    """'1 state' / '3 states' (``many`` defaults to ``one + 's'``)."""
    return f"{int(n):,} " + (one if int(n) == 1 else (many or one + "s"))


# Widget values that should survive a visit to another page (Streamlit drops the state of widgets that are
# not rendered during a run).  ``_sticky`` is called right before the widget is created.
def _sticky(key: str, valid=None) -> None:
    keep = "_keep_" + key
    if key not in st.session_state and keep in st.session_state:
        v = st.session_state[keep]
        if valid is None or valid(v):
            st.session_state[key] = v
    if key in st.session_state:
        st.session_state[keep] = st.session_state[key]


def _na_dash(frame: pd.DataFrame, col: str = "mean_late_days"):
    """(frame, column config) for the average-delay column.  It stays numeric (sortable) unless some state has no
    late orders; then the missing values are shown as a dash (st.dataframe would print 'None') and the column is text."""
    help_ = "Average delay beyond the estimated date, among late orders only"
    if frame[col].isna().any():
        frame = frame.assign(**{col: frame[col].map(lambda v: ui.DASH if pd.isna(v) else f"{v:.1f}")})
        return frame, st.column_config.TextColumn("Avg days late", help=help_)
    return frame, st.column_config.NumberColumn("Avg days late", format="%.1f", help=help_)


def _height(n_rows: int, cap: int = 1100) -> int:
    """Pixel height that shows ``n_rows`` rows plus the header without an inner scrollbar (up to ``cap``)."""
    return int(min(36 * (n_rows + 1) + 4, cap))


def _rate_ci(rate, lo, hi) -> str:
    return f"{ui.fmt_pct(rate)} ({ui.fmt_pct(lo)} to {ui.fmt_pct(hi)})"


def _codes(frame: pd.DataFrame, limit: int = 6, with_n: bool = True) -> str:
    parts = [f"{r.customer_state} (n = {int(r.n_orders):,})" if with_n else str(r.customer_state)
             for r in frame.itertuples()]
    more = f" and {len(parts) - limit} more" if len(parts) > limit else ""
    return ", ".join(parts[:limit]) + more


def _table_frame(report: pd.DataFrame) -> pd.DataFrame:
    """Regional report rows ordered reliable-first, then by late rate (insufficient data last)."""
    if report.empty:
        return report
    return (report.sort_values(["reliable", "late_rate", "n_orders"], ascending=[False, False, False],
                               kind="mergesort").reset_index(drop=True))


def _short_flag(flag: str) -> str:
    """Short on-screen form of the report's ``flag`` (the CSV keeps the full wording)."""
    f = str(flag)
    if f.startswith("Above overall"):
        return "Above overall"
    if f.startswith("Below overall"):
        return "Below overall"
    if f.startswith("Interval includes"):
        return "Overlaps overall"
    if f.startswith("Insufficient"):
        return "Insufficient data"
    return f


def _filter_report(report: pd.DataFrame, query: str, hide_small: bool) -> pd.DataFrame:
    out = report
    q = (query or "").strip().lower()
    if q and len(out):
        out = out[out["customer_state"].str.lower().str.contains(q, regex=False)
                  | out["state_name"].str.lower().str.contains(q, regex=False)]
    if hide_small and len(out):
        out = out[out["reliable"]]
    return out.reset_index(drop=True)


def _default_compare(perf: pd.DataFrame, n_top: int = 3, n_worst: int = 2) -> list[str]:
    """Top-volume states plus the highest-late-rate states that meet the minimum sample."""
    if perf.empty:
        return []
    top = perf.sort_values("n_orders", ascending=False, kind="mergesort")["customer_state"].head(n_top).tolist()
    worst = perf[perf["reliable"]].sort_values("late_rate", ascending=False,
                                               kind="mergesort")["customer_state"].head(n_worst).tolist()
    return list(dict.fromkeys(top + worst))


# --------------------------------------------------------------------------- #
# Generated interpretations (every number below is computed from the data in view)
# --------------------------------------------------------------------------- #
def interpret_map(perf: pd.DataFrame, kpi: dict, high: pd.DataFrame, min_orders: int, metric: str,
                  pairs: pd.DataFrame | None) -> list[tuple[str, str, str]]:
    """(callout kind, label, text) triples next to the map: observed / association / sample / action."""
    rel = perf[perf["reliable"]]
    small = perf[~perf["reliable"]]
    overall = kpi["late_rate"]
    out: list[tuple[str, str, str]] = []
    if rel.empty:
        out.append(("watch", "Sample size",
                    f"No state has at least {min_orders:,} delivered orders in this view, so no state is "
                    f"ranked. Lower the minimum sample or widen the filters."))
        return out

    hi = rel.sort_values("late_rate", ascending=False, kind="mergesort").iloc[0]
    lo = rel.sort_values("late_rate", ascending=True, kind="mergesort").iloc[0]
    one = len(rel) == 1
    if metric == "n_orders":
        tot = float(perf["n_orders"].sum())
        top = perf.sort_values("n_orders", ascending=False, kind="mergesort")
        top3 = top.head(3)
        obs = (f"**{ui.state_label(top.iloc[0]['customer_state'])}** is the largest destination with "
               f"{ui.fmt_int(top.iloc[0]['n_orders'])} orders ({ui.fmt_pct(top.iloc[0]['n_orders'] / tot)} of the "
               f"{ui.fmt_int(tot)} orders in view)"
               + (f"; the top {len(top3)} states together account for {ui.fmt_pct(top3['n_orders'].sum() / tot)}. "
                  "Volume is very uneven, so rates for small states rest on far fewer orders."
                  if len(top) > 1 else "."))
    elif metric == "mean_duration" and one:
        obs = (f"Only **{ui.state_label(hi['customer_state'])}** has at least {min_orders:,} orders in this view "
               f"({ui.fmt_n(hi['n_orders'])}), so no states are compared: its mean delivery time is "
               f"{ui.fmt_days(hi['mean_duration'])} against {ui.fmt_days(kpi['mean_duration'])} overall.")
    elif metric == "mean_duration":
        longest = rel.sort_values("mean_duration", ascending=False, kind="mergesort").iloc[0]
        shortest = rel.sort_values("mean_duration", ascending=True, kind="mergesort").iloc[0]
        obs = (f"Among states with at least {min_orders:,} orders, mean delivery time is longest in "
               f"**{ui.state_label(longest['customer_state'])}** at {ui.fmt_days(longest['mean_duration'])} "
               f"({ui.fmt_n(longest['n_orders'])}) and shortest in **{ui.state_label(shortest['customer_state'])}** at "
               f"{ui.fmt_days(shortest['mean_duration'])} ({ui.fmt_n(shortest['n_orders'])}); the overall mean is "
               f"{ui.fmt_days(kpi['mean_duration'])}. Longer distances alone can explain slower delivery, so this is not "
               f"by itself a performance problem.")
    elif one:
        obs = (f"Only **{ui.state_label(hi['customer_state'])}** has at least {min_orders:,} delivered orders in this "
               f"view, so no states are ranked or compared: its late rate is {ui.fmt_pct(hi['late_rate'])} "
               f"({ui.fmt_int(hi['n_late'])} late of {ui.fmt_int(hi['n_orders'])} orders, {ui.fmt_n(hi['n_orders'])}) "
               f"against {ui.fmt_pct(overall)} for all orders in view ({ui.fmt_pp(hi['late_rate'] - overall)}).")
    else:
        ratio = hi["late_rate"] / overall if overall > 0 else float("nan")
        obs = (f"Among the {len(rel)} states with at least {min_orders:,} delivered orders, "
               f"**{ui.state_label(hi['customer_state'])}** has the highest observed late rate: "
               f"{ui.fmt_pct(hi['late_rate'])} ({ui.fmt_int(hi['n_late'])} late of {ui.fmt_int(hi['n_orders'])} orders, "
               f"{ui.fmt_n(hi['n_orders'])}), {ui.fmt_pp(hi['late_rate'] - overall)} versus the overall "
               f"{ui.fmt_pct(overall)}" + (f" ({ratio:.1f}x the overall rate)" if np.isfinite(ratio) else "") +
               f". The lowest is **{ui.state_label(lo['customer_state'])}** at {ui.fmt_pct(lo['late_rate'])} "
               f"({ui.fmt_n(lo['n_orders'])}).")
    out.append(("info", "Observed", obs))

    n_high = len(high)
    holds = _interval_text(hi)
    if n_high:
        names = ", ".join(high["customer_state"].head(6))
        more = f" and {n_high - 6} more" if n_high > 6 else ""
        assoc = (f"{n_high} of {_plural(len(rel), 'state')} {'has' if n_high == 1 else 'have'} a {CI_LEVEL} Wilson "
                 f"interval entirely above the overall rate ({names}{more}). For the highest one, {holds}. Intervals "
                 f"are unadjusted for the {_plural(len(rel), 'state')} screened, so a few flags can arise by chance "
                 f"(see the High-delay regions tab). This is an association in historical data, not a cause.")
    else:
        assoc = (f"No state's {CI_LEVEL} Wilson interval lies entirely above the overall late rate "
                 f"({ui.fmt_pct(overall)}). For the highest observed state, {holds}. Differences of this size are "
                 f"compatible with chance variation.")
    out.append(("watch" if n_high else "info", "Statistical association", assoc))

    if len(small):
        out.append(("watch", "Sample size",
                    f"{_plural(len(small), 'state')} below the {min_orders:,}-order minimum "
                    f"({_codes(small.sort_values('n_orders', ascending=False))}) {'is' if len(small) == 1 else 'are'} "
                    f"shown grey as insufficient data and never ranked."))
    else:
        out.append(("success", "Sample size", f"Every state in view has at least {min_orders:,} orders."))

    act = (f"Suggested next step: for {'this state' if one else 'the highest-rate state'}, look at which seller "
           "states, routes and product categories contribute most of its late orders, and compare its delivery "
           "times with the promised delivery windows. ")
    if pairs is not None and len(pairs):
        into = pairs[(pairs["customer_state"] == hi["customer_state"]) & pairs["reliable"]]
        if len(into):
            r = into.sort_values("late_rate", ascending=False, kind="mergesort").iloc[0]
            act += (f"In {hi['customer_state']}, the route with the highest late rate (n at least {min_orders:,}) is "
                    f"{r['state_pair']} at {ui.fmt_pct(r['late_rate'])} ({ui.fmt_n(r['n_orders'])}). ")
    act += ("This data cannot show why orders are late, and it describes historical orders, not current "
            "logistics performance.")
    out.append(("action", "Suggested action", act))
    return out


def _lookup_tone(h: pd.Series, overall: float) -> str:
    """Delta colour of the route look-up: only a route with enough orders whose interval excludes the overall rate
    is coloured; small or overlapping results stay neutral."""
    if not h["reliable"]:
        return "neutral"
    if h["late_ci_low"] > overall:
        return "bad"
    return "good" if h["late_ci_high"] < overall else "neutral"


def _interval_text(row: pd.Series) -> str:
    """Wording of one state's interval relative to the overall rate (uses the report's own columns)."""
    return (f"{ui.state_label(row['customer_state'])}'s {CI_LEVEL} Wilson interval is "
            f"{ui.fmt_pct(row['late_ci_low'])} to {ui.fmt_pct(row['late_ci_high'])}")


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
ui.inject_css()
df_all, df, flt = ui.get_filtered_orders()
ui.page_header(
    "Regional Intelligence",
    "Where do deliveries run late or slow? Compare customer states and seller-to-customer routes, with sample "
    "sizes and uncertainty shown beside every rate.",
    icon="map", eyebrow="Geography")
_period = pd.to_datetime(df_all["purchase_date"], errors="coerce")
st.caption(f"Historical Olist orders, {_period.min():%b %Y} to {_period.max():%b %Y}; not a measure of current carrier "
           "or logistics performance. Differences between states are associations, not causes.")
ui.filter_banner(flt)
ui.stop_if_empty(df)

with st.container(border=True):
    c1, c2 = st.columns([3, 2], vertical_alignment="center")
    with c1:
        _sticky("reg_min_orders")
        min_orders = int(st.slider(
            "Minimum orders for a state or route to be ranked", min_value=10, max_value=1000,
            value=config.DEFAULT_MIN_ORDERS, step=10, key="reg_min_orders",
            help="States / routes with fewer delivered orders are labelled 'insufficient data' and never "
                 "flagged as high-delay."))
    with c2:
        st.caption(f"Default {config.DEFAULT_MIN_ORDERS}. Rates from small samples swing widely, so below this "
                   "size a segment is shown but never ranked or flagged. Rates use the filtered orders only.")
# A fixed placeholder keeps the element tree stable, so the selected tab survives a sidebar-filter change.
_state_note = st.empty()
if flt["states"]:
    _state_note.caption(f"The sidebar customer-state filter is on ({', '.join(flt['states'])}), so only those states "
                        "appear below and 'overall' means the overall rate of the filtered orders.")

key = _data_key(flt, len(df))
reg = _regional_bundle(df, key, min_orders)
kpi, perf, report, high, holm = reg["kpi"], reg["perf"], reg["report"], reg["high"], reg["holm"]
rb = _route_bundle(df, key, min_orders)
pairs_all = rb["pairs"]
rel = perf[perf["reliable"]]

top_row = rel.sort_values("late_rate", ascending=False, kind="mergesort").iloc[0] if len(rel) else None
only_one = len(rel) == 1
ui.kpi_row([
    dict(label="Customer states in view", value=ui.fmt_int(len(perf)),
         sublabel=f"{len(rel)} with at least {min_orders:,} orders ({ui.fmt_n(kpi['n_orders'])} in total)",
         icon="map-pin", accent="blue"),
    dict(label="Overall late rate", value=ui.fmt_pct(kpi["late_rate"]),
         sublabel=f"{ui.fmt_int(kpi['n_late'])} late of {ui.fmt_int(kpi['n_orders'])} orders"
         + (f" · small sample (under {min_orders:,})" if kpi["n_orders"] < min_orders else ""),
         icon="clock", accent="slate"),
    dict(label="Only ranked state" if only_one else "Highest observed late rate",
         value=ui.fmt_pct(top_row["late_rate"]) if top_row is not None else ui.DASH,
         sublabel=(f"{ui.state_label(top_row['customer_state'])} · {ui.fmt_n(top_row['n_orders'])}"
                   + (" · the only state with enough orders" if only_one else "")
                   if top_row is not None else f"no state has {min_orders:,}+ orders"),
         delta=(ui.fmt_pp(top_row["late_rate"] - kpi["late_rate"]) + " vs overall")
         if top_row is not None and not only_one else None,
         delta_tone="bad", icon="alert",
         accent="slate" if top_row is None else "blue" if only_one else "red"),
    dict(label="High-delay segments", value=ui.fmt_int(len(high)) if reg["n_screened"] else ui.DASH,
         sublabel=(f"of {_plural(reg['n_screened'], 'state')} screened (interval above overall, n at least "
                   f"{min_orders:,})" if reg["n_screened"] else f"no state has {min_orders:,}+ orders to screen"),
         icon="target", accent="slate" if not reg["n_screened"] else "amber" if len(high) else "green"),
])

tab_map, tab_cmp, tab_high, tab_routes = st.tabs(
    ["Map & performance table", "Compare states", "High-delay regions", "Seller → customer routes"])

# ----------------------------------------------------------------------- #
# Tab 1: choropleth + interpretation + regional table + export
# ----------------------------------------------------------------------- #
with tab_map:
    ui.section_header("Brazil state map", "Customer state of the delivery address. Grey states have too few orders "
                      "to rank; pale grey states have none in the current view.", icon="map")
    _sticky("reg_map_metric")
    metric_label = st.radio("Map metric", list(MAP_METRICS), horizontal=True, key="reg_map_metric")
    metric = MAP_METRICS[metric_label]
    with st.container(key="stack-md-map"):      # below ~1100 px the commentary goes under the map, not into a 200 px column
        map_col, txt_col = st.columns([3, 2], gap="large")
        with map_col:
            fig_map = charts.brazil_choropleth(perf, metric, min_orders=min_orders)
            fig_map.update_layout(height=620)
            _plot(fig_map, "reg_map_chart")
        with txt_col:
            for kind, label, text in interpret_map(perf, kpi, high, min_orders, metric, pairs_all):
                ui.callout(kind, label, text)
            st.caption("Labels: Observed = counted in the data; Statistical association = interval based, unadjusted; "
                       "Suggested action = a lead to investigate. No model predictions are shown on this page.")

    ui.section_header("Regional performance table",
                      "One row per customer state. Click a column header to sort. 'Assessment' compares each "
                      f"state's {CI_LEVEL} Wilson interval with the overall late rate.", icon="bar-chart")
    f1, f2, f3 = st.columns([2, 1, 1], vertical_alignment="bottom")
    _sticky("reg_table_search")
    _sticky("reg_table_hide_small")
    query = f1.text_input("Search state (code or name)", key="reg_table_search", placeholder="e.g. RJ or Rio")
    hide_small = f2.checkbox("Hide insufficient data", key="reg_table_hide_small", value=False)
    shown = _filter_report(_table_frame(report), query, hide_small)
    if shown.empty:
        ui.empty_state("No state matches this search", "Clear the search box or untick 'Hide insufficient data'.")
    else:
        view = shown[["customer_state", "state_name", "n_orders", "on_time_rate", "late_rate", "late_ci_low",
                      "late_ci_high", "mean_duration", "mean_late_days", "diff_vs_overall_pp", "flag"]].copy()
        view["flag"] = view["flag"].map(_short_flag)
        view.insert(5, "late_ci", [f"{ui.fmt_pct(a)} to {ui.fmt_pct(b)}" for a, b in
                                   zip(view.pop("late_ci_low"), view.pop("late_ci_high"))])
        view, late_days_cfg = _na_dash(view)
        st.dataframe(
            view, hide_index=True, width="stretch", key="reg_table", height=_height(len(view)),
            column_config={
                "customer_state": st.column_config.TextColumn("State", width="small"),
                "state_name": st.column_config.TextColumn("Name", width="medium"),
                "n_orders": st.column_config.NumberColumn("Orders (n)", format="localized"),
                "on_time_rate": st.column_config.NumberColumn("On-time rate", format="percent"),
                "late_rate": st.column_config.NumberColumn("Late rate", format="percent"),
                "late_ci": st.column_config.TextColumn(f"Late rate {CI_LEVEL} CI",
                                                       help=f"{CI_LEVEL} Wilson interval for the late rate"),
                "mean_duration": st.column_config.NumberColumn("Mean (days)", format="%.1f", width="small",
                                                                help="Mean delivery time in days"),
                "mean_late_days": late_days_cfg,
                "diff_vs_overall_pp": st.column_config.NumberColumn("vs overall (pp)", format="%+.1f"),
                "flag": st.column_config.TextColumn(
                    "Assessment", width="small",
                    help=f"'Above' / 'Below' = the state's {CI_LEVEL} Wilson interval lies entirely above / below the "
                         "overall late rate; 'Overlaps' = the interval includes it. Unadjusted for the number of "
                         "states compared."),
            })
        n_small = int((~shown["reliable"]).sum())
        st.caption(f"{len(shown)} of {_plural(len(report), 'state')} shown. Overall late rate in this view: "
                   f"{ui.fmt_pct(kpi['late_rate'])} ({ui.fmt_n(kpi['n_orders'])}). "
                   + (f"{n_small} state{'s' if n_small != 1 else ''} below {min_orders:,} orders are marked "
                      "'Insufficient data' and sorted last. " if n_small else "")
                   + "Rates and intervals are descriptive, not adjusted for the number of states compared.")
    csv_bytes = shown.to_csv(index=False).encode("utf-8")
    st.download_button("Download this table as CSV", data=csv_bytes, key="reg_download",
                       file_name="deliveryiq_regional_performance.csv", mime="text/csv",
                       icon=":material/download:", disabled=shown.empty,
                       help="Rows shown above for the current filters and minimum-sample setting, "
                            "with all report columns (counts, intervals, flag). No personal identifiers.")

# ----------------------------------------------------------------------- #
# Tab 2: side-by-side comparison
# ----------------------------------------------------------------------- #
with tab_cmp:
    ui.section_header("Compare customer states", "Pick states to see volume, late rate (with interval) and delivery "
                      "time side by side. Default: the busiest states plus the highest late rates that meet the "
                      "minimum sample.", icon="layers")
    opts = sorted(perf["customer_state"].tolist())
    sig = tuple(opts)
    _sticky("reg_compare_states", valid=lambda v: isinstance(v, list))
    if st.session_state.get("_reg_cmp_sig") != sig or "reg_compare_states" not in st.session_state:
        st.session_state["reg_compare_states"] = _default_compare(perf)
        st.session_state["_reg_cmp_sig"] = sig
    else:
        st.session_state["reg_compare_states"] = [s for s in st.session_state["reg_compare_states"] if s in opts]
    st.session_state["_keep_reg_compare_states"] = list(st.session_state["reg_compare_states"])
    chosen = st.multiselect("States to compare", opts, key="reg_compare_states", format_func=ui.state_label,
                            placeholder="Choose one or more states")
    if not chosen:
        ui.empty_state("Select at least one state to compare", "Use the box above to add customer states.",
                       icon="map-pin")
    else:
        _plot(charts.state_comparison_chart(perf, chosen, min_orders=min_orders, overall_rate=kpi["late_rate"],
                                            title="Selected states side by side"), "reg_cmp_chart")
        sel = perf[perf["customer_state"].isin(chosen)].copy()
        sel["_o"] = sel["customer_state"].map({s: i for i, s in enumerate(chosen)})
        sel = sel.sort_values("_o")
        sel["vs_overall_pp"] = (sel["late_rate"] - kpi["late_rate"]) * 100
        sel["assessment"] = np.select(
            [~sel["reliable"], sel["late_ci_low"] > kpi["late_rate"], sel["late_ci_high"] < kpi["late_rate"]],
            [f"Insufficient data (n < {min_orders})", "Above overall", "Below overall"],
            default="Overlaps overall")
        sel.insert(0, "name", sel["customer_state"].map(lambda c: config.STATE_NAMES.get(c, c)))
        cmp_view = sel[["customer_state", "name", "n_orders", "late_rate", "late_ci_low", "late_ci_high",
                        "mean_duration", "median_duration", "mean_late_days", "vs_overall_pp",
                        "assessment"]].copy()
        cmp_view.insert(4, "late_ci", [f"{ui.fmt_pct(a)} to {ui.fmt_pct(b)}" for a, b in
                                       zip(cmp_view.pop("late_ci_low"), cmp_view.pop("late_ci_high"))])
        cmp_view, cmp_late_days_cfg = _na_dash(cmp_view)
        st.dataframe(
            cmp_view, hide_index=True, width="stretch", key="reg_cmp_table", height=_height(len(cmp_view)),
            column_config={
                "customer_state": st.column_config.TextColumn("State", width="small"),
                "name": st.column_config.TextColumn("Name", width="medium"),
                "n_orders": st.column_config.NumberColumn("Orders (n)", format="localized"),
                "late_rate": st.column_config.NumberColumn("Late rate", format="percent"),
                "late_ci": st.column_config.TextColumn(f"Late rate {CI_LEVEL} CI", width="medium"),
                "mean_duration": st.column_config.NumberColumn("Mean (days)", format="%.1f", width="small"),
                "median_duration": st.column_config.NumberColumn("Median (days)", format="%.1f"),
                "mean_late_days": cmp_late_days_cfg,
                "vs_overall_pp": st.column_config.NumberColumn("vs overall (pp)", format="%+.1f"),
                "assessment": st.column_config.TextColumn(
                    "Assessment", width="medium",
                    help=f"Compares the state's {CI_LEVEL} Wilson interval with the overall late rate: 'Above' / "
                         "'Below' = the interval lies entirely above / below it; 'Overlaps' = the interval "
                         "includes it."),
            })
        ok = sel[sel["reliable"]]
        if len(ok) >= 2:
            a = ok.sort_values("late_rate", ascending=False, kind="mergesort").iloc[0]
            b = ok.sort_values("late_rate", ascending=True, kind="mergesort").iloc[0]
            diff = sa.proportion_diff_ci(a["n_late"], a["n_orders"], b["n_late"], b["n_orders"], ALPHA)
            zt = sa.two_proportion_ztest(a["n_late"], a["n_orders"], b["n_late"], b["n_orders"])
            overlap = not (a["late_ci_low"] > b["late_ci_high"])
            ui.callout("info", "Observed",
                       f"Among the selected states with at least {min_orders:,} orders, the late rate ranges from "
                       f"{ui.fmt_pct(b['late_rate'])} in {ui.state_label(b['customer_state'])} "
                       f"({ui.fmt_n(b['n_orders'])}) to {ui.fmt_pct(a['late_rate'])} in "
                       f"{ui.state_label(a['customer_state'])} ({ui.fmt_n(a['n_orders'])}), a gap of "
                       f"{ui.fmt_pp(a['late_rate'] - b['late_rate'])}. Mean delivery time ranges from "
                       f"{ui.fmt_days(ok['mean_duration'].min())} to {ui.fmt_days(ok['mean_duration'].max())}.")
            ui.callout("watch", "Statistical association",
                       f"Highest vs lowest of the selection: difference {ui.fmt_pp(diff['diff'])} "
                       f"({CI_LEVEL} interval {ui.fmt_pp(diff['ci_low'])} to {ui.fmt_pp(diff['ci_high'])}); "
                       f"two-proportion z-test {_p_text(zt['p_value'])} (unadjusted). The two states' own intervals "
                       f"{'overlap' if overlap else 'do not overlap'}. Picking the extremes of a selection inflates "
                       "apparent differences, and states differ in distance, seller mix and products, so this is "
                       "not evidence of a cause.")
        elif len(sel) and len(ok) < len(sel):
            ui.callout("watch", "Sample size", f"Selected states below {min_orders:,} orders are greyed out and not "
                       "compared statistically. Select at least two states that meet the minimum.")
        elif len(sel) == 1:
            ui.callout("info", "Observed", "Select a second state to compare rates and see the gap.")

# ----------------------------------------------------------------------- #
# Tab 3: high-delay region identification
# ----------------------------------------------------------------------- #
with tab_high:
    ui.section_header("High-delay region screening",
                      f"A state is labelled a high-delay segment only if it has at least {min_orders:,} orders AND "
                      f"the lower end of its {CI_LEVEL} Wilson interval is above the overall late rate "
                      f"({ui.fmt_pct(kpi['late_rate'])}).", icon="target")
    if reg["n_screened"] >= 2:
        ui.callout("watch", "Read with care: many comparisons",
                   f"{reg['n_screened']} states are screened at once with unadjusted {CI_LEVEL} intervals. Even if all "
                   f"states were truly equal, about {reg['n_screened'] * ALPHA / 2:.1f} would be expected to look "
                   "'high' by chance alone. The Holm column below shows which flags survive a correction for this "
                   "(state vs all other orders in view). A flag is a lead to investigate, not a proven cause.")
    else:
        only = ("No state meets" if reg["n_screened"] == 0 else "Only 1 state meets")
        ui.callout("watch", "Read with care",
                   f"{only} the minimum sample in this view, so there is little to compare. "
                   "A flag is a lead to investigate, not a proven cause.")
    if reg["n_screened"] == 0:
        ui.empty_state("No state meets the minimum sample", f"Lower the minimum below {min_orders:,} orders or "
                       "widen the filters to screen states.", icon="target")
    else:
        if high.empty:
            ui.callout("success", "No high-delay segments found",
                       (f"None of the {reg['n_screened']} screened states has" if reg["n_screened"] != 1
                        else "The one screened state does not have")
                       + f" an interval entirely above the overall late rate of {ui.fmt_pct(kpi['late_rate'])} "
                       f"(overall {ui.fmt_n(kpi['n_orders'])}).")
        else:
            hv = high.merge(holm, on="customer_state", how="left")
            hv.insert(1, "name", hv["customer_state"].map(lambda c: config.STATE_NAMES.get(c, c)))
            hv["holds"] = np.where(hv["p_holm"] < ALPHA, "Yes", "No")
            hv["excess_pp"] = hv["excess_vs_overall"] * 100
            hv["p_holm_txt"] = [_p_text(v).replace("p ", "").replace("= ", "") for v in hv["p_holm"]]
            hv.insert(5, "late_ci", [f"{ui.fmt_pct(a)} to {ui.fmt_pct(b)}" for a, b in
                                     zip(hv.pop("late_ci_low"), hv.pop("late_ci_high"))])
            st.dataframe(
                hv[["customer_state", "name", "n_orders", "late_rate", "late_ci", "excess_pp", "rate_ratio",
                    "p_holm_txt", "holds"]],
                hide_index=True, width="stretch", key="reg_high_table", height=_height(len(hv)),
                column_config={
                    "customer_state": st.column_config.TextColumn("State", width="small"),
                    "name": st.column_config.TextColumn("Name", width="medium"),
                    "n_orders": st.column_config.NumberColumn("Orders (n)", format="localized"),
                    "late_rate": st.column_config.NumberColumn("Late rate", format="percent"),
                    "late_ci": st.column_config.TextColumn(f"Late rate {CI_LEVEL} CI"),
                    "excess_pp": st.column_config.NumberColumn("Excess vs overall (pp)", format="%+.1f",
                                                               help="Late rate minus overall late rate, in percentage points"),
                    "rate_ratio": st.column_config.NumberColumn("Rate ratio", format="%.2fx"),
                    "p_holm_txt": st.column_config.TextColumn(
                        "Holm-adjusted p", help="State vs all other orders, two-proportion z-test, Holm-adjusted"),
                    "holds": st.column_config.TextColumn("Survives Holm?", width="small"),
                })
            n_hold = int((hv["p_holm"] < ALPHA).sum())
            top = hv.iloc[0]
            many = len(hv) != 1
            ui.callout("info", "Observed",
                       f"{_plural(len(hv), 'state')} {'are' if many else 'is'} flagged as "
                       f"{'high-delay segments' if many else 'a high-delay segment'}. The "
                       f"largest excess is {ui.state_label(top['customer_state'])} at {ui.fmt_pct(top['late_rate'])} "
                       f"({ui.fmt_n(top['n_orders'])}), {top['excess_vs_overall'] * 100:+.1f} pp above overall "
                       f"({top['rate_ratio']:.1f}x). {'Flagged states together account' if many else 'The flagged state accounts'} for "
                       f"{ui.fmt_int(hv['n_orders'].sum())} orders ({ui.fmt_pct(hv['n_orders'].sum() / kpi['n_orders'])} "
                       f"of the view).")
            ui.callout("watch", "Statistical association",
                       f"{n_hold} of {len(hv)} flagged {'states remain' if many else 'state remains'} significant at the "
                       f"{ALPHA:.2f} level after Holm "
                       "adjustment. Statistical significance is not the same as practical importance (check the "
                       "excess and the order count) and none of this shows why "
                       f"{'these states are' if many else 'this state is'} slower.")
            ui.callout("action", "Suggested action",
                       "Investigate flagged states first on the Seller → customer routes tab: is the excess "
                       "concentrated in a few seller states or routes? Then compare their late orders with the "
                       "delivery promise made at checkout.")
        _plot(charts.state_late_rate_bar(perf, min_orders=min_orders, overall_rate=kpi["late_rate"],
                                         include_small=False, title="Late-delivery rate by customer state"),
              "reg_high_bar")
    small = perf[~perf["reliable"]].sort_values("n_orders", ascending=False, kind="mergesort")
    if len(small):
        ui.section_header("Insufficient data", f"Fewer than {min_orders:,} orders: shown for completeness, never "
                          "ranked or flagged. Any rate shown for them elsewhere on this page is marked as insufficient data and "
                          "should not be interpreted.",
                          icon="inbox")
        st.dataframe(
            pd.DataFrame({"State": small["customer_state"], "Name": small["customer_state"].map(
                lambda c: config.STATE_NAMES.get(c, c)), "Orders (n)": small["n_orders"],
                "Status": f"Insufficient data (n < {min_orders})"}),
            hide_index=True, width="stretch", key="reg_small_table", height=_height(len(small)),
            column_config={"Orders (n)": st.column_config.NumberColumn(format="localized")})

# ----------------------------------------------------------------------- #
# Tab 4: seller -> customer routes
# ----------------------------------------------------------------------- #
with tab_routes:
    ui.section_header("Seller state → customer state routes",
                      "How do orders perform depending on where the (representative) seller ships from and where "
                      "the customer lives?", icon="truck")
    ms = rb["multi_seller_share"]
    ui.callout("info", "How multi-seller orders are handled",
               f"An order can contain items from several sellers ({ui.fmt_pct(ms, 2)} of the orders in view do). "
               "Each order is assigned to its representative seller: the one with the largest summed item price "
               "(ties go to the lexicographically smallest seller id). Seller state, route and same-state flags "
               "therefore describe that seller only.")

    same = rb["same"]
    if len(same) == 2:
        s_ = same.set_index("group")
        sa_, cr = s_.loc["same_state"], s_.loc["cross_state"]
        ui.kpi_row([
            dict(label="Same-state orders", value=ui.fmt_pct(sa_["late_rate"]),
                 sublabel=f"late · {ui.fmt_n(sa_['n_orders'])} · mean {ui.fmt_days(sa_['mean_duration'])}",
                 icon="check", accent="green"),
            dict(label="Cross-state orders", value=ui.fmt_pct(cr["late_rate"]),
                 sublabel=f"late · {ui.fmt_n(cr['n_orders'])} · mean {ui.fmt_days(cr['mean_duration'])}",
                 icon="truck", accent="blue"),
            dict(label="Gap (cross minus same)", value=ui.fmt_pp(cr["late_rate"] - sa_["late_rate"]),
                 sublabel=f"{ui.fmt_days(cr['mean_duration'] - sa_['mean_duration'])} longer on average"
                 if cr["mean_duration"] >= sa_["mean_duration"] else
                 f"{ui.fmt_days(sa_['mean_duration'] - cr['mean_duration'])} shorter on average",
                 icon="trend", accent="amber"),
        ])
        zt = sa.two_proportion_ztest(cr["n_late"], cr["n_orders"], sa_["n_late"], sa_["n_orders"])
        ui.callout("info", "Observed",
                   f"Orders shipped across state lines are late {ui.fmt_pct(cr['late_rate'])} of the time "
                   f"({ui.fmt_int(cr['n_late'])} of {ui.fmt_int(cr['n_orders'])}) versus {ui.fmt_pct(sa_['late_rate'])} "
                   f"for same-state orders ({ui.fmt_int(sa_['n_late'])} of {ui.fmt_int(sa_['n_orders'])}), and take "
                   f"{ui.fmt_days(cr['mean_duration'])} versus {ui.fmt_days(sa_['mean_duration'])} on average. "
                   f"Two-proportion z-test {_p_text(zt['p_value'])} (unadjusted). Cross-state orders also travel "
                   "farther and differ in product mix, so the gap is an association, not a cause.")
    elif len(same) == 1:
        g = same.iloc[0]
        ui.callout("info", "Same-state vs cross-state",
                   f"Only {g['group'].replace('_', '-')} orders exist in this view ({ui.fmt_n(g['n_orders'])}), "
                   "so no comparison is possible.")
    else:
        ui.empty_state("No orders to compare", "Widen the filters.")

    # ---- route table ----
    ui.section_header("Route table", "Click a column header to sort. Filter by seller state and customer state.",
                      icon="layers")
    seller_opts = sorted(pairs_all["seller_state"].unique().tolist()) if len(pairs_all) else []
    cust_opts = sorted(pairs_all["customer_state"].unique().tolist()) if len(pairs_all) else []
    r1, r2, r3 = st.columns([2, 2, 1], vertical_alignment="bottom")
    _sticky("reg_pair_seller", valid=lambda v: isinstance(v, list) and all(x in seller_opts for x in v))
    _sticky("reg_pair_customer", valid=lambda v: isinstance(v, list) and all(x in cust_opts for x in v))
    _sticky("reg_pair_only_reliable")
    f_seller = r1.multiselect("Seller state", seller_opts, key="reg_pair_seller", format_func=ui.state_label,
                              placeholder="All seller states")
    f_cust = r2.multiselect("Customer state", cust_opts, key="reg_pair_customer", format_func=ui.state_label,
                            placeholder="All customer states")
    only_rel = r3.checkbox("Hide small routes", value=True, key="reg_pair_only_reliable",
                           help="Hide routes with fewer orders than the minimum sample above")
    pv = pairs_all
    if f_seller:
        pv = pv[pv["seller_state"].isin(f_seller)]
    if f_cust:
        pv = pv[pv["customer_state"].isin(f_cust)]
    n_before = len(pv)
    table_pairs = pv[pv["reliable"]] if only_rel else pv
    if table_pairs.empty:
        ui.empty_state("No routes match these filters",
                       f"Try fewer filters, or untick the minimum-sample box ({_plural(n_before, 'route')} below "
                       f"{min_orders:,} orders)." if n_before else "Try fewer filters.", icon="truck")
    else:
        tp = table_pairs.sort_values(["n_orders", "late_rate"], ascending=[False, False], kind="mergesort")
        tp = tp.assign(route=tp["state_pair"], scope=np.where(tp["same_state"], "Same state", "Cross state"),
                       status=np.where(tp["reliable"], "OK", f"Insufficient data (n < {min_orders})"))
        st.dataframe(
            tp[["route", "scope", "n_orders", "mean_duration", "late_rate", "late_ci_low", "late_ci_high",
                "status"]],
            hide_index=True, width="stretch", key="reg_route_table",
            column_config={
                "route": st.column_config.TextColumn("Route", width="small",
                                                     help="Seller state → customer state"),
                "scope": st.column_config.TextColumn("Scope", width="small"),
                "n_orders": st.column_config.NumberColumn("Orders (n)", format="localized"),
                "mean_duration": st.column_config.NumberColumn("Mean delivery (days)", format="%.1f"),
                "late_rate": st.column_config.NumberColumn("Late rate", format="percent"),
                "late_ci_low": st.column_config.NumberColumn("CI low", format="percent"),
                "late_ci_high": st.column_config.NumberColumn("CI high", format="percent"),
                "status": st.column_config.TextColumn("Sample", width="medium"),
            })
        hidden = n_before - len(table_pairs)
        st.caption(f"{_plural(len(table_pairs), 'route')} shown"
                   + (f"; {_plural(hidden, 'route')} with fewer than {min_orders:,} orders hidden" if hidden else "")
                   + f". Late-rate intervals are {CI_LEVEL} Wilson intervals, unadjusted for the number of routes.")

    # ---- top routes chart ----
    ui.section_header("Top routes", "Ranks only routes that meet the minimum sample.", icon="bar-chart")
    t1, t2 = st.columns([2, 1], vertical_alignment="bottom")
    _sticky("reg_route_metric")
    _sticky("reg_route_topn")
    rm_label = t1.radio("Rank routes by", list(ROUTE_METRICS), horizontal=True, key="reg_route_metric")
    top_n = int(t2.slider("Routes to show", 5, 25, 12, key="reg_route_topn"))
    _plot(charts.state_pair_bar(pv, top_n=top_n, metric=ROUTE_METRICS[rm_label], min_orders=min_orders,
                                overall_rate=kpi["late_rate"]), "reg_route_bar")

    # ---- heat map ----
    ui.section_header("Seller × customer heat map", "Cells with fewer than the minimum number of orders are "
                      "masked grey.", icon="layers")
    h1, h2 = st.columns([2, 1], vertical_alignment="bottom")
    _sticky("reg_heat_metric")
    _sticky("reg_heat_top")
    hm_label = h1.radio("Heat-map metric", list(HEAT_METRICS), horizontal=True, key="reg_heat_metric")
    heat_n = h2.selectbox("States per axis", [8, 12, 16, 27], index=1, key="reg_heat_top",
                          format_func=lambda v: "All" if v == 27 else f"Top {v}")
    hm = _heat_matrices(df, key, min_orders, 0 if heat_n == 27 else int(heat_n))
    _plot(charts.state_pair_heatmap(hm, metric=HEAT_METRICS[hm_label], min_orders=min_orders), "reg_heat_chart")

    # ---- single pair lookup ----
    ui.section_header("Route look-up", "Choose a seller state and a customer state to see one route.",
                      icon="map-pin")
    all_sellers = sorted(df["seller_state"].dropna().astype(str).unique().tolist())
    all_custs = sorted(df["customer_state"].dropna().astype(str).unique().tolist())
    l1, l2 = st.columns(2)
    _sticky("reg_lookup_seller", valid=lambda v: v in all_sellers)
    _sticky("reg_lookup_customer", valid=lambda v: v in all_custs)
    sel_s = l1.selectbox("Seller state (origin)", all_sellers, key="reg_lookup_seller", format_func=ui.state_label,
                         index=all_sellers.index("SP") if "SP" in all_sellers else 0)
    sel_c = l2.selectbox("Customer state (destination)", all_custs, key="reg_lookup_customer",
                         format_func=ui.state_label,
                         index=all_custs.index("RJ") if "RJ" in all_custs else 0)
    hit = pairs_all[(pairs_all["seller_state"] == sel_s) & (pairs_all["customer_state"] == sel_c)]
    if hit.empty:
        ui.empty_state(f"No orders on {sel_s} → {sel_c}", "This seller-to-customer pair has no delivered orders under "
                       "the current filters.", icon="truck")
    else:
        h = hit.iloc[0]
        ui.kpi_row([
            dict(label=f"Seller {sel_s} → Customer {sel_c}", value=ui.fmt_int(h["n_orders"]),
                 sublabel=f"delivered {'order' if int(h['n_orders']) == 1 else 'orders'} ({ui.fmt_n(h['n_orders'])})",
                 icon="package", accent="blue"),
            dict(label="Mean delivery time", value=ui.fmt_days(h["mean_duration"]),
                 sublabel=f"median {ui.fmt_days(h['median_duration'])}", icon="clock", accent="slate"),
            dict(label="Late rate" if h["reliable"] else "Late rate (small sample)", value=ui.fmt_pct(h["late_rate"]),
                 sublabel=f"{CI_LEVEL} interval {ui.fmt_pct(h['late_ci_low'])} to {ui.fmt_pct(h['late_ci_high'])}"
                 + ("" if h["reliable"] else f" · under {min_orders:,} orders, indicative only"),
                 delta=ui.fmt_pp(h["late_rate"] - kpi["late_rate"]) + " vs overall",
                 delta_tone=_lookup_tone(h, kpi["late_rate"]),
                 icon="alert",
                 accent={"bad": "red", "good": "green"}.get(_lookup_tone(h, kpi["late_rate"]), "slate")),
        ])
        if not h["reliable"]:
            ui.callout("watch", "Insufficient data",
                       f"This route has {ui.fmt_n(h['n_orders'])}, below the {min_orders:,}-order minimum. Treat its "
                       "rate as indicative only; the interval is wide.")
        else:
            verdict = ("entirely above" if h["late_ci_low"] > kpi["late_rate"]
                       else "entirely below" if h["late_ci_high"] < kpi["late_rate"] else "overlapping")
            ui.callout("info", "Observed",
                       f"On {sel_s} → {sel_c}, {ui.fmt_int(h['n_late'])} of {ui.fmt_int(h['n_orders'])} orders were "
                       f"late; the interval is {verdict} the overall rate of {ui.fmt_pct(kpi['late_rate'])}.")
