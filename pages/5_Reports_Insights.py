"""Reports & Insights: key findings, downloads, data explorer and model evaluation.

All numbers come from ``src.insights`` / ``src.analytics`` / ``models/model_metrics.json``; the page only lays them
out.  Everything is computed for the CURRENT sidebar filters, except the model evaluation, which is a fixed
historical test and does not change with the filters.
"""
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:      # lets the page run standalone (e.g. streamlit.testing AppTest)
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import analytics as an  # noqa: E402
from src import charts, config, exports, insights as ins, ui  # noqa: E402

KEY = "rep_"
DASH = ui.DASH

PAGE_CSS = """
.rp-strip{display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:.7rem; margin:.2rem 0 .6rem;}
.rp-how{background:#fff; border:1px solid var(--border); border-radius:12px; padding:.7rem .85rem;}
.rp-how p{margin:.4rem 0 0; font-size:.84rem; color:#475569; line-height:1.45;}
.rp-tag{display:inline-block; font-size:.68rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em;
  padding:.14rem .55rem; border-radius:999px; white-space:nowrap;}
.rp-tag.observed{background:#DBEAFE; color:#1D4ED8;}
.rp-tag.association{background:#EDE9FE; color:#5B21B6;}
.rp-tag.model{background:#CCFBF1; color:#0F766E;}
.rp-tag.action{background:#FEF3C7; color:#92400E;}
.rp-tag.data{background:#F1F5F9; color:#475569;}
.rp-grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,430px),1fr)); gap:.9rem; margin:.4rem 0 1rem;}
.rp-card{background:#fff; border:1px solid var(--border); border-left:5px solid var(--sev); border-radius:14px;
  padding:.95rem 1.05rem .9rem; box-shadow:var(--shadow); min-width:0;}
.rp-card-top{display:flex; flex-wrap:wrap; align-items:center; gap:.4rem; margin-bottom:.45rem;}
.rp-sev{font-size:.68rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em; color:var(--sev);
  border:1px solid var(--sev); border-radius:999px; padding:.1rem .5rem;}
.rp-n{margin-left:auto; font-size:.78rem; color:var(--slate-light); font-weight:600; font-variant-numeric:tabular-nums;}
.rp-head{font-size:1.02rem; font-weight:700; color:var(--navy); margin:.1rem 0 .35rem; line-height:1.3;}
.rp-text{font-size:.88rem; color:#334155; line-height:1.55; margin:0 0 .6rem; overflow-wrap:anywhere;}
.rp-chips{display:flex; flex-wrap:wrap; gap:.35rem;}
.rp-chip{font-size:.76rem; background:#F1F5F9; border-radius:8px; padding:.18rem .55rem; color:#334155;}
.rp-chip b{color:var(--navy); font-weight:700;}
.rp-warn{margin-top:.5rem; font-size:.78rem; color:#92400E; background:#FFFBEB; border:1px solid #FDE68A;
  border-radius:8px; padding:.25rem .5rem;}
.rp-reco{display:flex; gap:.7rem; align-items:flex-start; background:#fff; border:1px solid var(--border);
  border-left:5px solid #F59E0B; border-radius:12px; padding:.7rem .9rem; margin:.45rem 0; box-shadow:var(--shadow);}
.rp-reco .num{flex:0 0 auto; width:1.6rem; height:1.6rem; border-radius:50%; background:#FEF3C7; color:#92400E;
  font-weight:800; font-size:.8rem; display:flex; align-items:center; justify-content:center;}
.rp-reco .txt{font-size:.9rem; color:#334155; line-height:1.55; min-width:0; overflow-wrap:anywhere;}
.rp-dl-title{font-weight:700; color:var(--navy); font-size:1.02rem;}
.rp-dl-desc{font-size:.86rem; color:#475569; margin:.25rem 0 .5rem; line-height:1.5;}
.rp-dl-meta{display:flex; flex-wrap:wrap; gap:.4rem; margin-bottom:.3rem;}
[data-testid="stMain"] button:disabled, [data-testid="stMain"] button[kind="primary"]:disabled{
  opacity:.4; cursor:not-allowed; filter:grayscale(.6);}
@media (max-width:640px){ .rp-grid{grid-template-columns:minmax(0,1fr);} }
"""

SEVERITY = {"alert": ("Alert", "#DC2626"), "watch": ("Watch", "#D97706"), "info": ("Info", "#64748B")}
TAG_CLASS = {exports.OBSERVED: "observed", exports.ASSOCIATION: "association", exports.DATA_NOTE: "data",
             "Model prediction": "model", "Suggested action": "action"}


def tag(label: str) -> str:
    """HTML of a small type label (Observed / Statistical association / Model prediction / Suggested action)."""
    return f'<span class="rp-tag {TAG_CLASS.get(label, "data")}">{ui.esc(label)}</span>'


def note(kind: str, label: str, title: str | None, body: str) -> None:
    """Callout carrying a type label (Model prediction / Suggested action ...) in front of its title."""
    icon = {"info": "info", "watch": "alert", "alert": "x-circle", "success": "check"}.get(kind, "info")
    kind = kind if kind in ("info", "watch", "alert", "success") else "info"
    t = ui.esc(title) if title else ""
    st.html(f'<div class="diq-callout {kind}" role="note"><span class="diq-callout-ico">{ui.icon_svg(icon, 20)}</span>'
            f'<div class="diq-callout-body"><div class="diq-callout-title">{tag(label)} {t}</div>'
            f'<div class="diq-callout-text">{ui.esc(body)}</div></div></div>')


def chips(pairs) -> str:
    return "".join(f'<span class="rp-chip"><b>{ui.esc(k)}</b> {ui.esc(v)}</span>' for k, v in pairs)


def page_link(path: str, label: str, icon: str, key: str) -> None:
    """Cross-link to another page (silently skipped when the page is run standalone)."""
    try:
        st.page_link(path, label=label, icon=icon)
    except Exception:  # standalone run: the multipage registry does not know the page
        st.caption(label)


def reset_explorer_filters(lo: date, hi: date) -> None:
    """Callback: put every explorer widget back to its default."""
    st.session_state[KEY + "ex_search"] = ""
    st.session_state[KEY + "ex_status"] = "All"
    st.session_state[KEY + "ex_dates"] = (lo, hi)
    st.session_state[KEY + "ex_states"] = []
    st.session_state[KEY + "ex_cats"] = []
    st.session_state[KEY + "ex_page"] = 1


def step_page(delta: int, n_pages: int) -> None:
    """Callback for the previous / next page buttons."""
    cur = int(st.session_state.get(KEY + "ex_page", 1))
    st.session_state[KEY + "ex_page"] = min(max(1, cur + delta), max(1, n_pages))


# --------------------------------------------------------------------------- #
# Cached computations (keyed on the filter parameters; the frame itself is not hashed)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, max_entries=12)
def cached_insights(_df: pd.DataFrame, fkey: tuple, min_orders: int, alpha: float):
    return ins.generate_insights(_df, min_orders, alpha)


@st.cache_data(show_spinner=False, max_entries=12)
def cached_regional(_df: pd.DataFrame, fkey: tuple, min_orders: int, alpha: float) -> pd.DataFrame:
    return ins.regional_report(_df, min_orders, alpha)


@st.cache_data(show_spinner=False, max_entries=6)
def cached_summary(_df: pd.DataFrame, fkey: tuple, description: str, min_orders: int, alpha: float,
                   _metrics: dict | None, metrics_sig: str) -> str:
    items = ins.generate_insights(_df, min_orders, alpha)
    return ins.summary_report(_df, description, items, _metrics, alpha=alpha)


@st.cache_data(show_spinner="Preparing the shipment CSV…", max_entries=3)
def cached_shipments_csv(_df: pd.DataFrame, fkey: tuple, columns: tuple) -> bytes:
    return exports.shipments_csv_bytes(_df, list(columns))


@st.cache_data(show_spinner=False, max_entries=8)
def cached_chart(name: str, _df: pd.DataFrame, fkey: tuple, min_orders: int):
    k = an.kpi_summary(_df)
    if name == "Monthly delivery performance":
        return charts.monthly_performance_chart(an.monthly_performance(_df, min_orders), min_orders=min_orders)
    if name == "Late rate by customer state":
        return charts.state_late_rate_bar(an.state_performance(_df, min_orders), min_orders=min_orders,
                                          overall_rate=k["late_rate"])
    # "unknown" is the bucket of orders with no recorded category (a data gap, not a product category): the findings
    # exclude it, so the chart does too; the overall reference line still covers every order in view
    known = _df[_df["product_category"].astype(str) != ins.UNKNOWN_CATEGORY]
    return charts.category_late_rate_bar(an.category_performance(known, min_orders), min_orders=min_orders,
                                         overall_rate=k["late_rate"])


CHART_CHOICES = ("Monthly delivery performance", "Late rate by customer state", "Late rate by product category")


def metrics_signature(metrics: dict | None) -> str:
    return str((metrics or {}).get("generated_at", "none"))


# --------------------------------------------------------------------------- #
# Page frame
# --------------------------------------------------------------------------- #
ui.inject_css()
st.html(f"<style>{PAGE_CSS}</style>")
df_all, df, flt = ui.get_filtered_orders()
metrics = ui.get_model_metrics()
n = len(df)
fkey = (str(flt["date_range"]), tuple(flt["states"]), tuple(flt["categories"]), n, len(df_all))

ui.page_header("Reports & Insights",
               "What the data says for the orders in view, what you can download, and how far the delay-risk model "
               "can be trusted. Everything is computed from the data and the trained model, nothing is typed in.",
               icon="file-text", eyebrow="DeliveryIQ")
ui.filter_banner(flt)

ui.section_header("How to read this page", icon="info")
st.html('<div class="rp-strip">'
        f'<div class="rp-how">{tag(exports.OBSERVED)}<p>Counted directly from the orders in view, always with n and an interval.</p></div>'
        f'<div class="rp-how">{tag(exports.ASSOCIATION)}<p>A statistical test or correlation: two things move together. It does not show that one causes the other.</p></div>'
        f'<div class="rp-how">{tag("Model prediction")}<p>Output of the trained model, judged on later months it never saw (Model evaluation tab).</p></div>'
        f'<div class="rp-how">{tag("Suggested action")}<p>Where an operations manager could look next: a lead to check, never a verdict.</p></div>'
        '</div>')
cols = st.columns(4)
for col, (path, label, icon) in zip(cols, [("pages/1_Overview.py", "Overview", ":material/dashboard:"),
                                            ("pages/2_Delay_Analysis.py", "Delay Analysis", ":material/query_stats:"),
                                            ("pages/3_Regional_Intelligence.py", "Regional Intelligence", ":material/map:"),
                                            ("pages/4_Delay_Prediction.py", "Delay Prediction", ":material/psychology:")]):
    with col:
        page_link(path, label, icon, label)

tab_find, tab_dl, tab_ex, tab_model = st.tabs(["Key findings", "Downloads", "Data explorer", "Model evaluation"])

# =========================================================================== #
# Tab 1: key findings
# =========================================================================== #
with tab_find:
    ui.section_header("Analysis settings", "These two controls change every finding, the regional CSV and the summary report.",
                      icon="settings")
    c1, c2 = st.columns([3, 1])
    with c1:
        min_orders = st.slider("Minimum orders before a segment is ranked or flagged", 20, 1000,
                               config.DEFAULT_MIN_ORDERS, step=10, key=KEY + "min_orders",
                               help="Segments (state, category, route, month) with fewer orders than this are shown with "
                                    "their rate but never ranked or called higher / lower.")
    with c2:
        conf = st.selectbox("Interval confidence", [0.90, 0.95, 0.99], index=1, key=KEY + "confidence",
                            format_func=lambda x: f"{x:.0%}")
    alpha = round(1 - float(conf), 4)
    lvl = f"{conf:.0%}"

    if n == 0:
        ui.empty_state("No orders match the current filters",
                       "There is nothing to analyse. Widen the date range or clear the state / category filters in the sidebar.")
        insights_list = ins.generate_insights(df, min_orders, alpha)
    else:
        insights_list = cached_insights(df, fkey, int(min_orders), alpha)
        k = an.kpi_summary(df)
        lo, hi = an.wilson_ci(k["n_late"], k["n_orders"], alpha)
        flagged = [i for i in insights_list if i.kind not in ("overview",) and i.severity in ("alert", "watch")]
        n_alert = sum(i.severity == "alert" for i in flagged)
        ui.kpi_row([
            dict(label="Orders in view", value=ui.fmt_int(k["n_orders"]), sublabel=f"of {ui.fmt_int(len(df_all))} delivered orders",
                 icon="package", accent="blue"),
            dict(label="Late rate", value=ui.fmt_pct(k["late_rate"]),
                 sublabel=f"{ui.fmt_int(k['n_late'])} late orders · {lvl} interval {ui.fmt_pct(lo)} to {ui.fmt_pct(hi)} · "
                          f"{ui.fmt_n(k['n_orders'])}", icon="alert", accent="red"),
            dict(label="Mean delivery time", value=ui.fmt_days(k["mean_duration"]),
                 sublabel=f"median {ui.fmt_days(k['median_duration'])} · late orders arrive {ui.fmt_days(k['mean_late_days'])} after the estimate on average",
                 icon="truck", accent="slate"),
            dict(label="Findings flagged", value=ui.fmt_int(len(flagged)),
                 sublabel=f"{n_alert} alert · {len(flagged) - n_alert} watch", icon="target", accent="green"),
        ])
        if n < min_orders:
            ui.callout("watch", "Small sample",
                       f"Only {ui.fmt_int(n)} orders match, below the minimum of {ui.fmt_int(min_orders)}. Comparisons are "
                       "skipped on purpose: rates from so few orders are mostly noise.")

        ui.section_header("Findings for the current filters",
                          "They are recomputed whenever the sidebar filters, the minimum-orders slider or the confidence level change. "
                          "Sorted by strength of signal.", icon="bar-chart")
        only_flagged = st.toggle("Show only flagged findings (alert or watch)", key=KEY + "only_flagged", value=False)
        shown = [i for i in exports.sort_insights(insights_list) if i.kind != "overview"]
        if only_flagged:
            shown = [i for i in shown if i.severity in ("alert", "watch")]
        if not shown:
            ui.empty_state("No findings to show", "Nothing is flagged for the current filters. Switch off the toggle to see the "
                                                    "descriptive findings.", icon="check")
        cards = []
        for i in shown:
            label, color = SEVERITY.get(i.severity, SEVERITY["info"])
            warn = ""
            small = ui.small_sample_note(i.n, min_orders) if i.kind not in ("insufficient_data", "no_data") else None
            if small:
                warn = f'<div class="rp-warn">{ui.esc(small)}</div>'
            cards.append(
                f'<div class="rp-card" style="--sev:{color}"><div class="rp-card-top">{tag(exports.insight_type_label(i))}'
                f'<span class="rp-sev">{ui.esc(label)}</span><span class="rp-n">{ui.esc(ui.fmt_n(i.n))}</span></div>'
                f'<div class="rp-head">{ui.esc(i.title)}</div><p class="rp-text">{ui.esc(i.text)}</p>'
                f'<div class="rp-chips">{chips(exports.insight_metric_pairs(i))}</div>{warn}</div>')
        if cards:
            st.html(f'<div class="rp-grid">{"".join(cards)}</div>')
        st.caption("Alert = interval or test is clear AND the worst segment is at least "
                   f"{ins.ALERT_RATIO:g}x the comparison rate. Watch = a clear difference exists. Info = descriptive, or no clear "
                   "difference. Labels describe statistical strength, not business importance.")

        ui.section_header("Suggested actions", "Cautious, evidence-based leads to investigate or review. None of them is a root-cause claim.",
                          icon="target")
        recs = ins.recommendations(insights_list)
        if recs:
            st.html("".join(f'<div class="rp-reco"><span class="num">{j}</span><div class="txt">{tag("Suggested action")} '
                            f'{ui.esc(r)}</div></div>' for j, r in enumerate(recs, 1)))
        else:
            ui.empty_state("No suggestions for this selection", "The current orders do not support any recommendation.", icon="check")

        ui.section_header("Model prediction (separate from the findings above)", icon="cpu")
        head = exports.model_headline(metrics)
        if head is None:
            ui.callout("info", "Model evaluation not available",
                       f"The trained model's metrics file is missing. Train the model with `{ui.TRAIN_COMMAND}` to see its evaluation.")
        else:
            note("watch", "Model prediction", f"{head['label']}: {exports.signal_phrase(head['test_roc_auc'])[0]}, not an accurate predictor",
                       f"On the untouched test window the model reaches ROC-AUC {ui.fmt_num(head['test_roc_auc'], 2)} (0.5 = no skill) and PR-AUC "
                       f"{ui.fmt_num(head['test_pr_auc'], 3)} against a no-skill level of {ui.fmt_num(head['test_late_rate'], 3)} "
                       f"({ui.fmt_n(head['test_n'])}). "
                       + (f"It is much weaker than on the validation window (ROC-AUC {ui.fmt_num(head['val_roc_auc'], 2)}). "
                          if (head['val_roc_auc'] or 0) - (head['test_roc_auc'] or 0) > 0.02 else
                          f"Validation ROC-AUC was {ui.fmt_num(head['val_roc_auc'], 2)}. ")
                       + "Details and charts are in the Model evaluation tab.")

    ui.section_header("Methodology note", icon="info")
    with st.expander("What 'reliable' means, and how much to trust a flag", expanded=True):
        m_by_kind = {i.kind: i.metrics for i in insights_list}
        scr = [f"{ui.fmt_int(m_by_kind[kk]['n_screened'])} {nm}" for kk, nm in
               (("region_high", "regions"), ("category_delay", "categories"), ("pair_high", "routes"))
               if kk in m_by_kind and m_by_kind[kk].get("n_screened")]
        tested = m_by_kind.get("month_high", {}).get("n_tested")
        overall_txt = (f"{ui.fmt_pct(an.kpi_summary(df)['late_rate'])} for the orders in view" if n
                       else "undefined while no orders are in view")
        st.markdown(
            f"- **Reliable segment (sample size).** A segment counts as *reliable* when it has at least **{min_orders:,}** delivered orders "
            "in the current view (slider above). Smaller segments are still shown with their n, but they are never ranked or flagged. "
            "This says only that the sample is large enough, not that the finding is true.\n"
            f"- **Reliably higher / lower than overall.** The segment's {lvl} Wilson confidence interval lies entirely above (or below) the "
            f"overall late rate ({overall_txt}). These intervals are **unadjusted**: "
            + (f"in this view the screens inspected {', '.join(scr)}, so " if scr else "many segments are inspected at once, so ")
            + "some segments will look different by chance alone. Treat a flagged segment as a lead to check, not a confirmed effect.\n"
            + (f"- **Adjusted tests.** Each month is compared with all other months by a two-proportion test, with a Bonferroni adjustment over "
               f"{ui.fmt_int(tested)} months tested. The chi-square and correlation tests are reported per test, unadjusted.\n"
               if tested else
               "- **Adjusted tests.** The month-by-month screen is Bonferroni-adjusted; it needs at least two months with enough orders.\n")
            + "- **Significance is not importance.** With tens of thousands of orders, tiny differences become statistically significant. Look "
            "at the effect size (Cramér's V, rho, percentage-point difference) as well as the p-value.\n"
            "- **Association, not causation.** The data are observational. A region, month or category being linked with more late orders "
            "does not show that it causes them; carriers, sellers, weather and demand peaks overlap.\n"
            "- **Retrospective measures.** Processing time (purchase to carrier hand-over) is only known after the order, so it can describe "
            "past performance but is never used to predict.\n"
            "- **Historical data.** Olist orders from 2016 to 2018 do not describe current logistics performance.")

# =========================================================================== #
# Tab 2: downloads
# =========================================================================== #
with tab_dl:
    ui.section_header("Downloads", f"Files are built for the current filters ({ui.fmt_int(n)} orders). Minimum orders "
                      f"{ui.fmt_int(min_orders)} and {lvl} intervals come from the Key findings tab.", icon="database")
    empty = n == 0
    if empty:
        ui.empty_state("No orders match the current filters", "The downloads are disabled until at least one order is in view. "
                       "Widen the date range or clear the state / category filters in the sidebar.")
    date_rng = flt["date_range"]
    fname = dict(date_range=date_rng, states=flt["states"], categories=flt["categories"])  # filters go into file names

    # ---- (a) shipments ------------------------------------------------------
    with st.container(border=True, key=KEY + "dl_ship"):
        st.html('<div class="rp-dl-title">Filtered shipment dataset (CSV)</div>'
                '<div class="rp-dl-desc">One row per delivered order in view, with the analytical columns below. '
                'UTF-8 with a byte-order mark so Excel shows accents; no row-number column.</div>')
        with st.expander("Choose columns and read the data dictionary", expanded=False):
            chosen = st.multiselect("Columns to include", list(exports.EXPORT_COLUMNS), default=list(exports.EXPORT_COLUMNS),
                                    key=KEY + "dl_cols", format_func=exports.column_label)
            st.dataframe(exports.data_dictionary(), hide_index=True, width="stretch", height=300,
                         column_config={"column": "Column", "label": "Label", "description": "Meaning"})
            st.caption(exports.EXCLUDED_NOTE)
        if not chosen:
            chosen = list(exports.EXPORT_COLUMNS)
            st.warning("No column selected; the full whitelist is used.")
        if empty:
            data_ship = exports.shipments_csv_bytes(df, chosen)
        else:
            data_ship = cached_shipments_csv(df, fkey, tuple(chosen))
        st.html('<div class="rp-dl-meta">'
                f'<span class="rp-chip"><b>{ui.fmt_int(n)}</b> rows</span><span class="rp-chip"><b>{len(chosen)}</b> columns</span>'
                f'<span class="rp-chip"><b>{ui.esc(exports.human_size(len(data_ship)))}</b></span></div>')
        st.download_button("Download shipment CSV", data_ship, file_name=exports.export_filename("shipments", **fname),
                           mime="text/csv", key=KEY + "dl_ship_btn", type="primary", icon=":material/download:",
                           disabled=empty, on_click="ignore")

    # ---- (b) regional ---------------------------------------------------------
    with st.container(border=True, key=KEY + "dl_region"):
        st.html('<div class="rp-dl-title">Region-wise performance (CSV)</div>'
                '<div class="rp-dl-desc">One row per customer state: orders, late rate with interval, delivery times, freight and '
                'whether the interval excludes the overall late rate. The file equals the table below.</div>')
        regional = cached_regional(df, fkey, int(min_orders), alpha) if not empty else ins.regional_report(df, int(min_orders), alpha)
        data_reg = exports.regional_csv_bytes(regional)
        st.html('<div class="rp-dl-meta">'
                f'<span class="rp-chip"><b>{ui.fmt_int(len(regional))}</b> {"state" if len(regional) == 1 else "states"}</span>'
                f'<span class="rp-chip"><b>{regional.shape[1]}</b> columns</span>'
                f'<span class="rp-chip"><b>{ui.esc(exports.human_size(len(data_reg)))}</b></span></div>')
        if not regional.empty:
            st.dataframe(regional, hide_index=True, width="stretch", height=min(300, 40 + 35 * len(regional)), column_config={
                "customer_state": "State", "state_name": "Name", "n_orders": st.column_config.NumberColumn("Orders (n)", format="localized"),
                "n_late": st.column_config.NumberColumn("Late orders", format="localized"),
                "late_rate": st.column_config.NumberColumn("Late rate", format="percent"),
                "late_ci_low": st.column_config.NumberColumn("CI low", format="percent"),
                "late_ci_high": st.column_config.NumberColumn("CI high", format="percent"),
                "on_time_rate": st.column_config.NumberColumn("On-time rate", format="percent"),
                "mean_duration": st.column_config.NumberColumn("Mean days", format="%.1f"),
                "median_duration": st.column_config.NumberColumn("Median days", format="%.1f"),
                "mean_late_days": st.column_config.NumberColumn("Mean days late", format="%.1f"),
                "mean_freight": st.column_config.NumberColumn("Mean freight (R$)", format="%.2f"),
                "mean_weight_g": st.column_config.NumberColumn("Mean weight (g)", format="%.0f"),
                "reliable": "Enough orders", "overall_late_rate": st.column_config.NumberColumn("Overall late rate", format="percent"),
                "diff_vs_overall_pp": st.column_config.NumberColumn("Diff vs overall (pp)", format="%+.1f"),
                "flag": "Interval vs overall"})
        st.download_button("Download regional CSV", data_reg, file_name=exports.export_filename("regional_performance", **fname),
                           mime="text/csv", key=KEY + "dl_region_btn", type="primary", icon=":material/download:",
                           disabled=empty, on_click="ignore")

    # ---- (c) summary report -----------------------------------------------------
    with st.container(border=True, key=KEY + "dl_summary"):
        st.html('<div class="rp-dl-title">Summary report of important findings</div>'
                '<div class="rp-dl-desc">Filters, key figures, findings with sample sizes, suggested actions, the model evaluation and '
                'the methodology caveats, as Markdown or as a self-contained HTML page.</div>')
        if empty:
            md_text = ins.summary_report(df, flt["description"], insights_list, metrics, alpha=alpha)
        else:
            md_text = cached_summary(df, fkey, flt["description"], int(min_orders), alpha, metrics, metrics_signature(metrics))
        md_bytes = md_text.encode("utf-8")
        html_bytes = exports.summary_html_bytes(md_text)
        st.html('<div class="rp-dl-meta">'
                f'<span class="rp-chip"><b>{ui.esc(exports.human_size(len(md_bytes)))}</b> Markdown</span>'
                f'<span class="rp-chip"><b>{ui.esc(exports.human_size(len(html_bytes)))}</b> HTML</span></div>')
        b1, b2, _ = st.columns([1, 1, 2])
        with b1:
            st.download_button("Download report (.md)", md_bytes, file_name=exports.export_filename("summary", ext="md", **fname),
                               mime="text/markdown", key=KEY + "dl_md_btn", type="primary", icon=":material/download:",
                               disabled=empty, on_click="ignore", width="stretch")
        with b2:
            st.download_button("Download report (.html)", html_bytes, file_name=exports.export_filename("summary", ext="html", **fname),
                               mime="text/html", key=KEY + "dl_html_btn", icon=":material/download:", disabled=empty,
                               on_click="ignore", width="stretch")
        with st.expander("Preview the report"):
            st.markdown(md_text)

    # ---- (d) charts -----------------------------------------------------------------
    with st.container(border=True, key=KEY + "dl_chart"):
        st.html('<div class="rp-dl-title">Chart export (interactive HTML)</div>'
                '<div class="rp-dl-desc">Pick a key chart and save it as a standalone, interactive HTML file for slides or e-mail. '
                'The file loads the Plotly library from the internet when opened.</div>')
        choice = st.selectbox("Chart", CHART_CHOICES, key=KEY + "dl_chart_choice")
        if empty:
            ui.empty_state("Nothing to chart", "No orders match the current filters.")
        else:
            fig = cached_chart(choice, df, fkey, int(min_orders))
            chart_bytes = charts.figure_to_html_bytes(fig)
            st.plotly_chart(fig, width="stretch", config=charts.PLOTLY_CONFIG, key=KEY + "dl_chart_preview")
            st.html(f'<div class="rp-dl-meta"><span class="rp-chip"><b>{ui.esc(exports.human_size(len(chart_bytes)))}</b> HTML</span></div>')
            st.download_button("Download chart (.html)", chart_bytes,
                               file_name=exports.export_filename(choice, ext="html", **fname), mime="text/html",
                               key=KEY + "dl_chart_btn", type="primary", icon=":material/download:", on_click="ignore")

# =========================================================================== #
# Tab 3: data explorer
# =========================================================================== #
with tab_ex:
    ui.section_header("Data explorer", "Browse individual orders. These filters apply on top of the sidebar filters. "
                      "Only one page of rows is drawn at a time.", icon="filter")
    if n == 0:
        ui.empty_state("No orders match the current filters",
                       "Widen the date range or clear the state / category filters in the sidebar.")
    else:
        lo_d, hi_d = df["purchase_date"].min().date(), df["purchase_date"].max().date()
        state_opts = sorted(df["customer_state"].dropna().astype(str).unique())
        cat_opts = sorted(df["product_category"].dropna().astype(str).unique())
        # sanitise stored values BEFORE the widgets exist (the sidebar filters may have narrowed the options)
        for kk, opts in ((KEY + "ex_states", state_opts), (KEY + "ex_cats", cat_opts)):
            cur = st.session_state.get(kk) or []
            clean = [x for x in cur if x in opts]
            if kk not in st.session_state or clean != list(cur):
                st.session_state[kk] = clean
        # The explorer date range lives in session state only (no value= on the widget, which would clash with the stored
        # value whenever the sidebar range changes the widget's bounds).  It follows the sidebar range while untouched,
        # and is clipped into it once the user has narrowed it.
        cur_dates = st.session_state.get(KEY + "ex_dates")
        prev_bounds = st.session_state.get(KEY + "ex_bounds")
        seq = list(cur_dates) if isinstance(cur_dates, (list, tuple)) else ([cur_dates] if cur_dates is not None else [])
        if not seq or (prev_bounds is not None and tuple(seq) == tuple(prev_bounds)):
            st.session_state[KEY + "ex_dates"] = (lo_d, hi_d)
        elif len(seq) == 2 and (seq[0] < lo_d or seq[1] > hi_d):
            new_lo, new_hi = max(seq[0], lo_d), min(seq[1], hi_d)
            st.session_state[KEY + "ex_dates"] = (new_lo, new_hi) if new_lo <= new_hi else (lo_d, hi_d)
        elif len(seq) == 1 and not (lo_d <= seq[0] <= hi_d):
            st.session_state[KEY + "ex_dates"] = (lo_d, hi_d)
        st.session_state[KEY + "ex_bounds"] = (lo_d, hi_d)

        f1, f2 = st.columns([3, 2])
        with f1:
            search = st.text_input("Search", key=KEY + "ex_search", placeholder="Order ID, city or product category",
                                   help="Case-insensitive match on order ID, customer city, seller city and product category.")
        with f2:
            status_label = st.radio("Delivery status", ["All", "On time", "Late"], horizontal=True, key=KEY + "ex_status")
        g1, g2, g3 = st.columns(3)
        with g1:
            ex_dates = st.date_input("Purchase date range", min_value=lo_d, max_value=hi_d,
                                     key=KEY + "ex_dates", format="YYYY-MM-DD")
        with g2:
            ex_states = st.multiselect("Customer state", state_opts, key=KEY + "ex_states", format_func=ui.state_label,
                                       placeholder="All states in view", select_all=False)
        with g3:
            ex_cats = st.multiselect("Product category", cat_opts, key=KEY + "ex_cats", format_func=ui.category_label,
                                     placeholder="All categories in view", select_all=False)

        dr = None
        if isinstance(ex_dates, (list, tuple)):
            dr = (ex_dates[0], ex_dates[1] if len(ex_dates) > 1 else hi_d) if len(ex_dates) else None
        elif ex_dates is not None:
            dr = (ex_dates, hi_d)
        status = {"All": "all", "On time": "on_time", "Late": "late"}[status_label]
        filtered = exports.explorer_filter(df, dr, ex_states, ex_cats, status, search)

        h1, h2, h3, h4 = st.columns([3, 2, 1.3, 1.3])
        with h1:
            with st.popover("Choose columns", icon=":material/view_column:", width="stretch"):
                shown_cols = st.multiselect("Columns shown", list(exports.EXPORT_COLUMNS),
                                            default=list(exports.DEFAULT_EXPLORER_COLUMNS), key=KEY + "ex_cols",
                                            format_func=exports.column_label)
        if not shown_cols:
            shown_cols = list(exports.DEFAULT_EXPLORER_COLUMNS)
        with h2:
            sort_by = st.selectbox("Sort by", shown_cols, key=KEY + "ex_sort", format_func=exports.column_label,
                                   index=shown_cols.index("order_purchase_timestamp") if "order_purchase_timestamp" in shown_cols else 0)
        with h3:
            sort_dir = st.radio("Order", ["Ascending", "Descending"], key=KEY + "ex_dir", horizontal=False)
        with h4:
            page_size = st.selectbox("Rows per page", [25, 50, 100, 250], index=1, key=KEY + "ex_size")

        n_f = len(filtered)
        if n_f == 0:
            ui.empty_state("No orders match the explorer filters",
                           f"The sidebar filters leave {ui.fmt_int(n)} orders, but none match the search, status, date, state "
                           "or category chosen here.", icon="inbox")
            st.button("Reset explorer filters", key=KEY + "ex_reset", on_click=reset_explorer_filters, args=(lo_d, hi_d),
                      icon=":material/restart_alt:")
        else:
            xname = dict(date_range=dr, states=ex_states or flt["states"], categories=ex_cats or flt["categories"],
                         extra=([status] if status != "all" else []) + (["search"] if str(search or "").strip() else []))
            sig = (fkey, search, status, str(dr), tuple(ex_states), tuple(ex_cats), sort_by, sort_dir, page_size)
            if st.session_state.get(KEY + "ex_sig") != sig:
                st.session_state[KEY + "ex_sig"] = sig
                st.session_state[KEY + "ex_page"] = 1
            n_pages = max(1, -(-n_f // int(page_size)))
            st.session_state[KEY + "ex_page"] = min(max(1, int(st.session_state.get(KEY + "ex_page", 1))), n_pages)
            sort_col = "is_late" if sort_by == "delivery_status" else sort_by
            base_cols = [c for c in exports.EXPORT_COLUMNS if c in filtered.columns]
            ordered = filtered[base_cols].sort_values(sort_col, ascending=sort_dir == "Ascending", kind="mergesort",
                                                      na_position="last")
            view, info = exports.explorer_view(ordered, None, True, int(st.session_state[KEY + "ex_page"]), int(page_size))

            n_late_f = int((pd.to_numeric(filtered["is_late"], errors="coerce") == 1).sum())
            st.html('<div class="diq-pillrow">'
                    f'<span class="diq-pill"><b>{ui.fmt_int(n_f)}</b> matching orders</span>'
                    f'<span class="diq-pill">{ui.fmt_pct(n_late_f / n_f)} late ({ui.fmt_n(n_f)})</span>'
                    f'<span class="diq-pill">rows {ui.fmt_int(info["start"])} to {ui.fmt_int(info["end"])}</span></div>')

            kinds = {"text": st.column_config.TextColumn, "status": st.column_config.TextColumn}
            cfg = {}
            for c in shown_cols:
                label, desc, kind = exports.COLUMN_INFO[c]
                if kind in kinds:
                    cfg[c] = kinds[kind](label, help=desc)
                elif kind == "datetime":
                    cfg[c] = st.column_config.DatetimeColumn(label, help=desc, format="YYYY-MM-DD HH:mm")
                else:
                    fmt = {"int": "%d", "flag": "%d", "float": "%.2f", "money": "R$ %.2f", "days": "%.1f"}[kind]
                    cfg[c] = st.column_config.NumberColumn(label, help=desc, format=fmt)
            show_df = exports.export_frame(view, shown_cols)
            st.dataframe(show_df, hide_index=True, width="stretch", height=min(740, 38 + 35 * (len(show_df) + 1)),
                         column_config=cfg, key=KEY + "ex_table")

            p1, p2, p3, p4 = st.columns([1, 1, 1.4, 3])
            with p1:
                st.button("Previous", key=KEY + "ex_prev", on_click=step_page, args=(-1, n_pages), disabled=info["page"] <= 1,
                          icon=":material/chevron_left:", width="stretch")
            with p2:
                st.button("Next", key=KEY + "ex_next", on_click=step_page, args=(1, n_pages), disabled=info["page"] >= n_pages,
                          icon=":material/chevron_right:", icon_position="right", width="stretch")
            with p3:
                st.number_input("Page", 1, n_pages, key=KEY + "ex_page", step=1, disabled=n_pages == 1,
                                label_visibility="collapsed")
            with p4:
                st.caption(f"Page {info['page']:,} of {n_pages:,} · {int(page_size):,} rows per page")

            st.html('<div class="diq-section-sub" style="margin:.8rem 0 .3rem">Download exactly what is shown '
                    f'({len(shown_cols)} chosen columns, current sort order).</div>')
            d1, d2, _ = st.columns([1.2, 1.2, 2])
            sel_cols = list(shown_cols)
            with d1:
                st.download_button(f"This page ({ui.fmt_int(len(show_df))} rows)", exports.shipments_csv_bytes(view, sel_cols),
                                   file_name=exports.export_filename("explorer_page", ext="csv", **xname), mime="text/csv",
                                   key=KEY + "ex_dl_page", icon=":material/download:", on_click="ignore", width="stretch")
            with d2:
                st.download_button(f"All matching ({ui.fmt_int(n_f)} rows)", lambda: exports.shipments_csv_bytes(ordered, sel_cols),
                                   file_name=exports.export_filename("explorer_all", ext="csv", **xname), mime="text/csv",
                                   key=KEY + "ex_dl_all", type="primary", icon=":material/download:", on_click="ignore",
                                   width="stretch")

# =========================================================================== #
# Tab 4: model evaluation
# =========================================================================== #
with tab_model:
    ui.section_header("Model evaluation", "How well the delay-risk model works on months it never saw. This is a fixed historical "
                      "evaluation: it does not change with the sidebar filters.", icon="cpu")
    head = exports.model_headline(metrics)
    if metrics is None or head is None:
        ui.empty_state("Model evaluation is not available",
                       "The metrics file written by the training step was not found or could not be read. Train the model once "
                       "(about a minute) and reload this page.", icon="cpu")
        st.code(ui.TRAIN_COMMAND, language="bash")
    else:
        sig_noun, sig_cmp = exports.signal_phrase(head["test_roc_auc"])
        drop = (head["val_roc_auc"] - head["test_roc_auc"]) if head["val_roc_auc"] is not None and head["test_roc_auc"] is not None else None
        ui.callout("watch", f"Read this first: {sig_noun}, not an accurate predictor",
                   f"{head['label']} was selected. On the untouched test window it ranks late orders {sig_cmp} "
                   f"(ROC-AUC {ui.fmt_num(head['test_roc_auc'], 2)}; PR-AUC {ui.fmt_num(head['test_pr_auc'], 3)} against a no-skill "
                   f"{ui.fmt_num(head['test_late_rate'], 3)})"
                   + (f", {ui.fmt_num(drop, 2)} ROC-AUC below the validation figure" if drop is not None and drop > 0.02 else "")
                   + ". Use it as a rough screening aid, never as a reliable forecast of an individual order.")
        lo_r, hi_r = head["test_roc_ci"]
        ui.kpi_row([
            dict(label="Selected model", value=head["label"], sublabel=f"trained on {ui.fmt_int(exports.dig(metrics, 'split', 'train', 'n'))} orders",
                 icon="cpu", accent="blue"),
            dict(label="Test ROC-AUC", value=ui.fmt_num(head["test_roc_auc"], 2),
                 sublabel=f"95% interval {ui.fmt_num(lo_r, 2)} to {ui.fmt_num(hi_r, 2)} · 0.5 = no skill · {ui.fmt_n(head['test_n'])}",
                 icon="target", accent="amber"),
            dict(label="Test PR-AUC", value=ui.fmt_num(head["test_pr_auc"], 3),
                 sublabel=f"no-skill {ui.fmt_num(head['test_late_rate'], 3)} · lift {ui.fmt_num(head['test_lift'], 1)}x",
                 icon="percent", accent="amber"),
            dict(label="Validation ROC-AUC", value=ui.fmt_num(head["val_roc_auc"], 2),
                 sublabel="tuned window: optimistic, not an honest estimate", icon="bar-chart", accent="slate"),
        ])

        # ---- split ---------------------------------------------------------------
        ui.section_header("Train / validation / test split", "Chronological, so the test window is genuinely later than anything the "
                          "model was fitted or tuned on.", icon="calendar")
        spl = exports.split_table(metrics)
        st.dataframe(spl, hide_index=True, width="stretch", column_config={
            "Orders (n)": st.column_config.NumberColumn(format="localized"),
            "Late orders": st.column_config.NumberColumn(format="localized"),
            "Late rate": st.column_config.ProgressColumn("Late rate", format="percent", min_value=0,
                                                         max_value=float(max(spl["Late rate"].max(), 0.01)) if len(spl) else 1.0)})
        rates = {r["Window"]: r["Late rate"] for _, r in spl.iterrows()}
        if len(rates) == 3:
            spread = (max(rates.values()) - min(rates.values()))
            ui.callout("info", "Why the late rate differs between windows",
                       f"The late rate is {ui.fmt_pct(rates['Train'])} in training, {ui.fmt_pct(rates['Validation'])} in validation and "
                       f"{ui.fmt_pct(rates['Test'])} in test, a spread of {ui.fmt_num(100 * spread, 1)} percentage points. Delay "
                       "regimes change over time, so thresholds and probabilities fitted on one period are applied to another.")
        notes = exports.split_notes(metrics)
        if notes:
            with st.expander("Selection bias, non-stationarity and other notes from the training run"):
                for t in notes:
                    st.markdown(f"- {t}")

        # ---- comparison -----------------------------------------------------------
        ui.section_header("Model comparison", "A constant-probability baseline is included: a model has to beat it to be useful.",
                          icon="bar-chart")
        st.plotly_chart(charts.model_comparison_chart(metrics), width="stretch", config=charts.PLOTLY_CONFIG, key=KEY + "m_compare")
        tbl = exports.model_metrics_table(metrics)
        te = tbl[tbl["Window"] == "Test"]
        va = tbl[tbl["Window"] == "Validation"].set_index("Model")
        if len(te):
            best = te.loc[te["ROC-AUC"].astype(float).idxmax()]
            drops = []
            for _, r in te.iterrows():
                v = va["ROC-AUC"].get(r["Model"])
                if v is not None and float(v) > 0.5:
                    drops.append(f"{r['Model']} {ui.fmt_num(v, 2)} to {ui.fmt_num(r['ROC-AUC'], 2)}")
            same = best["Model"] == head["label"]
            note("info", "Model prediction", None,
                 f"On the test window the highest ROC-AUC is {ui.fmt_num(best['ROC-AUC'], 2)} ({best['Model']})"
                 + ("; that is the selected model. " if same else f"; the selected model ({head['label']}) has "
                    f"{ui.fmt_num(head['test_roc_auc'], 2)}. ")
                 + ("ROC-AUC fell from validation to test for every model: " + "; ".join(drops) + "." if drops else "")
                 + " The differences between models are small compared with that drop, and none is far above the no-skill "
                   f"PR-AUC of {ui.fmt_num(head['test_late_rate'], 3)}.")
        num_fmt = {c: st.column_config.NumberColumn(c, format="%.3f") for c in
                   ("No-skill PR-AUC", "ROC-AUC", "PR-AUC", "Precision", "Recall", "F1", "Brier", "Accuracy")}
        num_fmt["Lift (top 10%)"] = st.column_config.NumberColumn("Lift (top 10%)", format="%.2fx")
        num_fmt["Orders (n)"] = st.column_config.NumberColumn("Orders (n)", format="localized")
        st.dataframe(tbl, hide_index=True, width="stretch", column_config=num_fmt)
        st.caption("Validation rows are tuned / in-sample for the selected model (see the validation note in the metrics file). "
                   "No-skill PR-AUC is the share of late orders in that window. Brier: lower is better.")

        # ---- confusion matrix ------------------------------------------------------
        ui.section_header("What the model gets right and wrong", icon="target")
        cc1, cc2 = st.columns([1, 1])
        with cc1:
            st.plotly_chart(charts.confusion_matrix_chart(metrics), width="stretch", config=charts.PLOTLY_CONFIG, key=KEY + "m_conf")
        with cc2:
            reading = exports.confusion_reading(metrics)
            note("info", "Model prediction", "Plain-language reading",
                 reading or "No confusion matrix is stored for the selected model.")

        # ---- curves ----------------------------------------------------------------
        ui.section_header("Ranking and calibration curves", "Solid line = test window; dashed = validation (tuned).", icon="trend")
        r1, r2 = st.columns(2)
        with r1:
            st.plotly_chart(charts.roc_curve_chart(metrics), width="stretch", config=charts.PLOTLY_CONFIG, key=KEY + "m_roc")
        with r2:
            st.plotly_chart(charts.pr_curve_chart(metrics), width="stretch", config=charts.PLOTLY_CONFIG, key=KEY + "m_pr")
        st.plotly_chart(charts.calibration_chart(metrics), width="stretch", config=charts.PLOTLY_CONFIG, key=KEY + "m_cal")
        cav = exports.dig(metrics, "selected_model", "test_window_caveat") or {}
        pr_ci = head["test_pr_ci"]
        note("info", "Model prediction", "How to read the curves",
                   f"ROC: test AUC {ui.fmt_num(head['test_roc_auc'], 2)} versus {ui.fmt_num(head['val_roc_auc'], 2)} on validation; the closer "
                   "the curve is to the diagonal, the closer the model is to guessing. "
                   f"Precision-recall: test PR-AUC {ui.fmt_num(head['test_pr_auc'], 3)}"
                   + (f" (95% interval {ui.fmt_num(pr_ci[0], 3)} to {ui.fmt_num(pr_ci[1], 3)})" if pr_ci[0] is not None and pr_ci[1] is not None else "")
                   + f", where the no-skill line sits at {ui.fmt_num(head['test_late_rate'], 3)}. Calibration: on the test window the model "
                   f"predicted an average risk of {ui.fmt_pct(cav.get('mean_predicted'))} while {ui.fmt_pct(cav.get('observed_late_rate'))} "
                   "of orders were actually late, so its probabilities should be read as relative risk, not as exact chances.")

        # ---- risk bands -------------------------------------------------------------
        ui.section_header("Risk bands", "Orders are grouped by predicted probability; the chart shows how often each group was actually late.",
                          icon="layers")
        st.plotly_chart(charts.risk_band_chart(metrics), width="stretch", config=charts.PLOTLY_CONFIG, key=KEY + "m_bands")
        bt = exports.risk_band_table(metrics)
        st.dataframe(bt, hide_index=True, width="stretch", column_config={
            "Orders (n)": st.column_config.NumberColumn(format="localized"),
            "Late orders": st.column_config.NumberColumn(format="localized"),
            "Late rate": st.column_config.NumberColumn(format="percent"), "CI low": st.column_config.NumberColumn("95% CI low", format="percent"),
            "CI high": st.column_config.NumberColumn("95% CI high", format="percent")})
        diag = exports.dig(metrics, "risk_bands", "test_diagnostics", "note")
        thr = exports.dig(metrics, "risk_bands", "thresholds") or {}
        td = exports.dig(metrics, "risk_bands", "test_diagnostics") or {}
        overlap = [k_.replace("-", " and ") for k_, v_ in (td.get("adjacent_intervals_overlap") or {}).items() if v_]
        note("watch", "Model prediction", "Reading the bands on the test window",
                   (f"Bands are cut at a predicted probability of {ui.fmt_num(thr.get('moderate'), 3)} (Moderate) and "
                    f"{ui.fmt_num(thr.get('high'), 3)} (High), chosen on the validation window. " if thr else "")
                   + (str(diag) + " " if diag else "")
                   + (f"On the test window the intervals of {', '.join(overlap)} overlap, so those bands cannot be told apart; do not "
                      "treat them as different risk levels without further evidence." if overlap else
                      "Adjacent bands are statistically separated on the test window, but the separation is modest in size."))

        # ---- importance -------------------------------------------------------------
        ui.section_header("What the model relies on", "Permutation importance: how much the score drops when one input is shuffled. "
                          "A predictive association, not a cause.", icon="layers")
        win = st.radio("Window", ["validation", "test"], horizontal=True, key=KEY + "m_imp_window",
                       format_func=lambda w: "Validation (used for tuning)" if w == "validation" else "Test (out-of-sample, diagnostic)")
        st.plotly_chart(charts.feature_importance_chart(metrics, window=win), width="stretch", config=charts.PLOTLY_CONFIG,
                        key=KEY + "m_imp")
        cmp_note = exports.dig(metrics, "permutation_importance", "comparison_note")
        if cmp_note:
            note("info", "Model prediction", "Is the ranking stable?", str(cmp_note))

        # ---- which metrics matter -----------------------------------------------------
        ui.section_header("Which metrics matter for delay-risk prediction", icon="info")
        for pt in exports.metric_guidance_points(metrics):
            st.markdown(f"- {pt}")

        # ---- leakage ----------------------------------------------------------------------
        ui.section_header("Leakage audit", icon="check")
        audit = metrics.get("leakage_audit") or {}
        checks = audit.get("checks") or []
        feats = (metrics.get("features") or {})
        used = list(feats.get("numeric") or []) + list(feats.get("categorical") or [])
        if audit.get("passed"):
            ui.callout("success", "No data leakage found",
                       f"All {len(checks)} automated checks passed. The model uses {len(used)} inputs that are known when the order is placed ("
                       + ", ".join(charts.pretty_feature_name(f) for f in used)
                       + "); delivery dates, outcomes and review scores are never used, the windows are strictly chronological and no "
                         "order appears in two windows.")
        else:
            ui.callout("alert", "Leakage audit did not pass", "Inspect the checks below before trusting any figure on this page.")
        if checks:
            with st.expander("All audit checks"):
                for c in checks:
                    st.markdown(f"- {c}")

        # ---- honest summary ------------------------------------------------------------------
        ui.section_header("Honest summary: limited predictive power and drift", icon="alert")
        warn_list = exports.dig(metrics, "selected_model", "test_window_warnings") or []
        for pt in exports.honest_summary(metrics):
            st.markdown(f"- {pt}")
        if warn_list:
            with st.expander("Warnings recorded by the training run"):
                for w in warn_list:
                    st.markdown(f"- {w}")
        cav2 = exports.dig(metrics, "selected_model", "test_window_caveat") or {}
        worse = bool(cav2.get("brier_not_better_than_baseline"))
        weaker = (head["val_roc_auc"] is not None and head["test_roc_auc"] is not None
                  and head["val_roc_auc"] - head["test_roc_auc"] > 0.02)
        note("alert", "Model prediction", "Bottom line",
             f"The model finds {exports.signal_phrase(head['test_roc_auc'])[0]} (test ROC-AUC {ui.fmt_num(head['test_roc_auc'], 2)}, "
             f"PR-AUC lift {ui.fmt_num(head['test_lift'], 1)}x)"
             + (f", much weaker on later months than on the window it was tuned on (ROC-AUC {ui.fmt_num(head['val_roc_auc'], 2)})"
                if weaker else "")
             + ("; its probabilities are no better than quoting the base rate" if worse else "")
             + ". It cannot see events such as strikes, weather or demand peaks and does not explain why orders are late. "
               "Historical 2016-2018 Olist data is not current logistics performance.")
