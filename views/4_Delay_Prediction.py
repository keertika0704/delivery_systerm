"""Delay Prediction page: model-based late-delivery risk for a hypothetical NEW order.

Uses purchase-time information only.  Every number shown is computed from the trained model bundle,
``models/model_metrics.json`` or the processed orders; the limitations of the model (modest test-window
signal, distribution shift, overlapping bands) are stated next to every result.
"""
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:      # lets the page run standalone (e.g. streamlit.testing AppTest)
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import charts, config, prediction, ui  # noqa: E402
from src import prediction_helpers as H  # noqa: E402

PAL = config.PALETTE
MAX_SCENARIOS = 3
K_LAST, K_SCEN = "pred_last", "pred_scenarios"
BAND_ACCENT = {"Low": "green", "Moderate": "amber", "High": "red"}

ui.inject_css()
st.html("""<style>
.pred-field-err{color:#B91C1C; font-size:.82rem; font-weight:600; margin:-.55rem 0 .6rem; line-height:1.35;}
[data-testid="stMain"] [data-testid="stFormSubmitButton"] button p, [data-testid="stMain"] button[kind="primary"] p{color:#fff !important;}
.pred-note{color:#526071; font-size:.82rem; line-height:1.45; margin:-.35rem 0 .6rem;}
.diq-kpi-top{min-height:2.45rem; align-items:flex-start;}
</style>""")

df_all, _df_filtered, flt = ui.get_filtered_orders()
bundle = None
metrics: dict = {}


# --------------------------------------------------------------------------- #
# small render helpers
# --------------------------------------------------------------------------- #
def pills(items: list[str], warn: bool = False) -> None:
    cls = "diq-pill is-warn" if warn else "diq-pill"
    st.html('<div class="diq-pillrow">' + "".join(f'<span class="{cls}">{ui.esc(t)}</span>' for t in items) + "</div>")


def nb(text) -> str:
    """Dates with non-breaking hyphens so a KPI sub-label never wraps in the middle of a date."""
    return str(text).replace("-", "\u2011")


def bullets(lines: list[str]) -> str:
    return "\n".join("• " + ln for ln in lines)


def band_display(tbl: pd.DataFrame, current: str | None):
    """Formatted copy of the band table (strings) with the current band row highlighted."""
    d = pd.DataFrame({
        "Risk band": tbl["Risk band"],
        "Probability range": tbl["Probability range"],
        "Validation late rate": tbl["Validation late rate"].map(ui.fmt_pct),
        "Validation n": tbl["Validation n"].map(ui.fmt_int),
        "Test late rate": tbl["Test late rate"].map(ui.fmt_pct),
        "Test 95% CI": tbl["Test 95% CI"],
        "Test n": tbl["Test n"].map(ui.fmt_int),
        "This order": tbl["This order"],
    })
    if current is None:
        d = d.drop(columns=["This order"])

    def hi(row):
        on = current is not None and row["Risk band"] == current
        return ["background-color:#EFF6FF; font-weight:600" if on else "" for _ in row]

    return d.style.apply(hi, axis=1)


# --------------------------------------------------------------------------- #
# page start
# --------------------------------------------------------------------------- #
bundle = ui.require_model()          # friendly error + st.stop() when the model is missing
metrics = ui.get_model_metrics() or {}
opts = bundle["form_options"]
thresholds = bundle["risk_thresholds"]
info = H.model_info(bundle, metrics)
defaults = H.form_defaults(df_all, bundle)
yrs = sorted({str(info[k])[:4] for k in ("train_start", "test_end") if info.get(k)})

ui.page_header(
    "Delay Prediction",
    "Estimate the chance that a **new order** arrives after its promised date, using only information known at purchase. "
    "A modest-signal screening aid trained on historical Olist data"
    + (f" ({yrs[0]}–{yrs[-1]})" if yrs else "") + ", not a forecast.",
    icon="target", eyebrow="Model-based risk")
ui.callout("watch", "Read this first", H.read_first(bundle, metrics))
if flt.get("active"):
    ui.callout("info", "Sidebar filters do not apply on this page",
               "A prediction describes the hypothetical order you enter below. The historical context shown with a result "
               "uses all delivered orders, not the filtered view.")

tab_one, tab_batch, tab_model = st.tabs(["Predict an order", "Score a CSV file", "What changes the risk?"])


# --------------------------------------------------------------------------- #
# TAB 1: single order
# --------------------------------------------------------------------------- #
SCROLL_JS = """
(function () {
  var el = document.querySelector('.st-key-pred-result');
  if (el && el.scrollIntoView) { el.scrollIntoView({behavior: 'smooth', block: 'start'}); }
})();
"""


def pin_scenario() -> None:
    last = st.session_state.get(K_LAST)
    scen = st.session_state.setdefault(K_SCEN, [])
    if not last or len(scen) >= MAX_SCENARIOS:
        return
    if any(s["key"] == last["key"] for s in scen):
        st.session_state["pred_pin_msg"] = "That exact order is already pinned."
        return
    res = last["result"]
    scen.append({"key": last["key"], "label": f"Scenario {chr(65 + len(scen))}", "probability": res["probability"],
                 "risk_label": res["risk_label"], "description": last["description"], "inputs": last["inputs"]})
    st.session_state["pred_pin_msg"] = ""


def clear_scenarios() -> None:
    st.session_state[K_SCEN] = []
    st.session_state["pred_pin_msg"] = ""


_FORM_KEYS = {"customer_state": "pred_customer_state", "seller_state": "pred_seller_state", "product_category": "pred_category",
              "n_items": "pred_items", "total_price": "pred_price", "total_freight": "pred_freight", "purchase_date": "pred_date",
              "total_weight_g": "pred_weight", "n_sellers": "pred_n_sellers", "total_volume_cm3": "pred_volume",
              "max_product_length_cm": "pred_length", "max_product_height_cm": "pred_height", "max_product_width_cm": "pred_width"}


def seed_state(key: str, value) -> None:
    """Give a widget its starting value through Session State (only when it has none yet).

    Widgets below are created WITHOUT ``value=`` / ``index=``: Streamlit logs a warning when a widget has both a
    default and a value set through Session State, so all starting values go through here instead.
    """
    if value is not None and key not in st.session_state:
        st.session_state[key] = value


def seed_form(last: dict | None, form_defaults: dict) -> None:
    """Fill the form on the first run of this page visit: from the stored result when the user came back, else defaults.

    Streamlit drops widget state when another page runs, while ``pred_last`` / the pinned scenarios stay,
    so without restoring, the result would describe an order the form no longer shows.
    """
    if "pred_customer_state" in st.session_state:
        return
    src = last["inputs"] if last else form_defaults
    for field, key in _FORM_KEYS.items():
        v = src.get(field)
        if field in opts and v not in opts[field]:      # a value the model does not know: fall back to the first option
            v = opts[field][0]
        seed_state(key, v)
    if last and last["inputs"].get("estimated_delivery_days") is not None:
        inp = last["inputs"]
        seed_state(f"pred_promise_{inp['seller_state']}_{inp['customer_state']}", inp["estimated_delivery_days"])


def render_pin() -> None:
    """Pin-to-compare button, shown right under the headline numbers (not at the bottom of a 4,000 px result)."""
    scen = st.session_state.get(K_SCEN, [])
    pc1, pc2 = st.columns([1, 3], vertical_alignment="center")
    with pc1:
        st.button("Pin to compare", key="pred_pin", on_click=pin_scenario, icon=":material/push_pin:",
                  disabled=len(scen) >= MAX_SCENARIOS, width="stretch",
                  help=f"Keep up to {MAX_SCENARIOS} results side by side.")
    with pc2:
        msg = st.session_state.get("pred_pin_msg")
        st.caption(msg if msg else (f"{len(scen)} of {MAX_SCENARIOS} scenarios pinned." if scen else
                                    "Change an input, predict again and pin each result to compare them."))


def render_result(last: dict, current_inputs: dict) -> None:
    res, inputs = last["result"], last["inputs"]
    p, label = res["probability"], res["risk_label"]
    seller, customer = inputs["seller_state"], inputs["customer_state"]
    promise = (res["derived_features"] or {}).get("estimated_delivery_days")
    history = H.historical_context(df_all, seller, customer, inputs["product_category"], config.DEFAULT_MIN_ORDERS)
    what_if = H.what_if_promise(inputs, bundle, promise) if promise else pd.DataFrame()
    wins = info["windows"]
    ranges = H.band_ranges(res["thresholds"])

    ui.section_header("2. Result", "The model's estimate for the order you described. Read it together with the limitations below.",
                      icon="target")
    pills([last["description"]])
    if current_inputs != inputs:
        changed = [H.FIELD_LABELS.get(k, k) for k in sorted(set(inputs) | set(current_inputs)) if inputs.get(k) != current_inputs.get(k)]
        ui.callout("info", "This result is for the previously submitted order",
                   "The form no longer matches this result (changed: " + ", ".join(changed) + "). "
                   "Press **Predict Delay Risk** again to update it.")
    else:
        st.html('<div class="pred-note">Edited the form? Changes only count once you press <b>Predict Delay Risk</b>; '
                "this result always describes the order shown above.</div>")

    route_h = history["route"]
    t = wins.get("test")
    ui.kpi_row([
        dict(label="Late-delivery probability", value=ui.fmt_pct(p), sublabel="Model estimate, not a certainty",
             icon="percent", accent=BAND_ACCENT[label]),
        dict(label="Predicted risk band", value=label, sublabel=f"Probability {ranges[label]}",
             icon="check" if label == "Low" else "alert", accent=BAND_ACCENT[label]),
        dict(label="Test-window late rate", value=ui.fmt_pct(t["late_rate"]) if t else ui.DASH,
             sublabel=(f"{ui.fmt_n(t['n'])} · {nb(t['start'])} to {nb(t['end'])}" if t else "metrics file missing"),
             icon="bar-chart", accent="slate", help="Share of late orders in the most recent evaluation window."),
        dict(label="Observed on this route", value=ui.fmt_pct(route_h["late_rate"]) if route_h["n"] else ui.DASH,
             sublabel=(f"{seller} to {customer}, {ui.fmt_n(route_h['n'])}" if route_h["n"] else f"no orders on {seller} to {customer}"),
             icon="map-pin", accent="slate", help="Historical late rate of delivered orders on this seller-to-customer state route (all years)."),
    ])
    render_pin()
    st.plotly_chart(charts.probability_gauge(p, res["thresholds"]), width="stretch", key="pred_gauge", config=charts.PLOTLY_CONFIG)

    for w in res["warnings"]:
        ui.callout("watch", "Outside what the model was trained on", H.humanise_warning(w))
    small = ui.small_sample_note(route_h["n"], config.DEFAULT_MIN_ORDERS) if route_h["n"] else None

    for item in H.interpretation(res, bundle, metrics, history, (seller, customer), what_if):
        ui.callout(item["kind"], item["title"], item["body"], footnote=small if item["tag"] == "Observed" else None)
    if res.get("probability_note"):
        ui.callout("watch", "How much to trust this number", res["probability_note"])

    # the same warning is the notice at the top of the page: here only the evaluation figures, collapsed
    with st.expander("Limitations: please read (model evaluation figures)", expanded=False):
        ui.callout("watch", "Limitations: please read", bullets(H.limitations(bundle, metrics, include_calibration=False, include_bands=False)),
                   footnote="These figures come from the model's stored evaluation (models/model_metrics.json).")

    if len(what_if) > 1:
        ui.section_header("How the estimate responds to the delivery promise",
                          "Same order re-scored with a longer promise. An association in historical data, not a recommendation to pad promises.",
                          icon="clock")
        wi = pd.DataFrame({
            "Extra days vs entered promise": what_if["extra_days"].map(lambda d: "entered value" if d == 0 else f"+{d} days"),
            "Delivery promise (days)": what_if["promise_days"],
            "Estimated late probability": what_if["probability"].map(ui.fmt_pct),
            "Risk band": what_if["risk_label"],
        })
        st.dataframe(wi, hide_index=True, width="stretch", key="pred_whatif_table")

    ui.section_header("What each risk band has meant historically",
                      f"Thresholds used by the model: Moderate from {res['thresholds']['moderate']:.1%}, High from "
                      f"{res['thresholds']['high']:.1%} (chosen on the validation window).", icon="layers")
    st.dataframe(band_display(H.band_table(res["thresholds"], metrics, label), label), hide_index=True, width="stretch",
                 key="pred_band_table")
    sep = H.band_separation_text(metrics)
    if sep:
        ui.callout("info", "Do the bands separate?", sep + " Validation figures are in-sample for the thresholds, so the test window "
                   "is the fairer guide.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        ui.section_header("Assumptions used", "Optional inputs left blank are filled with training-window medians.", icon="settings")
        ass = res["assumptions_used"]
        if ass:
            rows = [{"Input": H.FIELD_LABELS.get(k, k), "Value used": H.fmt_value(v["value"]) if v["value"] is not None else ui.DASH,
                     "Source": v["source"]} for k, v in ass.items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", key="pred_assumptions")
        else:
            st.caption("No defaults were applied: every model input came from your entries.")
        with st.expander("Exactly what the model saw"):
            feats = pd.DataFrame([{"Feature": charts.pretty_feature_name(k), "Value": (ui.fmt_num(v, 2) if isinstance(v, float) else str(v))}
                                  for k, v in res["derived_features"].items()])
            st.dataframe(feats, hide_index=True, width="stretch", key="pred_features_seen")
            st.caption("Includes features derived from your entries (freight ratio, same-state flag, purchase weekday).")
    with c2:
        ui.section_header("Model in use", None, icon="cpu")
        cal = f"calibrated ({info['calibration_method']})" if info["calibrated"] and info["calibration_method"] else (
            "calibrated" if info["calibrated"] else "not calibrated")
        pills([f"{info['name']} · {cal}",
               f"Trained on {ui.fmt_int(info['train_n'])} orders, {info['train_start']} to {info['train_end']}",
               f"Test ROC-AUC {ui.fmt_num(info['test_roc_auc'], 2)}",
               f"Test PR-AUC {ui.fmt_num(info['test_pr_auc'], 3)} (no-skill {ui.fmt_num(info['test_baseline'], 3)})"])
        st.caption("Test = the most recent window, untouched during model building. Accuracy is deliberately not shown as a headline: "
                   "only a small share of orders is late, so it would be misleading.")



def render_scenarios() -> None:
    scen = st.session_state.get(K_SCEN, [])
    if not scen:
        return
    ui.section_header("3. Compare scenarios", "Pinned results side by side: model estimates for hypothetical orders.", icon="bar-chart")
    st.plotly_chart(charts.scenario_comparison_chart(scen, thresholds), width="stretch", key="pred_scenario_chart",
                    config=charts.PLOTLY_CONFIG)
    tbl = pd.DataFrame([{"Scenario": s["label"], "Order": s["description"], "Estimated probability": ui.fmt_pct(s["probability"]),
                         "Risk band": s["risk_label"]} for s in scen])
    st.dataframe(tbl, hide_index=True, width="stretch", key="pred_scenario_table", column_config={
        "Scenario": st.column_config.TextColumn(width="small"), "Order": st.column_config.TextColumn(width="large"),
        "Estimated probability": st.column_config.TextColumn(width="small"),
        "Risk band": st.column_config.TextColumn(width="small")})
    st.button("Clear scenarios", key="pred_clear", on_click=clear_scenarios, icon=":material/delete:")


with tab_one:
    ui.section_header("1. Describe the order",
                      "Everything here is known at the moment of purchase. Required fields are the route, category, items, value and freight.",
                      icon="package")
    seed_form(st.session_state.get(K_LAST), defaults)
    state_opts_c, state_opts_s = list(opts["customer_state"]), list(opts["seller_state"])
    r1, r2 = st.columns(2)
    with r1:
        customer_state = st.selectbox("Customer state", state_opts_c, key="pred_customer_state", format_func=ui.state_label,
                                      help="Where the order will be delivered.")
    with r2:
        seller_state = st.selectbox("Seller state", state_opts_s, key="pred_seller_state", format_func=ui.state_label,
                                    help="Where the (main) seller ships from.")
    promise_default, promise_src = H.pair_promise(bundle, seller_state, customer_state)
    promise_key = f"pred_promise_{seller_state}_{customer_state}"
    seed_state(promise_key, promise_default)
    pair_txt = f"{seller_state} to {customer_state}"
    st.html(f'<div class="pred-note">Route {ui.esc(pair_txt)}: the delivery promise below is prefilled with the median promise of '
            + ("training orders on this route." if promise_src == "pair" else
               "all training orders (this route has too few training orders for its own median).") + "</div>")

    slots: dict = {}  # per-field error placeholders
    top = H.top_features(metrics, "validation", 1)
    with st.form("pred_form", border=True):
        a, b, c = st.columns(3)
        with a:
            category = st.selectbox("Product category", list(opts["product_category"]), key="pred_category", format_func=ui.category_label,
                                    help="Type to search. Category of the highest-priced item in the order.")
            slots["product_category"] = st.empty()
            total_price = st.number_input("Total product value (R$)", value=None, step=10.0, format="%.2f",
                                          key="pred_price", placeholder="e.g. 120.00", help="Sum of item prices, in Brazilian reais.")
            slots["total_price"] = st.empty()
        with b:
            purchase_date = st.date_input("Purchase date", min_value=date(2015, 1, 1),
                                          max_value=date(2035, 12, 31), key="pred_date", format="YYYY-MM-DD",
                                          help="The model only learned the day-of-week pattern of its training window "
                                               f"({opts.get('date_span', ['?', '?'])[0]} to {opts.get('date_span', ['?', '?'])[1]}); "
                                               "dates outside it trigger a warning.")
            slots["purchase_date"] = st.empty()
            total_freight = st.number_input("Total freight value (R$)", value=None, step=1.0, format="%.2f",
                                            key="pred_freight", placeholder="e.g. 18.50", help="Shipping fee charged to the customer, in reais.")
            slots["total_freight"] = st.empty()
        with c:
            n_items = st.number_input("Number of items", value=None, step=1, key="pred_items",
                                      help="Items in the order (whole number, at least 1).")
            slots["n_items"] = st.empty()
            weight_def = (opts.get("defaults") or {}).get("total_weight_g")
            total_weight = st.number_input("Total product weight (g)", value=None, step=100.0, format="%.0f", key="pred_weight",
                                           placeholder=f"typical: {weight_def:,.0f}" if weight_def else "optional",
                                           help="Optional. Left blank, the training-window median is used and listed under assumptions.")
            slots["total_weight_g"] = st.empty()
        promise = st.number_input(
            "Estimated delivery window (days)", value=None, step=1, key=promise_key,
            help="The delivery promise shown to the customer at checkout: days from purchase to the estimated delivery date. "
                 + ("It is the strongest predictor in this model. " if top == ["estimated_delivery_days"] else "")
                 + "Prefilled for the route; edit it to match your order. Left blank, the route median is used.")
        slots["estimated_delivery_days"] = st.empty()
        with st.expander("Advanced inputs (optional)"):
            st.caption("Leave blank to use training-window medians (listed under 'Assumptions used' in the result).")
            d1, d2, d3 = st.columns(3)
            dflt = opts.get("defaults") or {}

            def ph(k: str) -> str:
                return f"typical: {dflt[k]:,.0f}" if k in dflt else "optional"

            with d1:
                n_sellers = st.number_input("Number of sellers", value=None, step=1, key="pred_n_sellers", placeholder=ph("n_sellers"))
                slots["n_sellers"] = st.empty()
                volume = st.number_input("Package volume (cm³)", value=None, step=100.0, format="%.0f", key="pred_volume",
                                         placeholder=ph("total_volume_cm3"))
                slots["total_volume_cm3"] = st.empty()
            with d2:
                length = st.number_input("Longest product length (cm)", value=None, step=1.0, format="%.0f", key="pred_length",
                                         placeholder=ph("max_product_length_cm"))
                slots["max_product_length_cm"] = st.empty()
                height = st.number_input("Tallest product height (cm)", value=None, step=1.0, format="%.0f", key="pred_height",
                                         placeholder=ph("max_product_height_cm"))
                slots["max_product_height_cm"] = st.empty()
            with d3:
                width = st.number_input("Widest product width (cm)", value=None, step=1.0, format="%.0f", key="pred_width",
                                        placeholder=ph("max_product_width_cm"))
                slots["max_product_width_cm"] = st.empty()
        submitted = st.form_submit_button("Predict Delay Risk", type="primary", icon=":material/psychology:")

    def collect_inputs() -> dict:
        """The order described by the form widgets (optional fields left blank are omitted)."""
        vals = {"customer_state": customer_state, "seller_state": seller_state, "product_category": category,
                "n_items": n_items, "total_price": total_price, "total_freight": total_freight, "purchase_date": purchase_date}
        for k, v in (("total_weight_g", total_weight), ("estimated_delivery_days", promise), ("n_sellers", n_sellers),
                     ("total_volume_cm3", volume), ("max_product_length_cm", length), ("max_product_height_cm", height),
                     ("max_product_width_cm", width)):
            if v is not None:
                vals[k] = v
        return vals

    scroll_to_result = False
    if submitted:
        inputs = collect_inputs()
        _cleaned, errors, _warns = prediction.check_input(inputs, bundle)
        if errors:
            st.session_state.pop(K_LAST, None)
            by_field, general = H.field_errors(errors)
            for f, msgs in by_field.items():
                if f in slots:
                    with slots[f]:
                        st.html('<div class="pred-field-err">' + "<br>".join(ui.esc(m) for m in msgs) + "</div>")
                else:
                    general.extend(msgs)
            ui.callout("alert", "Please fix the highlighted input(s) and try again",
                       "No prediction was made because some inputs are missing or not valid. " + (bullets(general) if general else
                                                                                                    "The messages are shown under each field."))
        else:
            try:
                res = prediction.predict_risk(inputs, bundle)
            except ValueError as exc:
                st.session_state.pop(K_LAST, None)
                ui.callout("alert", "The prediction could not be made", str(exc))
            else:
                key = repr(sorted((k, str(v)) for k, v in inputs.items()))
                st.session_state[K_LAST] = {"inputs": inputs, "result": res, "key": key, "description": H.describe_order(
                    {**inputs, "estimated_delivery_days": (res["derived_features"] or {}).get("estimated_delivery_days")})}
                scroll_to_result = True

    last = st.session_state.get(K_LAST)
    if last:
        with st.container(key="pred-result"):
            if scroll_to_result:
                with st.spinner("Scoring this order and preparing the result…"):
                    render_result(last, collect_inputs())
            else:
                render_result(last, collect_inputs())
        if scroll_to_result:     # on a phone the result starts far below the button: bring it into view once
            st.html("<script>" + SCROLL_JS + "</script>", unsafe_allow_javascript=True)
    else:
        ui.empty_state("No prediction yet", "Fill in the order above and press **Predict Delay Risk**.", icon="target")
    render_scenarios()


# --------------------------------------------------------------------------- #
# TAB 2: batch scoring
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Scoring orders…", max_entries=4)
def score_upload(data: bytes, _bundle: dict, model_token: str) -> dict:
    """Parse + score an uploaded CSV (cached on the file bytes and the model file version)."""
    try:
        parsed, notes = H.parse_batch_csv(data)
    except H.BatchFileError as exc:
        return {"error": str(exc)}
    scored, summary = H.score_batch(parsed, _bundle)
    return {"scored": scored, "summary": summary, "notes": notes, "invalid": H.invalid_rows_report(scored),
            "csv": scored.drop(columns=["csv_row"]).to_csv(index=False).encode("utf-8")}


@st.cache_data(show_spinner=False)
def template_bytes(_df: pd.DataFrame, _bundle: dict, token: str) -> bytes:
    return H.template_csv(_df, _bundle)


with tab_batch:
    ui.section_header("Score many orders from a CSV file",
                      "Upload one row per hypothetical order. Each row is checked and scored on its own; rows with problems are "
                      "listed with the reason and never silently fixed.", icon="database")
    token = f"{config.MODEL_PATH.stat().st_mtime_ns if config.MODEL_PATH.exists() else 0}-{len(df_all)}"
    ui.callout("info", "File format",
               "Required columns: " + ", ".join(f"`{c}`" for c in H.BATCH_REQUIRED) + ".\n"
               "Optional columns (blank = training-window median, reported per row): " + ", ".join(f"`{c}`" for c in H.BATCH_OPTIONAL)
               + ".\n`order_ref` is an optional label of your own. Dates are `YYYY-MM-DD`; states are two-letter codes "
                 f"(e.g. SP); categories are the model's English names. Up to {H.MAX_BATCH_ROWS:,} rows / {H.MAX_BATCH_BYTES // 1_000_000} MB. "
                 "Other columns are ignored and not exported: do not upload names or addresses.")
    st.download_button("Download template (EXAMPLE rows)", data=template_bytes(df_all, bundle, token), file_name="deliveryiq_batch_template.csv",
                       mime="text/csv", key="pred_template_dl", icon=":material/download:",
                       help="The exact headers, with three rows labelled EXAMPLE built from the most common routes in the data.")
    up = st.file_uploader("Upload your CSV", type=["csv"], key="pred_batch_file", accept_multiple_files=False)
    if up is None:
        ui.empty_state("No file uploaded yet", "Download the template, replace the EXAMPLE rows with your orders and upload it here.", icon="inbox")
    else:
        out = score_upload(up.getvalue(), bundle, token)
        if "error" in out:
            ui.friendly_error("This file cannot be used", out["error"])
        else:
            s = out["summary"]
            for n in out["notes"]:
                ui.callout("info", None, n)
            ui.kpi_row([
                dict(label="Rows in file", value=ui.fmt_int(s["n_rows"]), icon="database", accent="blue"),
                dict(label="Scored", value=ui.fmt_int(s["n_valid"]), sublabel=f"mean probability {ui.fmt_pct(s['mean_probability'])}",
                     icon="check", accent="green"),
                dict(label="Rows with problems", value=ui.fmt_int(s["n_invalid"]), sublabel="not scored, see the list below",
                     icon="alert", accent="red" if s["n_invalid"] else "slate"),
                dict(label="High / Moderate / Low", value=f"{s['bands']['High']:,} / {s['bands']['Moderate']:,} / {s['bands']['Low']:,}",
                     sublabel="scored rows by predicted band", icon="layers", accent="amber"),
            ])
            if s["n_assumed"]:
                st.caption(f"{s['n_assumed']:,} scored row(s) used training-window medians for blank optional columns (column `assumptions_used`).")
            if s["n_warnings"]:
                ui.callout("watch", f"{s['n_warnings']:,} row(s) outside the training range",
                           "The model extrapolates for these rows (see column `input_warning`): treat their estimates with extra caution.")
            if s["n_invalid"]:
                ui.section_header("Rows that could not be scored", f"{s['n_invalid']:,} row(s). Row numbers are file lines (the header is line 1).", icon="alert")
                inv = out["invalid"]
                st.dataframe(inv.head(500), hide_index=True, width="stretch", key="pred_invalid_rows", column_config={
                    "csv_row": st.column_config.NumberColumn("File row", format="%d", width="small"),
                    "problem": st.column_config.TextColumn("Problem", width="large")})
                if len(inv) > 500:
                    st.caption(f"Showing the first 500 of {len(inv):,} problems.")
            if s["n_valid"] == 0:
                ui.empty_state("No row could be scored", "Fix the problems listed above and upload the file again.", icon="alert")
            else:
                ui.section_header("Scored orders", "Preview of the first 200 scored rows. The download holds every row, including the ones that could not be scored.", icon="file-text")
                show = H.batch_preview(out["scored"], 200)
                st.dataframe(show, hide_index=True, width="stretch", key="pred_batch_preview", column_config={
                    "csv_row": st.column_config.NumberColumn("File row", format="%d"),
                    "late_probability": st.column_config.NumberColumn("Late probability", format="percent",
                                                                      help="Model estimate, not a certainty."),
                    "risk_label": st.column_config.TextColumn("Risk band"),
                    "assumptions_used": st.column_config.TextColumn("Defaults used"),
                    "input_warning": st.column_config.TextColumn("Warning"),
                })
                st.download_button("Download scored CSV", data=out["csv"], file_name="deliveryiq_scored_orders.csv", mime="text/csv",
                                   key="pred_scored_dl", icon=":material/download:", type="primary")
                ui.callout("watch", "Use the scores with care", H.read_first(bundle, metrics))


# --------------------------------------------------------------------------- #
# TAB 3: model card
# --------------------------------------------------------------------------- #
with tab_model:
    ui.section_header("Model card", "What the model is, how it was checked, and where it falls short.", icon="cpu")
    cal = f"calibrated ({info['calibration_method']})" if info["calibrated"] and info["calibration_method"] else (
        "calibrated" if info["calibrated"] else "not calibrated")
    ci = info["test_roc_ci"]
    pci = info["test_pr_ci"]
    ui.kpi_row([
        dict(label="Model", value=info["name"], sublabel=cal.capitalize() + f"; trained {info['train_start']} to {info['train_end']}",
             icon="cpu", accent="blue"),
        dict(label="Training orders", value=ui.fmt_int(info["train_n"]), sublabel=f"late rate {ui.fmt_pct((info['windows'].get('train') or {}).get('late_rate'))}",
             icon="database", accent="slate"),
        dict(label="Test ROC-AUC", value=ui.fmt_num(info["test_roc_auc"], 2),
             sublabel=(f"95% CI {ci[0]:.2f} to {ci[1]:.2f} · 0.5 = no skill" if isinstance(ci, (list, tuple)) and len(ci) == 2 else "0.5 = no skill")
             + (f" · validation {ui.fmt_num(info['val_roc_auc'], 2)}" if info["val_roc_auc"] is not None else ""),
             icon="trend", accent="amber"),
        dict(label="Test PR-AUC", value=ui.fmt_num(info["test_pr_auc"], 3),
             sublabel=f"no-skill baseline {ui.fmt_num(info['test_baseline'], 3)}"
             + (f" · 95% CI {pci[0]:.3f} to {pci[1]:.3f}" if isinstance(pci, (list, tuple)) and len(pci) == 2 else ""),
             icon="target", accent="amber"),
    ])
    ui.callout("watch", "Limitations: please read", bullets(H.limitations(bundle, metrics)))

    w = info["windows"]
    if w:
        ui.section_header("The late rate shifted between windows",
                          "The model is trained on the earliest window and judged on later ones (chronological split, so no future data leaks into training).",
                          icon="calendar")
        names = {"train": "Training (model fitting)", "validation": "Validation (tuning, risk-band cuts)", "test": "Test (untouched until the end)"}
        shift = pd.DataFrame([{"Window": names[k], "Purchase dates": f"{v['start']} to {v['end']}", "Orders": ui.fmt_int(v["n"]),
                               "Late orders": ui.fmt_int(v["n_late"]), "Late rate": ui.fmt_pct(v["late_rate"])} for k, v in w.items()])
        st.dataframe(shift, hide_index=True, width="stretch", key="pred_windows_table")

    ui.section_header("What changes the risk?", "Permutation importance: how much the model's PR-AUC drops when one input is shuffled.",
                      icon="bar-chart")
    win = st.radio("Evaluation window", ["validation", "test"], horizontal=True, key="pred_imp_window",
                   format_func=lambda x: {"validation": "Validation window (also used for tuning)", "test": "Test window (out of sample)"}[x])
    st.plotly_chart(charts.feature_importance_chart(metrics, window=win), width="stretch", key="pred_importance", config=charts.PLOTLY_CONFIG)
    tv, tt = H.top_features(metrics, "validation"), H.top_features(metrics, "test")
    if tv:
        txt = "Largest drops on the validation window: " + ", ".join(charts.pretty_feature_name(f) for f in tv) + "."
        if tt:
            txt += " On the test window: " + ", ".join(charts.pretty_feature_name(f) for f in tt) + "."
        cmp_note = (metrics.get("permutation_importance") or {}).get("comparison_note")
        ui.callout("info", "Statistical association, not cause",
                   txt + " These are predictive associations in historical data: shuffling a column shows what the model leans on, not "
                         "what makes an order late, and the ranking can change between windows."
                   + (f"\n{H.friendly_feature_text(cmp_note)}" if cmp_note else ""))
    else:
        ui.empty_state("No importance values stored", "Train the model: `.venv/bin/python -m src.train_model`.", icon="bar-chart")

    ui.section_header("Do the risk bands hold up?", "Observed late rate per predicted band on the validation and test windows.", icon="layers")
    st.plotly_chart(charts.risk_band_chart(metrics), width="stretch", key="pred_band_chart", config=charts.PLOTLY_CONFIG)
    st.dataframe(band_display(H.band_table(thresholds, metrics, None), None), hide_index=True, width="stretch", key="pred_band_table_card")
    sep = H.band_separation_text(metrics)
    if sep:
        ui.callout("info", "How well do the bands separate?", sep)

    with st.expander("Calibration: do predicted probabilities match what happened?"):
        st.plotly_chart(charts.calibration_chart(metrics), width="stretch", key="pred_calibration", config=charts.PLOTLY_CONFIG)
        cav = bundle.get("test_window_caveat") or {}
        if cav:
            st.caption(f"Test window: mean predicted risk {ui.fmt_pct(cav.get('mean_predicted'))} vs {ui.fmt_pct(cav.get('observed_late_rate'))} observed "
                       f"({ui.fmt_n(cav.get('n'))}).")
