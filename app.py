"""DeliveryIQ — entry point (``streamlit run app.py``).

Builds the app shell: page config, CSS, hidden ``st.navigation`` over ``views/*.py`` and a
hand-rendered sidebar (brand, navigation, global filters, dataset info).  No raw CSV is read and
no model is trained here; everything comes from the cached loaders in ``src.ui``.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src import config  # noqa: E402   (light: paths and constants only)

st.set_page_config(
    page_title=f"{config.APP_NAME} · Shipment analytics",
    page_icon=":material/local_shipping:",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={"About": f"**{config.APP_NAME}** — {config.APP_TAGLINE}. Historical Olist 2016–2018 data; "
                         "descriptive analytics and a modest-signal risk model, not a causal study."},
)

# The first visit after a server start spends a few seconds importing the analysis libraries (pandas, scipy,
# scikit-learn, plotly).  Say so right away, instead of showing a blank page, and clear the notice as soon as they are in.
_boot = st.empty()
_boot.html(
    '<div role="status" style="margin:18vh auto 0;max-width:30rem;text-align:center;font-family:sans-serif;color:#0F172A">'
    f'<div style="font-weight:800;font-size:1.4rem">{config.APP_NAME}</div>'
    '<div style="margin-top:.5rem;color:#475569;line-height:1.5">Loading the analysis tools and the delivered-orders data. '
    'The first load after starting the app takes a few seconds.</div></div>')

from src import ui  # noqa: E402  (heavy imports: after the loading notice is on screen)

ui.inject_css()

PAGES = [
    st.Page("views/1_Overview.py", title="Overview", icon=":material/dashboard:", default=True),
    st.Page("views/2_Delay_Analysis.py", title="Delay Analysis", icon=":material/query_stats:", url_path="delay-analysis"),
    st.Page("views/3_Regional_Intelligence.py", title="Regional Intelligence", icon=":material/map:",
            url_path="regional-intelligence"),
    st.Page("views/4_Delay_Prediction.py", title="Delay Prediction", icon=":material/psychology:",
            url_path="delay-prediction"),
    st.Page("views/5_Reports_Insights.py", title="Reports & Insights", icon=":material/description:",
            url_path="reports-insights"),
]
page = st.navigation(PAGES, position="hidden")
_boot.empty()

if not ui.data_available():
    ui.render_sidebar(None)          # brand only
    ui.render_setup_page()
    st.stop()

ui.render_sidebar(PAGES, current=page)
page.run()
