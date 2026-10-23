"""DeliveryIQ — entry point (``streamlit run app.py``).

Builds the app shell: page config, CSS, hidden ``st.navigation`` over ``pages/*.py`` and a
hand-rendered sidebar (brand, navigation, global filters, dataset info).  No raw CSV is read and
no model is trained here; everything comes from the cached loaders in ``src.ui``.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src import config, ui  # noqa: E402

st.set_page_config(
    page_title=f"{config.APP_NAME} · Shipment analytics",
    page_icon=":material/local_shipping:",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={"About": f"**{config.APP_NAME}** — {config.APP_TAGLINE}. Historical Olist 2016–2018 data; "
                         "descriptive analytics and a modest-signal risk model, not a causal study."},
)
ui.inject_css()

PAGES = [
    st.Page("pages/1_Overview.py", title="Overview", icon=":material/dashboard:", url_path="overview", default=True),
    st.Page("pages/2_Delay_Analysis.py", title="Delay Analysis", icon=":material/query_stats:", url_path="delay-analysis"),
    st.Page("pages/3_Regional_Intelligence.py", title="Regional Intelligence", icon=":material/map:",
            url_path="regional-intelligence"),
    st.Page("pages/4_Delay_Prediction.py", title="Delay Prediction", icon=":material/psychology:",
            url_path="delay-prediction"),
    st.Page("pages/5_Reports_Insights.py", title="Reports & Insights", icon=":material/description:",
            url_path="reports-insights"),
]
page = st.navigation(PAGES, position="hidden")

if not ui.data_available():
    ui.render_sidebar(None)          # brand only
    ui.render_setup_page()
    st.stop()

ui.render_sidebar(PAGES, current=page)
page.run()
