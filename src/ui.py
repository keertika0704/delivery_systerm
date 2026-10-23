"""Streamlit UI toolkit for DeliveryIQ (the ONLY ``src`` module that imports streamlit).

Everything that pages share lives here: CSS, KPI cards, callouts, empty / error states, cached
data & model loaders, the global filter sidebar and number formatters.  Pages should call
:func:`get_filtered_orders` to obtain the data so that the sidebar filters apply everywhere, and
should never read CSV / parquet files or train models themselves.

Page-author cheat sheet (all HTML is escaped; pass plain text):

* ``inject_css()``                                   idempotent per run (safe to call from a page)
* ``page_header(title, subtitle, icon, eyebrow)``    page title block
* ``section_header(title, subtitle, icon)``          section title inside a page
* ``kpi_card(label, value, ...)`` / ``kpi_row(items)``  KPI tiles (responsive grid)
* ``callout(kind, title, body)`` / ``insight_box(insight)``  info | watch | alert | success boxes
* ``empty_state(message, hint)`` / ``friendly_error(title, detail, command)``
* ``get_filtered_orders() -> (df_all, df_filtered, filters)``
* ``get_orders() / get_quality_report() / get_model_bundle() / get_model_metrics()``  cached loaders
* ``fmt_int / fmt_pct / fmt_days / fmt_money_brl / fmt_num / fmt_pp / fmt_n``  NaN-safe ("—")
"""
from __future__ import annotations

import base64
import html as _html
import json
import math
import re
from datetime import date, datetime
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src.analytics import apply_filters

PAL = config.PALETTE
DASH = "—"

# Session-state keys of the global filters (stable; pages may read them but should prefer
# ``get_filtered_orders``).
K_DATE = "flt_date_range"
K_STATES = "flt_states"
K_CATEGORIES = "flt_categories"

PROCESS_COMMAND = ".venv/bin/python -m src.preprocessing"
TRAIN_COMMAND = ".venv/bin/python -m src.train_model"


# --------------------------------------------------------------------------- #
# Formatting helpers (all return "—" for missing / non-finite values)
# --------------------------------------------------------------------------- #
def _as_float(x: Any) -> float | None:
    """Convert to a finite float, or None for None / NaN / NA / inf / non-numeric."""
    if x is None or x is pd.NA or x is pd.NaT or isinstance(x, (str, bytes)):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def fmt_int(x: Any) -> str:
    """Integer with thousands separators: 96470 -> '96,470'."""
    v = _as_float(x)
    return DASH if v is None else f"{int(round(v)):,}"


def fmt_num(x: Any, digits: int = 2) -> str:
    """Decimal number with thousands separators."""
    v = _as_float(x)
    return DASH if v is None else f"{v:,.{digits}f}"


def fmt_pct(x: Any, digits: int = 1) -> str:
    """Format a FRACTION (0.0677) as a percentage ('6.8%')."""
    v = _as_float(x)
    return DASH if v is None else f"{100 * v:.{digits}f}%"


def fmt_pp(x: Any, digits: int = 1) -> str:
    """Format a difference of two fractions as signed percentage points ('+1.2 pp')."""
    v = _as_float(x)
    return DASH if v is None else f"{100 * v:+.{digits}f} pp"


def fmt_days(x: Any, digits: int = 1) -> str:
    """Duration in days: 12.34 -> '12.3 days'."""
    v = _as_float(x)
    return DASH if v is None else f"{v:,.{digits}f} days"


def fmt_money_brl(x: Any, digits: int = 2) -> str:
    """Brazilian reais: 1234.5 -> 'R$ 1,234.50'."""
    v = _as_float(x)
    if v is None:
        return DASH
    sign = "-" if v < 0 else ""
    return f"{sign}R$ {abs(v):,.{digits}f}"


def fmt_n(n: Any) -> str:
    """Sample size label: 1234 -> 'n = 1,234'."""
    v = _as_float(n)
    return "n = " + (DASH if v is None else f"{int(round(v)):,}")


def esc(x: Any) -> str:
    """HTML-escape any value (None -> '')."""
    return "" if x is None else _html.escape(str(x), quote=True)


def _inline_md(text: Any) -> str:
    """Escape text and support **bold**, `code` and newlines (nothing else)."""
    s = esc(text)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    return s.replace("\n", "<br>")


def state_label(code: Any) -> str:
    """'SP' -> 'SP — São Paulo' (unknown codes are returned unchanged)."""
    c = str(code)
    name = config.STATE_NAMES.get(c)
    return f"{c} — {name}" if name else c


def category_label(cat: Any) -> str:
    """'health_beauty' -> 'Health beauty'."""
    s = str(cat).replace("_", " ").strip()
    return s[:1].upper() + s[1:] if s else s


# --------------------------------------------------------------------------- #
# Icons (inline SVG, Lucide-style strokes) so icons look identical everywhere
# --------------------------------------------------------------------------- #
_ICONS: dict[str, str] = {
    "package": '<path d="M16.5 9.4 7.5 4.21"/><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><path d="m3.27 6.96 8.73 5.05 8.73-5.05"/><path d="M12 22.08V12"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
    "check": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="m22 4-10 10.01-3-3.01"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "x-circle": '<circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/>',
    "truck": '<path d="M14 18V6a2 2 0 0 0-2-2H4a2 2 0 0 0-2 2v11a1 1 0 0 0 1 1h2"/><path d="M15 18H9"/><path d="M19 18h2a1 1 0 0 0 1-1v-3.65a1 1 0 0 0-.22-.62l-3.48-4.35A1 1 0 0 0 17.52 8H14"/><circle cx="17" cy="18" r="2"/><circle cx="7" cy="18" r="2"/>',
    "map-pin": '<path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/>',
    "map": '<path d="M1 6v16l7-4 8 4 7-4V2l-7 4-8-4z"/><path d="M8 2v16"/><path d="M16 6v16"/>',
    "calendar": '<rect width="18" height="18" x="3" y="4" rx="2"/><path d="M16 2v4"/><path d="M8 2v4"/><path d="M3 10h18"/>',
    "dollar": '<path d="M12 2v20"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',
    "trend": '<path d="m22 7-8.5 8.5-5-5L2 17"/><path d="M16 7h6v6"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "percent": '<path d="M19 5 5 19"/><circle cx="6.5" cy="6.5" r="2.5"/><circle cx="17.5" cy="17.5" r="2.5"/>',
    "bar-chart": '<path d="M12 20V10"/><path d="M18 20V4"/><path d="M6 20v-4"/>',
    "inbox": '<path d="M22 12h-6l-2 3h-4l-2-3H2"/><path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>',
    "cpu": '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "filter": '<path d="M22 3H2l8 9.46V19l4 2v-8.54z"/>',
    "file-text": '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/>',
    "settings": '<path d="M12 2v4"/><path d="m16.2 7.8 2.9-2.9"/><path d="M18 12h4"/><path d="m16.2 16.2 2.9 2.9"/><path d="M12 18v4"/><path d="m4.9 19.1 2.9-2.9"/><path d="M2 12h4"/><path d="m4.9 4.9 2.9 2.9"/>',
    "layers": '<path d="m12 2 10 5-10 5L2 7z"/><path d="m2 17 10 5 10-5"/><path d="m2 12 10 5 10-5"/>',
}


def icon_svg(name: str | None, size: int = 18) -> str:
    """Inline icon element for a known icon name; unknown names are rendered as escaped text (e.g. an emoji).

    ``st.html`` strips ``<svg>`` tags, so icons are CSS masks (classes generated in the stylesheet)
    that inherit the current text colour.
    """
    if not name:
        return ""
    if name not in _ICONS:
        return f'<span class="diq-emoji">{esc(name)}</span>'
    return f'<span class="diq-i diq-i-{name}" style="width:{size}px;height:{size}px" aria-hidden="true"></span>'


def _icon_css() -> str:
    rules = []
    for name, body in _ICONS.items():
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" '
               f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{body}</svg>')
        url = "url(data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode() + ")"
        rules.append(f".diq-i-{name}{{-webkit-mask-image:{url};mask-image:{url};}}")
    return (".diq-i{display:inline-block; flex:0 0 auto; background-color:currentColor; -webkit-mask-repeat:no-repeat;"
            "mask-repeat:no-repeat; -webkit-mask-position:center; mask-position:center; -webkit-mask-size:contain;"
            "mask-size:contain; vertical-align:middle;}\n" + "\n".join(rules))


_LOGO_RAW = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" width="40" height="40">'
    '<rect width="48" height="48" rx="12" fill="#2563EB"/>'
    '<path d="M12 17.5 24 11l12 6.5v13L24 37l-12-6.5z" fill="none" stroke="#fff" stroke-width="2.4" stroke-linejoin="round"/>'
    '<path d="M12 17.5 24 24l12-6.5M24 24v13" fill="none" stroke="#fff" stroke-width="2.4" stroke-linejoin="round"/>'
    '<circle cx="35.5" cy="12.5" r="4.2" fill="#16A34A" stroke="#2563EB" stroke-width="2"/></svg>'
)
LOGO_HTML = ('<img src="data:image/svg+xml;base64,' + base64.b64encode(_LOGO_RAW.encode()).decode()
             + '" width="40" height="40" alt="DeliveryIQ logo">')

_ACCENTS = {
    "blue": (PAL["blue"], "#EFF6FF"),
    "green": (PAL["green"], "#F0FDF4"),
    "red": (PAL["red"], "#FEF2F2"),
    "amber": ("#D97706", "#FFFBEB"),
    "slate": (PAL["slate"], "#F1F5F9"),
}

# --------------------------------------------------------------------------- #
# CSS
# --------------------------------------------------------------------------- #
_CSS = """
:root{
  --navy:__navy__; --slate:__slate__; --slate-light:__slate_light__; --border:__border__;
  --bg:__background__; --card:__card__; --blue:__blue__; --green:__green__; --amber:__amber__; --red:__red__;
  --slate-text:#526071;
  --shadow:0 1px 2px rgba(15,23,42,.04), 0 4px 14px rgba(15,23,42,.05);
  --radius:14px;
}
.stApp, .stApp p, .stApp label, .stApp li, .stApp input, .stApp textarea, .stApp button,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp td, .stApp th,
.stApp [data-baseweb], [data-testid="stSidebar"] :is(div,p,span,a,label):not([data-testid="stIconMaterial"]){
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
.stApp{background:var(--bg);}

/* ---- chrome ---- */
footer, [data-testid="stFooter"], [data-testid="stDecoration"], [data-testid="stStatusWidget"]{display:none !important;}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stMainBlockContainer"], .block-container{max-width:1320px; padding:1.6rem 2.4rem 4rem;}
@media (max-width:640px){
  [data-testid="stMainBlockContainer"], .block-container{padding:3.4rem 1rem 3rem;}
}

/* ---- typography ---- */
.stApp h1{font-weight:800; letter-spacing:-.02em; color:var(--navy);}
.stApp h2, .stApp h3{font-weight:700; letter-spacing:-.01em; color:var(--navy);}
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p{color:#1E293B; line-height:1.6;}
[data-testid="stMain"] [data-testid="stCaptionContainer"]{color:var(--slate-text); opacity:1 !important;}

/* ---- page + section headers ---- */
.diq-page{margin:.2rem 0 1.2rem;}
.diq-eyebrow{font-size:.72rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase; color:var(--blue);}
.diq-title{display:flex; align-items:center; gap:.65rem; font-size:2rem; line-height:1.15; font-weight:800;
  letter-spacing:-.025em; color:var(--navy); margin:.15rem 0 .35rem; padding:0;}
.diq-title .diq-title-ico{display:inline-flex; width:2.4rem; height:2.4rem; border-radius:12px; align-items:center;
  justify-content:center; background:#DBEAFE; color:var(--blue); flex:0 0 auto;}
.diq-subtitle{color:var(--slate-text); font-size:1rem; max-width:62rem; margin:0; line-height:1.5;}
.diq-section{margin:1.8rem 0 .7rem; padding-top:.2rem;}
.diq-section-title{display:flex; align-items:center; gap:.5rem; font-size:1.22rem; font-weight:700; color:var(--navy);
  margin:0; padding:0; letter-spacing:-.01em;}
.diq-section-title svg{color:var(--blue);}
.diq-section-sub{color:var(--slate-text); font-size:.92rem; margin:.2rem 0 0; line-height:1.5;}
.diq-pillrow{display:flex; flex-wrap:wrap; gap:.4rem; margin:.2rem 0 .4rem;}
.diq-pill{display:inline-flex; align-items:center; gap:.35rem; font-size:.78rem; font-weight:600; color:var(--slate);
  background:#fff; border:1px solid var(--border); border-radius:999px; padding:.18rem .65rem;}
.diq-pill b{color:var(--navy);}
.diq-pill.is-warn{background:#FFFBEB; border-color:#FDE68A; color:#92400E;}

/* ---- KPI cards ---- */
.diq-kpi-grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:1rem; margin:.4rem 0 1rem;}
.diq-kpi{position:relative; background:var(--card); border:1px solid var(--border); border-radius:var(--radius);
  padding:1rem 1.1rem .95rem; box-shadow:var(--shadow); overflow:hidden; min-width:0;}
.diq-kpi::before{content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:var(--accent);}
.diq-kpi-top{display:flex; align-items:center; justify-content:space-between; gap:.6rem;}
.diq-kpi-label{font-size:.78rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em; color:var(--slate-text);}
.diq-kpi-ico{display:inline-flex; width:2rem; height:2rem; border-radius:10px; align-items:center; justify-content:center;
  background:var(--accent-bg); color:var(--accent); flex:0 0 auto;}
.diq-kpi-value{font-size:1.85rem; line-height:1.15; font-weight:800; letter-spacing:-.02em; color:var(--navy);
  margin:.45rem 0 .15rem; font-variant-numeric:tabular-nums; overflow-wrap:anywhere;}
.diq-kpi-sub{font-size:.8rem; color:var(--slate-text); line-height:1.35;}
.diq-kpi-delta{display:inline-block; margin-top:.25rem; font-size:.78rem; font-weight:700; padding:.08rem .45rem;
  border-radius:999px; background:#F1F5F9; color:var(--slate);}
.diq-kpi-delta.good{background:#DCFCE7; color:#166534;}
.diq-kpi-delta.bad{background:#FEE2E2; color:#991B1B;}

/* ---- callouts ---- */
.diq-callout{display:flex; gap:.8rem; align-items:flex-start; border-radius:12px; border:1px solid var(--c-border);
  background:var(--c-bg); padding:.85rem 1rem; margin:.5rem 0 .8rem; border-left:4px solid var(--c);}
.diq-callout-ico{color:var(--c); flex:0 0 auto; margin-top:.1rem;}
.diq-callout-body{min-width:0;}
.diq-callout-title{font-weight:700; color:var(--navy); font-size:.96rem; margin-bottom:.15rem;}
.diq-callout-text{color:#334155; font-size:.92rem; line-height:1.55;}
.diq-callout-foot{color:var(--slate-text); font-size:.78rem; margin-top:.3rem;}
.diq-callout code, .diq-empty code{background:rgba(15,23,42,.06); padding:.05rem .3rem; border-radius:5px; font-size:.85em;}
.diq-callout.info{--c:#2563EB; --c-bg:#EFF6FF; --c-border:#DBEAFE;}
.diq-callout.watch{--c:#D97706; --c-bg:#FFFBEB; --c-border:#FDE68A;}
.diq-callout.alert{--c:#DC2626; --c-bg:#FEF2F2; --c-border:#FECACA;}
.diq-callout.success{--c:#16A34A; --c-bg:#F0FDF4; --c-border:#BBF7D0;}
.diq-callout.action{--c:#047857; --c-bg:#ECFDF5; --c-border:#A7F3D0;}

/* ---- empty + error states ---- */
.diq-empty{text-align:center; background:#fff; border:1px dashed #CBD5E1; border-radius:var(--radius);
  padding:2rem 1.2rem; margin:.6rem 0 1rem; color:var(--slate);}
.diq-empty .ico{display:inline-flex; width:3rem; height:3rem; border-radius:50%; background:#F1F5F9; color:var(--slate-text);
  align-items:center; justify-content:center; margin-bottom:.6rem;}
.diq-empty .msg{font-weight:700; color:var(--navy); font-size:1.02rem;}
.diq-empty .hint{color:var(--slate-text); font-size:.9rem; margin-top:.25rem;}
.diq-error{background:#fff; border:1px solid #FECACA; border-left:5px solid var(--red); border-radius:var(--radius);
  padding:1.1rem 1.3rem; margin:.8rem 0 .4rem; box-shadow:var(--shadow);}
.diq-error .t{display:flex; gap:.5rem; align-items:center; font-weight:800; font-size:1.1rem; color:#991B1B;}
.diq-error .d{color:#334155; margin-top:.35rem; line-height:1.55; font-size:.95rem;}

/* ---- sidebar ---- */
[data-testid="stSidebar"]{background:var(--navy); border-right:1px solid #1E293B; min-width:300px;}
[data-testid="stSidebarContent"]{padding:0 .3rem;}
[data-testid="stSidebar"] [data-testid="stSidebarHeader"]{padding:.25rem .5rem 0; min-height:0; height:auto; margin-bottom:-1.2rem;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:.6rem;}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]{padding:0 1rem 1.5rem;}
.diq-brand{display:flex; align-items:center; gap:.7rem; padding:.2rem 0 .8rem; border-bottom:1px solid #1E293B;}
.diq-brand-name{font-size:1.3rem; font-weight:800; letter-spacing:-.02em; color:#fff; line-height:1.1;}
.diq-brand-tag{font-size:.7rem; color:#94A3B8; line-height:1.3; margin-top:.15rem;}
.diq-side-label{font-size:.68rem; font-weight:700; letter-spacing:.11em; text-transform:uppercase; color:#94A3B8;
  margin:.5rem 0 -.2rem;}
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]{border-radius:10px; padding:.5rem .75rem; color:#CBD5E1;
  transition:background .12s;}
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] p{color:inherit; font-weight:600; font-size:.93rem;}
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover{background:#1E293B; color:#fff;}
[data-testid="stSidebar"] .st-key-nav-active [data-testid="stPageLink-NavLink"]{background:var(--blue); color:#fff;}
[data-testid="stSidebar"] .st-key-nav-active [data-testid="stPageLink-NavLink"] p,
[data-testid="stSidebar"] .st-key-nav-active [data-testid="stPageLink-NavLink"] span{color:#fff;}
[data-testid="stSidebar"] label p, [data-testid="stSidebar"] label{color:#CBD5E1 !important; font-size:.82rem; font-weight:600;}
[data-testid="stSidebar"] [data-baseweb="input"], [data-testid="stSidebar"] [data-baseweb="select"] > div{
  background:#1E293B !important; border-color:#334155 !important; border-radius:10px;}
[data-testid="stSidebar"] input{color:#F1F5F9 !important; -webkit-text-fill-color:#F1F5F9;}
[data-testid="stSidebar"] [data-tag]{background:#1D4ED8 !important; color:#fff !important; border-radius:7px;}
[data-testid="stSidebar"] [data-tag] span, [data-testid="stSidebar"] [data-tag] button{color:#fff !important;}
[data-testid="stSidebar"] .stButton button{background:transparent; color:#CBD5E1; border:1px solid #334155; border-radius:10px;
  font-weight:600; width:100%;}
[data-testid="stSidebar"] .stButton button:hover{background:#1E293B; color:#fff; border-color:#475569;}
.diq-counter{background:#1E293B; border:1px solid #334155; border-radius:12px; padding:.7rem .85rem;}
.diq-counter .n{font-size:1.15rem; font-weight:800; color:#fff; font-variant-numeric:tabular-nums;}
.diq-counter .of{font-size:.8rem; color:#94A3B8;}
.diq-counter .bar{height:6px; background:#334155; border-radius:999px; margin-top:.5rem; overflow:hidden;}
.diq-counter .bar > span{display:block; height:100%; background:var(--blue); border-radius:999px;}
.diq-counter.zero{border-color:#B45309; background:#2A1F0B;}
.diq-counter.zero .n{color:#FCD34D;}
.diq-counter .warn{color:#FCD34D; font-size:.78rem; margin-top:.4rem; line-height:1.4;}
.diq-info{background:#111C33; border:1px solid #1E293B; border-radius:12px; padding:.8rem .9rem; font-size:.78rem; color:#94A3B8;}
.diq-info .row{display:flex; justify-content:space-between; gap:.6rem; padding:.18rem 0;}
.diq-info .k{color:#94A3B8; flex:0 0 auto;}
.diq-info .v{color:#E2E8F0; text-align:right; font-weight:600; min-width:0;}
.diq-info .hd{font-size:.68rem; font-weight:700; letter-spacing:.11em; text-transform:uppercase; color:#94A3B8; margin-bottom:.3rem;}
.diq-dot{display:inline-block; width:.5rem; height:.5rem; border-radius:50%; margin-right:.35rem; vertical-align:middle;}

/* ---- widgets in the main area ---- */
[data-testid="stMain"] .stButton button, [data-testid="stMain"] .stDownloadButton button,
[data-testid="stMain"] [data-testid="stFormSubmitButton"] button{border-radius:10px; font-weight:700; padding:.5rem 1.1rem;}
[data-testid="stMain"] button[kind="primary"], [data-testid="stMain"] button[kind="primaryFormSubmit"]{
  background:var(--blue); border-color:var(--blue); color:#fff;}
[data-testid="stMain"] button[kind="primary"]:hover, [data-testid="stMain"] button[kind="primaryFormSubmit"]:hover{
  background:#1D4ED8; border-color:#1D4ED8;}
[data-testid="stMain"] [data-baseweb="select"] > div, [data-testid="stMain"] [data-baseweb="input"]{border-radius:10px;}

/* ---- containers, charts, tables, tabs ---- */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]){}
[data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"][class*="stVerticalBlockBorderWrapper"],
[data-testid="stMain"] div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:var(--radius);}
[data-testid="stPlotlyChart"]{background:#fff; border:1px solid var(--border); border-radius:var(--radius);
  box-shadow:var(--shadow); padding:.5rem .5rem .2rem; overflow:hidden;}
[data-testid="stDataFrame"], [data-testid="stTable"]{border:1px solid var(--border); border-radius:12px; overflow:hidden;
  box-shadow:var(--shadow); background:#fff;}
[data-testid="stMetric"]{background:#fff; border:1px solid var(--border); border-radius:var(--radius); padding:.8rem 1rem;
  box-shadow:var(--shadow);}
[data-testid="stMetricLabel"] p{font-weight:700; text-transform:uppercase; letter-spacing:.05em; font-size:.74rem; color:var(--slate-text);}
[data-testid="stMetricValue"]{font-weight:800; color:var(--navy);}
[data-testid="stTabs"] [role="tablist"]{gap:.25rem; border-bottom:1px solid var(--border);}
[data-testid="stTabs"] button[role="tab"]{border-radius:10px 10px 0 0; padding:.55rem 1rem; font-weight:600; color:var(--slate-text);}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{color:var(--blue);}
[data-testid="stTabs"] [data-testid="stTab"]{padding:.55rem 1rem; font-weight:600; color:var(--slate-text);}
[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"]{color:var(--blue);}
[data-testid="stExpander"]{background:#fff; border:1px solid var(--border) !important; border-radius:12px; box-shadow:var(--shadow);}
[data-testid="stExpander"] summary{font-weight:600;}
[data-testid="stForm"]{background:#fff; border:1px solid var(--border); border-radius:var(--radius); box-shadow:var(--shadow);
  padding:1.2rem;}
[data-testid="stAlert"]{border-radius:12px;}
.stApp code{border-radius:6px;}
:focus-visible{outline:2px solid var(--blue); outline-offset:2px;}

/* ---- tablet / narrow desktop: no orphan cards, no clipped link labels ---- */
[data-testid="stMain"] [data-testid="stPageLink-NavLink"]{height:auto; min-height:2.4rem;}
[data-testid="stMain"] [data-testid="stPageLink-NavLink"] p{white-space:normal; overflow:visible; text-overflow:clip;
  overflow-wrap:anywhere; line-height:1.3;}
@media (max-width:1280px){
  .diq-kpi-grid:has(> .diq-kpi:nth-child(4):last-child){grid-template-columns:repeat(2,minmax(0,1fr)) !important;}
  .diq-kpi-grid:has(> .diq-kpi:nth-child(2):last-child){grid-template-columns:repeat(2,minmax(0,1fr)) !important;}
}
@media (max-width:1100px){   /* a container whose key starts with "stack-md" puts its columns under each other */
  [class*="st-key-stack-md"] [data-testid="stHorizontalBlock"]{flex-direction:column; gap:1rem;}
  [class*="st-key-stack-md"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]{
    width:100% !important; flex:1 1 100% !important; min-width:0 !important;}
}
@media (max-width:1180px){
  .diq-find-grid > :last-child:nth-child(odd){grid-column:1 / -1;}
}
@media (max-width:640px){
  .diq-title{font-size:1.5rem;}
  .diq-kpi-grid{grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:.7rem;}
  .diq-kpi-value{font-size:1.5rem;}
}
"""


def _css() -> str:
    out = _CSS + _icon_css()
    for key, val in PAL.items():
        out = out.replace(f"__{key}__", val)
    return out


def _run_id() -> str | None:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        return None if ctx is None else str(ctx.script_run_id)
    except Exception:  # pragma: no cover - defensive
        return None


def inject_css() -> None:
    """Inject the DeliveryIQ stylesheet once per script run (pages may call it safely)."""
    rid = _run_id()
    if rid is not None and st.session_state.get("_diq_css_run") == rid:
        return
    st.session_state["_diq_css_run"] = rid
    st.html(f"<style>{_css()}</style>")


# --------------------------------------------------------------------------- #
# Presentational components
# --------------------------------------------------------------------------- #
def page_header(title: str, subtitle: str | None = None, icon: str | None = None,
                eyebrow: str | None = None) -> None:
    """Large page title with optional icon chip, eyebrow label and subtitle."""
    ico = f'<span class="diq-title-ico">{icon_svg(icon, 22)}</span>' if icon else ""
    eb = f'<div class="diq-eyebrow">{esc(eyebrow)}</div>' if eyebrow else ""
    sub = f'<p class="diq-subtitle">{_inline_md(subtitle)}</p>' if subtitle else ""
    st.html(f'<div class="diq-page">{eb}<h1 class="diq-title">{ico}<span>{esc(title)}</span></h1>{sub}</div>')


def section_header(title: str, subtitle: str | None = None, icon: str | None = None) -> None:
    """Section heading inside a page."""
    ico = icon_svg(icon, 20) if icon else ""
    sub = f'<p class="diq-section-sub">{_inline_md(subtitle)}</p>' if subtitle else ""
    st.html(f'<div class="diq-section"><h2 class="diq-section-title">{ico}<span>{esc(title)}</span></h2>{sub}</div>')


def _kpi_html(label: str, value: str, delta: str | None = None, sublabel: str | None = None,
              icon: str | None = None, accent: str = "blue", delta_tone: str = "neutral",
              help: str | None = None) -> str:
    color, bg = _ACCENTS.get(accent, _ACCENTS["blue"])
    ico = f'<span class="diq-kpi-ico">{icon_svg(icon, 17)}</span>' if icon else ""
    tone = delta_tone if delta_tone in ("good", "bad") else ""
    dl = f'<span class="diq-kpi-delta {tone}">{esc(delta)}</span>' if delta else ""
    sub = f'<div class="diq-kpi-sub">{esc(sublabel)}</div>' if sublabel else ""
    title = f' title="{esc(help)}"' if help else ""
    return (f'<div class="diq-kpi" style="--accent:{color};--accent-bg:{bg}"{title}>'
            f'<div class="diq-kpi-top"><span class="diq-kpi-label">{esc(label)}</span>{ico}</div>'
            f'<div class="diq-kpi-value">{esc(value)}</div>{sub}{dl}</div>')


def kpi_card(label: str, value: str, delta: str | None = None, sublabel: str | None = None,
             icon: str | None = None, accent: str = "blue", delta_tone: str = "neutral",
             help: str | None = None) -> None:
    """One KPI tile.  ``value`` is a PRE-FORMATTED string (use the fmt_* helpers).

    ``accent``: blue | green | red | amber | slate.  ``delta_tone``: good | bad | neutral.
    ``icon``: a name from this module's icon set (package, clock, check, alert, truck, map-pin,
    calendar, dollar, trend, percent, bar-chart, ...) or a short emoji.
    """
    st.html(f'<div class="diq-kpi-grid" style="grid-template-columns:minmax(0,1fr)">'
            f'{_kpi_html(label, value, delta, sublabel, icon, accent, delta_tone, help)}</div>')


def kpi_row(items: Sequence[dict]) -> None:
    """A responsive grid of KPI tiles.  Each item is a dict of ``kpi_card`` keyword arguments
    (``label`` and ``value`` required; ``delta``, ``sublabel``, ``icon``, ``accent``, ``delta_tone``, ``help``)."""
    if not items:
        return
    cards = "".join(_kpi_html(**{**{"value": DASH}, **it}) for it in items)
    st.html(f'<div class="diq-kpi-grid">{cards}</div>')


_CALLOUT_ICON = {"info": "info", "watch": "alert", "alert": "x-circle", "success": "check", "action": "target"}


def callout(kind: str, title: str | None, body: str | None = None, footnote: str | None = None) -> None:
    """Coloured message box.  ``kind``: info | watch | alert | success | action (unknown -> info).

    ``action`` is a suggested next step (green, target icon): a lead to investigate, so it must not look like an error."""
    kind = kind if kind in _CALLOUT_ICON else "info"
    t = f'<div class="diq-callout-title">{esc(title)}</div>' if title else ""
    b = f'<div class="diq-callout-text">{_inline_md(body)}</div>' if body else ""
    f = f'<div class="diq-callout-foot">{_inline_md(footnote)}</div>' if footnote else ""
    st.html(f'<div class="diq-callout {kind}" role="note"><span class="diq-callout-ico">'
            f'{icon_svg(_CALLOUT_ICON[kind], 20)}</span><div class="diq-callout-body">{t}{b}{f}</div></div>')


_SEVERITY_KIND = {"info": "info", "watch": "watch", "alert": "alert", "success": "success"}


def insight_box(insight: Any, title: str | None = None, kind: str | None = None) -> None:
    """Render an ``src.insights.Insight`` (or plain text) as a callout, with its sample size.

    Passing a ``str`` shows it as the body; ``kind`` defaults to the insight severity.
    """
    if isinstance(insight, str):
        callout(kind or "info", title, insight)
        return
    sev = getattr(insight, "severity", "info")
    n = getattr(insight, "n", 0)
    callout(kind or _SEVERITY_KIND.get(sev, "info"), title or getattr(insight, "title", None),
            getattr(insight, "text", ""), footnote=f"Based on {fmt_int(n)} orders." if n else None)


def empty_state(message: str = "No data for the current filters",
                hint: str | None = "Widen the date range or clear the state / category filters in the sidebar.",
                icon: str = "inbox") -> None:
    """Friendly placeholder shown instead of a chart / table that has nothing to display."""
    h = f'<div class="hint">{_inline_md(hint)}</div>' if hint else ""
    st.html(f'<div class="diq-empty" role="status"><span class="ico">{icon_svg(icon, 24)}</span>'
            f'<div class="msg">{esc(message)}</div>{h}</div>')


def friendly_error(title: str, detail: str | None = None, command: str | None = None) -> None:
    """Readable error card (never a traceback).  ``command`` is shown in a copyable code block."""
    d = f'<div class="d">{_inline_md(detail)}</div>' if detail else ""
    st.html(f'<div class="diq-error" role="alert"><div class="t">{icon_svg("x-circle", 22)}'
            f'<span>{esc(title)}</span></div>{d}</div>')
    if command:
        st.code(command, language="bash")


def small_sample_note(n: Any, min_orders: int = config.DEFAULT_MIN_ORDERS) -> str | None:
    """Return a short warning string if ``n`` is below ``min_orders`` (else None)."""
    v = _as_float(n)
    if v is None or v >= min_orders:
        return None
    return f"Small sample ({fmt_n(v)}; below {min_orders:,}): treat rates as indicative only."


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
def _mtime(path) -> int:
    # Passed to the cached loaders below as a *non*-underscore argument on purpose: Streamlit leaves
    # underscore-prefixed arguments out of the cache key, which would hide a retrained model / rebuilt data.
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return -1


@st.cache_data(show_spinner="Loading delivered orders…")
def _read_orders(path: str, mtime_ns: int) -> pd.DataFrame:
    from src import preprocessing
    return preprocessing.load_processed(path)


def data_available() -> bool:
    """True when the processed analysis parquet exists."""
    return config.ORDERS_DELIVERED_PATH.is_file()


def render_setup_page() -> None:
    """Setup instructions shown when the processed data is missing."""
    page_header("Set up your data", "DeliveryIQ analyses the public Olist Brazilian E-Commerce dataset. "
                "The processed analysis table has not been created yet.", icon="database", eyebrow="First run")
    steps = (
        f"1. Download the dataset from {config.KAGGLE_URL} (free Kaggle account).\n"
        f"2. Copy these files into `{config.RAW_DIR}`: "
        + ", ".join(f"`{f}`" for f in list(config.REQUIRED_RAW_FILES.values()) + list(config.OPTIONAL_RAW_FILES.values()))
        + ".\n3. Build the processed tables with the command below, then reload this page."
    )
    friendly_error("Processed data not found", steps, PROCESS_COMMAND)
    st.caption("Optional: after the data exists, train the delay-prediction model with "
               f"`{TRAIN_COMMAND}`. The rest of the dashboard works without it.")


def get_orders() -> pd.DataFrame:
    """Delivered / eligible orders (cached).  Shows a setup message and stops the page if missing."""
    path = config.ORDERS_DELIVERED_PATH
    if not path.is_file():
        render_setup_page()
        st.stop()
    try:
        return _read_orders(str(path), _mtime(path))
    except Exception as exc:  # corrupt / unreadable file
        friendly_error("The processed data could not be read", str(exc), PROCESS_COMMAND)
        st.stop()
        raise  # pragma: no cover - st.stop raises


@st.cache_data(show_spinner=False)
def _read_json(path: str, mtime_ns: int) -> dict | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def get_quality_report() -> dict:
    """Data-quality report written by the preprocessing step ({} if missing)."""
    p = config.QUALITY_REPORT_PATH
    return _read_json(str(p), _mtime(p)) or {}


def get_model_metrics() -> dict | None:
    """Evaluation output of the trained model (None if the model has not been trained)."""
    p = config.MODEL_METRICS_PATH
    return _read_json(str(p), _mtime(p))


@st.cache_resource(show_spinner="Loading prediction model…")
def _load_model_cached(path: str, mtime_ns: int) -> tuple[dict | None, str | None]:
    from src import prediction
    try:
        return prediction.load_model(path), None
    except Exception as exc:  # missing file, version mismatch, corrupt bundle
        return None, str(exc)


def get_model_bundle() -> dict | None:
    """Trained model bundle (cached resource) or None when missing / unreadable.

    Non-prediction pages keep working without it; check :func:`get_model_load_error` for the reason.
    """
    p = config.MODEL_PATH
    bundle, err = _load_model_cached(str(p), _mtime(p))
    st.session_state["model_missing"] = bundle is None
    st.session_state["model_error"] = err
    return bundle


def get_model_load_error() -> str | None:
    """Why the model could not be loaded (None if it loaded fine)."""
    p = config.MODEL_PATH
    return _load_model_cached(str(p), _mtime(p))[1]


def require_model() -> dict:
    """Return the model bundle or show a friendly error with the training command and stop the page."""
    bundle = get_model_bundle()
    if bundle is None:
        friendly_error("The prediction model is not available",
                       "Train it once (about two minutes) and reload this page. The other pages keep working "
                       "without it.\n" + (get_model_load_error() or ""), TRAIN_COMMAND)
        st.stop()
    return bundle


# --------------------------------------------------------------------------- #
# Global filters
# --------------------------------------------------------------------------- #
def _bounds(df: pd.DataFrame) -> tuple[date, date]:
    s = df["purchase_date"] if len(df) else pd.Series([], dtype="datetime64[ns]")
    if s.dropna().empty:
        today = date.today()
        return today, today
    return s.min().date(), s.max().date()


def _to_date(v: Any) -> date | None:
    if v is None or v is pd.NaT:
        return None
    try:
        if isinstance(v, datetime):
            return v.date()
        if isinstance(v, date):
            return v
        return pd.Timestamp(v).date()
    except (ValueError, TypeError):
        return None


def _normalise_dates(val: Any, lo: date, hi: date) -> tuple[date, date, bool]:
    """Clip / order a stored date value into ``(start, end, complete)``.

    Handles a single date, a 1-tuple (range still being picked), reversed bounds and values
    outside the data span.
    """
    if val is None:
        return lo, hi, True
    items = [val] if not isinstance(val, (list, tuple)) else list(val)
    ds = [d for d in (_to_date(x) for x in items) if d is not None]
    if not ds:
        return lo, hi, True
    complete = len(ds) >= 2
    start = ds[0]
    end = ds[1] if complete else hi
    if start > end:
        start, end = end, start
    start, end = max(start, lo), min(end, hi)
    if start > end:  # entirely outside the data span
        return lo, hi, complete
    return start, end, complete


def _options(df: pd.DataFrame, col: str) -> list[str]:
    return sorted(df[col].dropna().astype(str).unique().tolist()) if col in df.columns and len(df) else []


def current_filters(df: pd.DataFrame) -> dict:
    """Read the sidebar filter state from ``st.session_state`` and validate it against ``df``.

    Works when no sidebar has been rendered (standalone page): defaults apply (no filtering).
    """
    lo, hi = _bounds(df)
    start, end, complete = _normalise_dates(st.session_state.get(K_DATE), lo, hi)
    states = [s for s in (st.session_state.get(K_STATES) or []) if s in set(_options(df, "customer_state"))]
    cats = [c for c in (st.session_state.get(K_CATEGORIES) or []) if c in set(_options(df, "product_category"))]
    date_active = (start, end) != (lo, hi)
    return {
        "date_range": (start, end), "data_range": (lo, hi), "date_complete": complete,
        "states": states, "categories": cats,
        "active": bool(date_active or states or cats),
    }


def describe_filters(flt: dict) -> str:
    """One-line plain-text description of the active filters (for report headers)."""
    s, e = flt["date_range"]
    parts = [f"purchase date {s:%Y-%m-%d} to {e:%Y-%m-%d}"]
    parts.append("customer state: " + (", ".join(flt["states"]) if flt["states"] else "all"))
    parts.append("product category: " + (f"{len(flt['categories'])} selected" if flt["categories"]
                                         and len(flt["categories"]) > 3 else
                                         ", ".join(category_label(c) for c in flt["categories"]) or "all"))
    return "; ".join(parts)


def _apply(df: pd.DataFrame, flt: dict) -> pd.DataFrame:
    return apply_filters(df, date_range=flt["date_range"], states=flt["states"] or None,
                         categories=flt["categories"] or None)


def get_filtered_orders() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """``(df_all, df_filtered, filters)`` honouring the sidebar filters.

    Also works when a page file is run on its own (no filters are active then).  ``filters`` has
    ``date_range``, ``states``, ``categories`` (empty list = all), ``active``, ``description``,
    ``n_total`` and ``n_filtered``.
    """
    df_all = get_orders()
    flt = current_filters(df_all)
    df = _apply(df_all, flt)
    flt.update(description=describe_filters(flt), n_total=len(df_all), n_filtered=len(df))
    return df_all, df, flt


def stop_if_empty(df: pd.DataFrame) -> None:
    """Show the standard empty state and stop the page when the filtered frame has no rows."""
    if len(df) == 0:
        empty_state("No orders match the current filters",
                    "Widen the date range or clear the state / category filters in the sidebar.")
        st.stop()


def filter_banner(flt: dict) -> None:
    """A compact row of pills summarising the active filters and the sample size."""
    s, e = flt["date_range"]
    states = ", ".join(flt["states"]) if flt["states"] else "All states"
    cats = (f"{len(flt['categories'])} categor{'y' if len(flt['categories']) == 1 else 'ies'}"
            if flt["categories"] else "All categories")
    pills = [f'<span class="diq-pill"><b>{fmt_int(flt.get("n_filtered"))}</b> of {fmt_int(flt.get("n_total"))} orders</span>',
             f'<span class="diq-pill">{s:%d %b %Y} → {e:%d %b %Y}</span>',
             f'<span class="diq-pill">{esc(states)}</span>',
             f'<span class="diq-pill">{esc(cats)}</span>']
    if flt.get("n_filtered", 1) == 0:
        pills.append('<span class="diq-pill is-warn">No orders match</span>')
    st.html(f'<div class="diq-pillrow">{"".join(pills)}</div>')


def _reset_filters(lo: date, hi: date) -> None:
    st.session_state[K_DATE] = (lo, hi)
    st.session_state[K_STATES] = []
    st.session_state[K_CATEGORIES] = []


def render_global_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render the sidebar filters (date range, customer state, product category) and return the
    filtered frame.  Values live in ``st.session_state`` under stable keys."""
    lo, hi = _bounds(df)
    states_opt = _options(df, "customer_state")
    cats_opt = _options(df, "product_category")

    # Sanitise stored values BEFORE widgets are created (data may have changed between runs).
    if K_DATE not in st.session_state:
        st.session_state[K_DATE] = (lo, hi)
    else:
        v = st.session_state[K_DATE]
        s, e, complete = _normalise_dates(v, lo, hi)
        if complete and tuple(_to_date(x) for x in v) != (s, e):
            st.session_state[K_DATE] = (s, e)
    for key, opts in ((K_STATES, states_opt), (K_CATEGORIES, cats_opt)):
        cur = st.session_state.get(key) or []
        clean = [x for x in cur if x in opts]
        if key not in st.session_state or clean != list(cur):
            st.session_state[key] = clean

    with st.sidebar:
        st.html('<div class="diq-side-label">Filters</div>')
        st.date_input("Purchase date range", min_value=lo, max_value=hi, key=K_DATE, format="YYYY-MM-DD",
                      help="Orders are filtered by the date the customer placed the order.")
        st.multiselect("Customer state", states_opt, key=K_STATES, format_func=state_label,
                       placeholder="All states", select_all=False)
        st.multiselect("Product category", cats_opt, key=K_CATEGORIES, format_func=category_label,
                       placeholder="All categories", select_all=False)
        st.button("Reset filters", key="flt_reset", on_click=_reset_filters, args=(lo, hi),
                  icon=":material/restart_alt:", width="stretch")

        flt = current_filters(df)
        out = _apply(df, flt)
        n, total = len(out), len(df)
        share = 0 if total == 0 else max(0.0, min(100.0, 100 * n / total))
        if not flt["date_complete"]:
            st.caption("Pick an end date to finish the range (showing from the start date onwards).")
        if n == 0:
            st.html(f'<div class="diq-counter zero" role="status"><div class="n">0 of {fmt_int(total)} orders match</div>'
                    '<div class="warn">No orders match these filters. Widen the date range or clear a '
                    'state / category filter.</div></div>')
        else:
            st.html(f'<div class="diq-counter" role="status"><span class="n">{fmt_int(n)}</span> '
                    f'<span class="of">of {fmt_int(total)} orders match</span>'
                    f'<div class="bar"><span style="width:{share:.1f}%"></span></div></div>')
    return out


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_brand() -> None:
    """Logo, product name and tagline (sidebar)."""
    st.html(f'<div class="diq-brand">{LOGO_HTML}<div><div class="diq-brand-name">{esc(config.APP_NAME)}</div>'
            f'<div class="diq-brand-tag">{esc(config.APP_TAGLINE)}</div></div></div>')


def _dataset_info_html(df: pd.DataFrame | None) -> str:
    rows: list[tuple[str, str]] = [("Source", "Olist (Kaggle)")]
    if df is not None and len(df):
        lo, hi = _bounds(df)
        rows += [("Period", f"{lo:%b %Y} – {hi:%b %Y}"), ("Orders analysed", f"{fmt_int(len(df))} delivered")]
    bundle = get_model_bundle()
    metrics = get_model_metrics() or {}
    if bundle is None:
        rows.append(("Model", '<span class="diq-dot" style="background:#F59E0B"></span>Not trained'))
    else:
        label = str(bundle.get("model_name", "model")).replace("_", " ").capitalize()
        trained = str(bundle.get("trained_at") or metrics.get("generated_at") or "")[:10]
        rows.append(("Model", f'<span class="diq-dot" style="background:#16A34A"></span>{esc(label)}'))
        if trained:
            rows.append(("Trained", esc(trained)))
        try:
            sel = (metrics.get("selected_model") or {}).get("name") or bundle.get("model_name")
            test = metrics["models"][sel]["test"]
            rows.append(("Test ROC-AUC", f'{fmt_num(test["roc_auc"], 2)} (PR-AUC {fmt_num(test["pr_auc"], 2)})'))
        except (KeyError, TypeError):
            pass
    body = "".join(f'<div class="row"><span class="k">{esc(k)}</span><span class="v">'
                   f'{v if "<span" in v else esc(v)}</span></div>' for k, v in rows)
    return f'<div class="diq-info"><div class="hd">Dataset &amp; model</div>{body}</div>'


#: On a phone the sidebar is a drawer that covers the page.  Tapping a navigation link used to leave it open over the
#: page that had just been selected; this closes it again (narrow screens only, navigation links only, so the filters
#: keep working without the drawer jumping shut).  Registered once per browser page load.
CLOSE_DRAWER_JS = """
(function () {
  if (window.__diqDrawerClose) return;
  window.__diqDrawerClose = true;
  document.addEventListener('click', function (ev) {
    try {
      var t = ev.target;
      if (!t || !t.closest || !t.closest('[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]')) return;
      if (!window.matchMedia('(max-width: 768px)').matches) return;
      setTimeout(function () {
        try {
          var b = document.querySelector('[data-testid="stSidebarCollapseButton"] button, button[data-testid="stSidebarCollapseButton"]');
          if (b) b.click();
        } catch (e) { /* a Streamlit markup change must never break the page */ }
      }, 250);
    } catch (e) { /* same */ }
  }, true);
})();
"""


def inject_drawer_js() -> None:
    """Install the close-the-drawer-after-navigation behaviour for narrow screens (see ``CLOSE_DRAWER_JS``)."""
    st.html(f"<script>{CLOSE_DRAWER_JS}</script>", unsafe_allow_javascript=True)


def render_sidebar(pages: Sequence[Any] | None = None, current: Any = None) -> pd.DataFrame | None:
    """Hand-rendered sidebar: brand, navigation, global filters, remaining-orders counter and
    dataset info.  ``current`` is the page returned by ``st.navigation`` (highlighted in the nav).
    Returns the filtered frame (or None when no data is available)."""
    with st.sidebar:
        render_brand()
        if pages:
            inject_drawer_js()
            st.html('<div class="diq-side-label">Navigate</div>')
            with st.container(key="nav-list"):
                for i, p in enumerate(pages):
                    with st.container(key="nav-active" if p is current else f"nav-item-{i}"):
                        st.page_link(p, label=p.title, icon=p.icon)
    if not data_available():
        return None
    df = get_orders()
    filtered = render_global_filters(df)
    with st.sidebar:
        st.html(_dataset_info_html(df))
    return filtered
