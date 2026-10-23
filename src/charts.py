"""Plotly figure builders for DeliveryIQ (pure Python, no Streamlit).

Every builder
* takes the DataFrame (or metrics dict) produced by the matching ``src.analytics`` function /
  ``models/model_metrics.json`` plus optional keyword arguments,
* returns a :class:`plotly.graph_objects.Figure` that uses the shared ``deliveryiq`` template,
* has a title, labelled axes with units, tooltips that carry ``n``,
* returns :func:`empty_figure` ("No data for the current filters") for ``None`` / empty / all-NaN
  input and **never raises** on empty input.

Colour rules (see ``config.PALETTE``): green = on time, amber = moderate risk, red = late / high
risk are *semantic* and only used for those meanings; categorical identity colours (blue, cyan,
pink, violet) are validated to stay apart from them; magnitude uses single-hue sequential ramps
(red ramp for late rates, blue ramp for durations / volumes).  Small samples are drawn in a
muted, hatched style and flagged in the legend -- never silently mixed in with reliable ones.
Everything is descriptive: nothing here implies causality.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from .config import DEFAULT_MIN_ORDERS, GEO_DIR, PALETTE, STATE_NAMES

__all__ = [
    "TEMPLATE", "PLOTLY_CONFIG", "CATEGORICAL", "GEOJSON_PATH", "NO_DATA_MESSAGE",
    "apply_template", "empty_figure", "pretty_feature_name", "pretty_category", "figure_to_html_bytes",
    "status_donut", "monthly_performance_chart", "duration_histogram", "state_late_rate_bar",
    "category_late_rate_bar", "category_duration_bar", "category_freight_bar", "state_duration_bar",
    "flagged_months_chart", "weight_vs_duration",
    "freight_vs_duration", "weight_late_rate_bars", "freight_late_rate_bars", "processing_histogram",
    "late_rate_by_processing_bucket", "state_pair_heatmap", "state_pair_bar", "state_comparison_chart",
    "delay_trend_chart", "brazil_choropleth", "model_comparison_chart", "confusion_matrix_chart",
    "roc_curve_chart", "pr_curve_chart", "calibration_chart", "feature_importance_chart",
    "risk_band_chart", "probability_gauge", "load_brazil_geojson", "scenario_comparison_chart",
]

NO_DATA_MESSAGE = "No data for the current filters"
GEOJSON_PATH: Path = GEO_DIR / "brazil_states.geojson"

# --------------------------------------------------------------------------- #
# Palette / template
# --------------------------------------------------------------------------- #
INK = PALETTE["navy"]                 # titles
INK_2 = PALETTE["slate"]              # body text
MUTED = PALETTE["slate_light"]        # axis / secondary text
GRID = "#E8EDF3"
BASELINE = "#CBD5E1"
NEUTRAL = "#94A3B8"                   # inconclusive / "no flag" marks
NEUTRAL_LIGHT = "#CBD5E1"             # small-sample marks
MASKED = "#F1F5F9"                    # cells hidden because of small n
MASKED_CELL = "#E2E8F0"               # masked heat-map cells: visibly grey against the card, not 'missing data' white
FONT = "Inter, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"

# Categorical identity colours (validated apart from semantic green/amber/red; CVD-safe order).
CATEGORICAL = [PALETTE["blue"], "#0891B2", "#DB2777", "#7C3AED", "#64748B"]
VALIDATION_BLUE = PALETTE["blue_light"]

#: Single-hue sequential ramps (light -> dark).
RED_SCALE = [[0.0, "#FEF2F2"], [0.35, "#FCA5A5"], [0.7, "#DC2626"], [1.0, "#7F1D1D"]]
BLUE_SCALE = [[0.0, "#EFF6FF"], [0.35, "#93C5FD"], [0.7, "#2563EB"], [1.0, "#1E3A8A"]]

RISK_COLORS = {"Low": PALETTE["green"], "Moderate": PALETTE["amber"], "High": PALETTE["red"]}

PLOTLY_CONFIG: dict[str, Any] = {
    "displaylogo": False,
    "displayModeBar": "hover",
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}

TEMPLATE = go.layout.Template(
    layout=dict(
        font=dict(family=FONT, size=12, color=INK_2),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=CATEGORICAL,
        title=dict(font=dict(size=17, color=INK), x=0, xanchor="left", y=0.985, yanchor="top",
                   subtitle=dict(font=dict(size=12, color=MUTED))),
        xaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor=BASELINE, linewidth=1,
                   ticks="outside", tickcolor=BASELINE, ticklen=4, tickfont=dict(size=11, color=MUTED),
                   title=dict(font=dict(size=12, color=INK_2), standoff=10), automargin=True),
        yaxis=dict(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False, showline=False,
                   ticks="", tickfont=dict(size=11, color=MUTED),
                   title=dict(font=dict(size=12, color=INK_2), standoff=10), automargin=True),
        legend=dict(font=dict(size=12, color=INK_2), bgcolor="rgba(0,0,0,0)", itemsizing="constant"),
        hoverlabel=dict(bgcolor="white", bordercolor=BASELINE, font=dict(family=FONT, size=12, color=INK)),
        hovermode="closest",
        barcornerradius=4,
        margin=dict(l=16, r=24, t=64, b=16),
        colorscale=dict(sequential=BLUE_SCALE, sequentialminus=BLUE_SCALE),
    )
)
pio.templates["deliveryiq"] = TEMPLATE


def apply_template(fig: go.Figure) -> go.Figure:
    """Apply the shared DeliveryIQ template to ``fig`` (in place) and return it."""
    fig.update_layout(template=TEMPLATE)
    return fig


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _title_dict(title: str, subtitle: str | None = None) -> dict:
    t: dict[str, Any] = {"text": title}
    if subtitle:
        t["subtitle"] = {"text": subtitle}
    return t


def _legend_rows(fig: go.Figure, container_px: int = 620) -> int:
    """Rough number of legend rows for a ~620 px wide chart (names drive the width)."""
    widths = [len(str(tr.name)) * 6.6 + 44 for tr in fig.data if tr.showlegend is not False and tr.name]
    if not widths:
        return 1
    rows, cur = 1, 0.0
    for w in widths:
        if cur and cur + w > container_px:
            rows += 1
            cur = 0.0
        cur += w
    return rows


#: Subtitle line length / assumed legend width for charts that must stay legible on a phone-width column.
_NARROW_WRAP = 52
_NARROW_LEGEND_PX = 340


def _finish(fig: go.Figure, title: str, subtitle: str | None = None, *, height: int = 420,
            legend: bool = False, top: int | None = None, bottom: int | None = None,
            right: int | None = None, left: int | None = None, wrap: int = 100,
            legend_px: int = 620) -> go.Figure:
    """Apply template, title/subtitle, size, margins and (optionally) a top-left legend.

    ``wrap`` = characters per subtitle line and ``legend_px`` = assumed chart width for the legend-row
    estimate; charts shown on phone-width screens pass small values so nothing is clipped or overlaps.
    """
    if subtitle and len(subtitle) > wrap:
        subtitle = _wrap(subtitle, wrap)
    n_sub = subtitle.count("<br>") + 1 if subtitle else 0
    t = top if top is not None else (64 if not subtitle else 78 + 15 * (n_sub - 1))
    if legend and top is None:
        t += 34 + 22 * (_legend_rows(fig, legend_px) - 1)
    fig.update_layout(
        template=TEMPLATE, title=_title_dict(title, subtitle), height=height,
        margin=dict(l=left if left is not None else 16, r=right if right is not None else 24,
                    t=t, b=bottom if bottom is not None else 16),
        showlegend=legend,
    )
    if legend:
        # y slightly above 1.0 so the legend never sits on the top y-axis tick label ("100%")
        fig.update_layout(legend=dict(orientation="h", x=0, xanchor="left", y=1.03, yanchor="bottom",
                                      yref="paper", traceorder="normal"))
    return fig


def empty_figure(message: str = NO_DATA_MESSAGE, title: str = "", height: int = 320,
                 hint: str | None = None) -> go.Figure:
    """A clean placeholder figure (same template) used whenever there is nothing to draw."""
    fig = go.Figure()
    text = message if not hint else f"{message}<br><span style='font-size:12px;color:{MUTED}'>{hint}</span>"
    fig.add_annotation(text=text, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                       font=dict(size=15, color=MUTED), align="center")
    fig.update_xaxes(visible=False, showgrid=False)
    fig.update_yaxes(visible=False, showgrid=False)
    fig.update_layout(template=TEMPLATE, height=height, showlegend=False,
                      margin=dict(l=16, r=24, t=64 if title else 16, b=16))
    if title:
        fig.update_layout(title=_title_dict(title))
    return fig


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").astype("float64").replace([np.inf, -np.inf], np.nan)


def _frame_ok(df: Any, required: Sequence[str] = (), key_metric: str | None = None) -> bool:
    """True when ``df`` is a non-empty DataFrame with ``required`` columns and some finite metric."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return False
    if any(c not in df.columns for c in required):
        return False
    if key_metric is not None and not _num(df, key_metric).notna().any():
        return False
    return True


def _pct(x: float, nd: int = 1) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:.{nd}f}%"


def _int(x: float) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{int(round(x)):,}"


def _nice_float(x: float, nd: int = 1) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:,.{nd}f}"


def _wrap(text: str, width: int = 16) -> str:
    """Wrap ``text`` on spaces into ``<br>`` separated lines."""
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "<br>".join(lines)


def pretty_category(name: Any, max_len: int | None = None) -> str:
    """``"health_beauty"`` -> ``"Health beauty"`` (optionally truncated with an ellipsis)."""
    s = str(name).replace("_", " ").strip()
    s = s[:1].upper() + s[1:] if s else s
    if max_len and len(s) > max_len:
        s = s[: max_len - 1].rstrip() + "…"
    return s


_FEATURE_NAMES = {
    "estimated_delivery_days": "Promised delivery time (days)",
    "customer_state": "Customer state",
    "seller_state": "Seller state",
    "same_state": "Seller & customer in same state",
    "product_category": "Product category",
    "n_items": "Number of items",
    "n_sellers": "Number of sellers",
    "n_products": "Number of products",
    "total_price": "Order value (R$)",
    "total_freight": "Freight cost (R$)",
    "freight_ratio": "Freight / item price",
    "total_weight_g": "Total weight (g)",
    "total_volume_cm3": "Package volume (cm³)",
    "max_product_length_cm": "Longest product length (cm)",
    "max_product_height_cm": "Tallest product height (cm)",
    "max_product_width_cm": "Widest product width (cm)",
    "purchase_dow": "Purchase weekday",
    "purchase_month": "Purchase month",
    "purchase_quarter": "Purchase quarter",
    "delivery_duration_days": "Delivery time (days)",
    "processing_days": "Seller processing time (days)",
}


def pretty_feature_name(name: Any) -> str:
    """Human-readable label for a model feature / column name."""
    key = str(name)
    if key in _FEATURE_NAMES:
        return _FEATURE_NAMES[key]
    for prefix in ("num__", "cat__", "remainder__"):
        if key.startswith(prefix):
            return pretty_feature_name(key[len(prefix):])
    return pretty_category(key)


def figure_to_html_bytes(fig: go.Figure, include_plotlyjs: str | bool = "cdn") -> bytes:
    """Standalone interactive HTML of ``fig`` as bytes (for ``st.download_button``)."""
    html = pio.to_html(fig, include_plotlyjs=include_plotlyjs, full_html=True, config=PLOTLY_CONFIG)
    return html.encode("utf-8")


def _state_label(code: Any) -> str:
    return STATE_NAMES.get(str(code), str(code))


def _ci_arrays(d: pd.DataFrame, rate_col: str = "late_rate") -> tuple[np.ndarray, np.ndarray] | None:
    """(plus, minus) error lengths from Wilson columns, or None if they are missing."""
    if "late_ci_low" not in d.columns or "late_ci_high" not in d.columns:
        return None
    rate = _num(d, rate_col).to_numpy()
    lo, hi = _num(d, "late_ci_low").to_numpy(), _num(d, "late_ci_high").to_numpy()
    plus = np.where(np.isfinite(hi), np.clip(hi - rate, 0, None), 0.0)
    minus = np.where(np.isfinite(lo), np.clip(rate - lo, 0, None), 0.0)
    return plus, minus


def _overall_rate(d: pd.DataFrame, overall_rate: float | None) -> float:
    if overall_rate is not None and np.isfinite(overall_rate):
        return float(overall_rate)
    n = _num(d, "n_orders")
    if "n_late" in d.columns and n.sum() > 0:
        return float(_num(d, "n_late").sum() / n.sum())
    w = n.fillna(0)
    r = _num(d, "late_rate")
    return float((r * w).sum() / w.sum()) if w.sum() > 0 else float("nan")


def _rate_range(d: pd.DataFrame, col: str, from_zero: bool, hi_col: str | None = None,
                lo_col: str | None = None) -> list[float] | None:
    """Axis range fitted to the *reliable* points so tiny-n outliers cannot squash the real signal."""
    if d is None or d.empty:
        return None
    vals = _num(d, col)
    hi = _num(d, hi_col) if hi_col and hi_col in d.columns else vals
    lo = _num(d, lo_col) if lo_col and lo_col in d.columns else vals
    top = float(np.nanmax(np.append(hi.to_numpy(), vals.to_numpy())))
    bot = float(np.nanmin(np.append(lo.to_numpy(), vals.to_numpy())))
    if not np.isfinite(top) or not np.isfinite(bot):
        return None
    span = max(top - (0.0 if from_zero else bot), 1e-6)
    return [0.0 if from_zero else max(bot - 0.15 * span, 0.0), min(top + 0.15 * span, 1.0 if not from_zero else top * 1.2 + 1e-6)]


def _pct_tickformat(rng: list[float] | None) -> str:
    """Percent tick format with enough decimals that neighbouring ticks never share a label (e.g. 99% x3)."""
    if not rng:
        return ".0%"
    span = abs(float(rng[1]) - float(rng[0]))
    return ".0%" if span >= 0.05 else (".1%" if span >= 0.005 else ".2%")


def _contiguous_runs(good: pd.DataFrame, step: pd.Timedelta, xcol: str = "_x") -> list[pd.DataFrame]:
    """Split a time-ordered frame into runs with no missing period in between (so lines / bands break at gaps)."""
    if good.empty:
        return []
    brk = (good[xcol].diff() > step * 1.6).to_numpy()
    ids = np.cumsum(brk)
    return [g for _, g in good.groupby(ids)]


def _gapped(good: pd.DataFrame, step: pd.Timedelta, xcol: str = "_x") -> pd.DataFrame:
    """``good`` with an all-NaN row inserted wherever periods are missing, so a line is not drawn across the gap."""
    runs = _contiguous_runs(good, step, xcol)
    parts: list[pd.DataFrame] = []
    for i, r in enumerate(runs):
        if i:
            gap = pd.DataFrame({xcol: [r[xcol].iloc[0] - step / 2]}, columns=good.columns)
            parts.append(gap)
        parts.append(r)
    return pd.concat(parts, ignore_index=True) if parts else good


def _freq_alias(step: pd.Timedelta) -> str:
    days = step.days
    if days >= 85:
        return "QS"
    if days >= 27:
        return "MS"
    return f"{max(days, 1)}D"


def _step(x: pd.Series) -> pd.Timedelta:
    """Smallest spacing between consecutive periods (1 month ~ 28 days, 1 week = 7 days, ...)."""
    diffs = x.sort_values().diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    return diffs.min() if len(diffs) else pd.Timedelta(days=1)


def _month_start(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype(str) + "-01", errors="coerce")


def _legend_proxy(fig: go.Figure, name: str, color: str, *, dash: str | None = None,
                  marker_symbol: str | None = None, pattern: bool = False) -> None:
    """Add a legend-only entry (no data) for reference lines / hatched swatches."""
    if dash:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=name,
                                 line=dict(color=color, width=2, dash=dash), hoverinfo="skip"))
    elif marker_symbol:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name,
                                 marker=dict(color=color, symbol=marker_symbol, size=9), hoverinfo="skip"))
    else:
        fig.add_trace(go.Bar(x=[None], y=[None], name=name, hoverinfo="skip",
                             marker=dict(color=color, pattern=dict(shape="/", fgcolor="#FFFFFF" if pattern else color,
                                                                   bgcolor=color, size=6) if pattern else None)))


# --------------------------------------------------------------------------- #
# Generic builders (shared by several public charts)
# --------------------------------------------------------------------------- #
def _plain_hbar(labels: Sequence[str], values: Sequence[float], *, title: str, subtitle: str | None,
                xtitle: str, ytitle: str, hover: Sequence[str], text: Sequence[str], color: str,
                ref_value: float | None = None, ref_name: str | None = None, xformat: str | None = None,
                small: Sequence[bool] | None = None, small_label: str | None = None,
                note: str | None = None) -> go.Figure:
    """Horizontal bars (first item on top) with value labels, optional reference line + small-n hatch."""
    labels = list(labels)
    values = [float(v) for v in values]
    small = list(small) if small is not None else [False] * len(labels)
    fig = go.Figure()
    for is_small in (False, True):
        idx = [i for i, s in enumerate(small) if s == is_small]
        if not idx:
            continue
        marker = dict(color=NEUTRAL_LIGHT if is_small else color)
        if is_small:
            marker["pattern"] = dict(shape="/", fgcolor="#FFFFFF", bgcolor=NEUTRAL_LIGHT, size=6)
        fig.add_trace(go.Bar(
            x=[values[i] for i in idx], y=[labels[i] for i in idx], orientation="h", marker=marker,
            text=[text[i] for i in idx], textposition="outside", cliponaxis=False,
            textfont=dict(size=11, color=INK_2), customdata=[hover[i] for i in idx],
            hovertemplate="%{customdata}<extra></extra>",
            name=small_label if is_small and small_label else "Segments",
            showlegend=bool(is_small and small_label),
        ))
    legend = bool(any(small) and small_label)
    xmax = max(values) if values else 1.0
    if ref_value is not None and np.isfinite(ref_value):
        xmax = max(xmax, ref_value)
        fig.add_shape(type="line", x0=ref_value, x1=ref_value, y0=0, y1=1, xref="x", yref="paper",
                      line=dict(color=INK, width=1.5, dash="dash"))
        if ref_name:
            legend = True
            _legend_proxy(fig, ref_name, INK, dash="dash")
    fig.update_xaxes(title=xtitle, range=[0, xmax * 1.22 if xmax > 0 else 1], showgrid=True, gridcolor=GRID,
                     showline=False, ticks="")
    if xformat:
        fig.update_xaxes(tickformat=xformat)
    fig.update_yaxes(title=ytitle, showgrid=False, autorange="reversed", categoryorder="array",
                     categoryarray=labels, ticks="")
    fig.update_layout(bargap=0.35)
    h = max(300, 26 * len(labels) + 170)
    return _finish(fig, title, subtitle, height=h, legend=legend)


def _segment_rate_bar(
    df: pd.DataFrame, key: str, *, title: str, subtitle_extra: str | None = None, xtitle: str, ytitle: str,
    min_orders: int, overall_rate: float | None, top_n: int | None, include_small: bool, ascending: bool,
    label_fn, name_fn, ci_label: str = "95% CI",
) -> go.Figure:
    """Late-rate bars per segment: sorted, CI whiskers, overall reference line, small-n de-emphasised."""
    if not _frame_ok(df, [key, "n_orders", "late_rate"], "late_rate"):
        return empty_figure(title=title)
    d = df.copy()
    d["_rate"] = _num(d, "late_rate")
    d["_n"] = _num(d, "n_orders").fillna(0)
    d = d[d["_rate"].notna() & (d["_n"] > 0)]
    if d.empty:
        return empty_figure(title=title)
    overall = _overall_rate(d, overall_rate)
    d["_small"] = d["_n"] < min_orders
    big = d[~d["_small"]].sort_values(["_rate", "_n"], ascending=[ascending, False], kind="mergesort")
    small = d[d["_small"]].sort_values(["_rate", "_n"], ascending=[ascending, False], kind="mergesort")
    n_hidden_big = n_hidden_small = 0
    if top_n is not None and len(big) > top_n:
        n_hidden_big = len(big) - top_n
        big = big.head(top_n)
    if not include_small:
        n_hidden_small = len(small)
        small = small.iloc[0:0]
    elif top_n is not None and len(small) > max(3, top_n // 3):
        keep = max(3, top_n // 3)
        n_hidden_small = len(small) - keep
        small = small.head(keep)
    d = pd.concat([big, small])
    if d.empty:
        return empty_figure(title=title, hint=f"All segments have fewer than {min_orders} orders.")

    ci = _ci_arrays(d)
    d["_plus"], d["_minus"] = (ci if ci is not None else (np.zeros(len(d)), np.zeros(len(d))))
    lo, hi = _num(d, "late_ci_low"), _num(d, "late_ci_high")
    cls = np.where(d["_small"], "small",
                   np.where(lo > overall, "above", np.where(hi < overall, "below", "overlap")))
    if ci is None:
        cls = np.where(d["_small"], "small", "overlap")
    d["_cls"] = cls
    d["_label"] = [f"{label_fn(k)}  ·  n={int(n):,}" for k, n in zip(d[key], d["_n"])]
    order = list(d["_label"])
    n_late = _num(d, "n_late") if "n_late" in d.columns else (d["_rate"] * d["_n"]).round()

    styles = [
        ("above", "Above overall (CI excludes it)", PALETTE["red"], False),
        ("overlap", "CI includes overall rate", NEUTRAL, False),
        ("below", "Below overall (CI excludes it)", PALETTE["green"], False),
        ("small", f"Small sample (n < {min_orders:,})", NEUTRAL_LIGHT, True),
    ]
    fig = go.Figure()
    for cname, legend_name, color, hatched in styles:
        sub = d[d["_cls"] == cname]
        if sub.empty:
            continue
        marker = dict(color=color)
        if hatched:
            marker["pattern"] = dict(shape="/", fgcolor="#FFFFFF", bgcolor=color, size=6)
        cd = np.column_stack([
            [name_fn(k) for k in sub[key]], sub["_n"].to_numpy(), n_late.loc[sub.index].to_numpy(),
            sub["_rate"].to_numpy(), _num(sub, "late_ci_low").to_numpy(), _num(sub, "late_ci_high").to_numpy(),
        ])
        fig.add_trace(go.Bar(
            x=sub["_rate"], y=sub["_label"], orientation="h", name=legend_name, marker=marker,
            error_x=dict(type="data", symmetric=False, array=sub["_plus"], arrayminus=sub["_minus"],
                         color=INK_2, thickness=1.2, width=3) if ci is not None else None,
            customdata=cd,
            hovertemplate=("<b>%{customdata[0]}</b><br>Late rate: <b>%{customdata[3]:.1%}</b><br>"
                           "Late orders: %{customdata[2]:,.0f} of %{customdata[1]:,.0f}<br>"
                           f"{ci_label}: " + "%{customdata[4]:.1%} to %{customdata[5]:.1%}<extra></extra>"),
        ))
    # value labels to the right of the whiskers
    xs = (d["_rate"] + d["_plus"]).to_numpy()
    fig.add_trace(go.Scatter(x=xs, y=d["_label"], mode="text", text=["\u00a0\u00a0\u00a0" + _pct(r) for r in d["_rate"]],
                             textposition="middle right", textfont=dict(size=11, color=INK_2),
                             hoverinfo="skip", showlegend=False, cliponaxis=False))
    xmax = float(np.nanmax(np.append(xs, overall)))
    fig.add_shape(type="line", x0=overall, x1=overall, y0=0, y1=1, xref="x", yref="paper",
                  line=dict(color=INK, width=1.5, dash="dash"))
    _legend_proxy(fig, f"Overall late rate ({_pct(overall)})", INK, dash="dash")
    fig.update_xaxes(title=xtitle, tickformat=".0%", range=[0, xmax * 1.2], showgrid=True, gridcolor=GRID,
                     showline=False, ticks="")
    fig.update_yaxes(title=ytitle, showgrid=False, ticks="", autorange="reversed", categoryorder="array",
                     categoryarray=order)
    fig.update_layout(bargap=0.3)
    notes = []
    if n_hidden_big:
        notes.append(f"{n_hidden_big} more reliable segments not shown")
    if n_hidden_small:
        notes.append(f"{n_hidden_small} segments with n < {min_orders:,} hidden")
    subtitle = " · ".join([s for s in [subtitle_extra, *notes] if s]) or None
    return _finish(fig, title, subtitle, height=max(320, 27 * len(d) + 190), legend=True, left=16, top=None,
                   wrap=_NARROW_WRAP, legend_px=_NARROW_LEGEND_PX)


# --------------------------------------------------------------------------- #
# Overview charts
# --------------------------------------------------------------------------- #
def status_donut(data: pd.DataFrame | Mapping[str, Any] | None, *, title: str = "Delivery outcome",
                 subtitle: str | None = None) -> go.Figure:
    """On-time vs late share. ``data`` = filtered orders frame (``is_late``) or ``analytics.kpi_summary`` dict."""
    n_total = n_late = None
    if isinstance(data, pd.DataFrame):
        if "is_late" in data.columns:
            v = pd.to_numeric(data["is_late"], errors="coerce").dropna()
            n_total, n_late = len(v), int((v == 1).sum())
    elif isinstance(data, Mapping):
        try:
            n_total, n_late = int(data.get("n_orders", 0)), int(data.get("n_late", 0))
        except (TypeError, ValueError):
            n_total = None
    if not n_total or n_total <= 0:
        return empty_figure(title=title, height=340)
    n_on = n_total - n_late
    fig = go.Figure(go.Pie(
        labels=["On time (or early)", "Late"], values=[n_on, n_late], hole=0.64, sort=False,
        direction="clockwise", rotation=0,
        marker=dict(colors=[PALETTE["green"], PALETTE["red"]], line=dict(color="#FFFFFF", width=3)),
        texttemplate="%{percent:.1%}", textposition="outside", textfont=dict(size=13, color=INK_2),
        hovertemplate="<b>%{label}</b><br>%{value:,} orders (%{percent})<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b style='font-size:26px;color:{INK}'>{_pct(n_late / n_total)}</b><br>"
             f"<span style='color:{MUTED}'>late</span><br><span style='color:{MUTED};font-size:11px'>"
             f"n = {n_total:,} orders</span>",
        x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=13))
    sub = subtitle or "Late = delivered after the estimated delivery date (calendar-date basis)"
    _finish(fig, title, sub, height=380, legend=True, top=78, bottom=56)
    fig.update_layout(legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.06, yanchor="top"))
    return fig


def monthly_performance_chart(monthly: pd.DataFrame | None, *, metric: str = "both",
                              min_orders: int = DEFAULT_MIN_ORDERS,
                              title: str = "Monthly delivery performance") -> go.Figure:
    """Late rate (with Wilson band) and on-time rate lines plus order volume, one panel each (shared x axis).

    ``monthly`` = ``analytics.monthly_performance`` output.  ``metric`` is ``"late"``, ``"on_time"`` or
    ``"both"``.  Months with ``n_orders < min_orders`` (e.g. 2016) are drawn as hollow grey markers and
    left out of the lines.  Separate panels are used instead of a dual y axis so each scale stays honest.
    """
    if not _frame_ok(monthly, ["purchase_year_month", "n_orders"], "n_orders"):
        return empty_figure(title=title)
    d = monthly.copy()
    d["_x"] = _month_start(d["purchase_year_month"])
    d = d[d["_x"].notna()].sort_values("_x")
    if d.empty:
        return empty_figure(title=title)
    d["_n"] = _num(d, "n_orders").fillna(0)
    d["_late"] = _num(d, "late_rate")
    d["_on"] = _num(d, "on_time_rate")
    ok = d["_n"] >= min_orders
    n_small = int((~ok).sum())
    step = _step(d["_x"])
    panels = []
    if metric in ("late", "both"):
        panels.append("late")
    if metric in ("on_time", "both"):
        panels.append("on_time")
    if not panels:
        panels = ["late"]
    nrows = len(panels) + 1
    heights = [0.34] * len(panels) + [0.24] if len(panels) == 2 else ([0.62, 0.38] if len(panels) == 1 else None)
    if len(panels) == 2:
        heights = [0.37, 0.37, 0.26]
    fig = make_subplots(rows=nrows, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=heights)
    for r, p in enumerate(panels, start=1):
        col = "_late" if p == "late" else "_on"
        color = PALETTE["red"] if p == "late" else PALETTE["green"]
        name = "Late rate" if p == "late" else "On-time rate"
        good = d[ok & d[col].notna()]
        if p == "late" and "late_ci_low" in d.columns:
            for k, run in enumerate(_contiguous_runs(good, step)):
                lo, hi = _num(run, "late_ci_low"), _num(run, "late_ci_high")
                fig.add_trace(go.Scatter(x=pd.concat([run["_x"], run["_x"][::-1]]),
                                         y=pd.concat([hi, lo[::-1]]), fill="toself",
                                         fillcolor="rgba(220,38,38,0.12)", mode="lines", line=dict(width=0),
                                         hoverinfo="skip", name="95% interval (Wilson)", legendgroup="ci",
                                         showlegend=(k == 0)), row=r, col=1)
        gp = _gapped(good, step)
        cd = np.column_stack([gp["_n"], _num(gp, "n_late") if "n_late" in gp.columns else np.full(len(gp), np.nan)])
        fig.add_trace(go.Scatter(
            x=gp["_x"], y=gp[col], mode="lines+markers", name=name,
            line=dict(color=color, width=2.5), marker=dict(size=8, color=color, line=dict(color="#FFFFFF", width=2)),
            customdata=cd,
            hovertemplate=name + ": <b>%{y:.1%}</b><br>Orders in month: %{customdata[0]:,.0f}<extra></extra>",
        ), row=r, col=1)
        rng = _rate_range(good, col, p == "late", "late_ci_high" if p == "late" else None,
                          "late_ci_low" if p == "late" else None)
        fig.update_yaxes(title=f"{name} (% of orders)", tickformat=_pct_tickformat(rng), range=rng, row=r, col=1)
    vr = nrows
    fig.add_trace(go.Bar(
        x=d["_x"][ok], y=d["_n"][ok], name="Orders", marker=dict(color=PALETTE["blue"]),
        hovertemplate="Orders in month: <b>%{y:,}</b><extra></extra>", showlegend=True,
    ), row=vr, col=1)
    if n_small:
        fig.add_trace(go.Bar(
            x=d["_x"][~ok], y=d["_n"][~ok], name=f"Small month (n < {min_orders:,})", marker=dict(color=NEUTRAL_LIGHT),
            hovertemplate="Orders in month: <b>%{y:,}</b> (too few to trust rates)<extra></extra>", showlegend=True,
        ), row=vr, col=1)
    fig.update_yaxes(title="Orders per month (count)", tickformat=",", row=vr, col=1)
    fig.update_xaxes(title="Purchase month", tickformat="%b %Y", row=vr, col=1)
    if len(d) <= 12:       # short windows: exactly one tick per month (no repeated "Oct 2016" sub-month ticks)
        fig.update_xaxes(dtick="M1", row=vr, col=1)
    fig.update_xaxes(showline=True, ticks="outside", row=vr, col=1)
    fig.update_layout(hovermode="x unified", bargap=0.25)
    sub = f"Grey months (n < {min_orders:,}, {n_small} here) are left out of the lines: their rate is too noisy" \
        if n_small else "Rates by purchase month; shaded band = 95% Wilson interval"
    return _finish(fig, title, sub, height=230 * nrows + 120, legend=True,
                   wrap=_NARROW_WRAP, legend_px=_NARROW_LEGEND_PX)


def duration_histogram(df: pd.DataFrame | pd.Series | None, col: str = "delivery_duration_days", *,
                       title: str = "Delivery time distribution", xtitle: str = "Delivery time (days from purchase)",
                       cap_quantile: float = 0.99, bin_width: float = 1.0, retrospective: bool = False,
                       color: str | None = None) -> go.Figure:
    """Histogram of a duration column (default delivery time) with mean/median lines and a stats box."""
    vals = _values_from(df, col)
    if vals is None:
        return empty_figure(title=title)
    return _histogram(vals, title=title, xtitle=xtitle, cap_quantile=cap_quantile, bin_width=bin_width,
                      color=color or PALETTE["blue"],
                      subtitle="Retrospective: known only after the order was handled" if retrospective else None)


def _values_from(df: Any, col: str) -> np.ndarray | None:
    if isinstance(df, pd.Series):
        s = pd.to_numeric(df, errors="coerce")
    elif isinstance(df, pd.DataFrame) and col in df.columns:
        s = _num(df, col)
    else:
        return None
    v = s.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype="float64")
    return v if v.size else None


def _histogram(v: np.ndarray, *, title: str, xtitle: str, cap_quantile: float, bin_width: float,
               color: str, subtitle: str | None) -> go.Figure:
    v = v[v >= 0] if (v >= 0).any() else v
    if v.size == 0:
        return empty_figure(title=title)
    n = int(v.size)
    mean, median = float(v.mean()), float(np.median(v))
    p90, p95 = float(np.quantile(v, 0.90)), float(np.quantile(v, 0.95))
    cap = max(float(np.quantile(v, cap_quantile)), bin_width)
    top = math.ceil(cap / bin_width) * bin_width
    edges = np.arange(0, top + bin_width, bin_width)
    counts, _ = np.histogram(v[v <= edges[-1]], bins=edges)
    n_beyond = int((v > edges[-1]).sum())
    centres = edges[:-1] + bin_width / 2
    fig = go.Figure(go.Bar(
        x=centres, y=counts, width=bin_width * 0.94, marker=dict(color=color), name="Orders",
        customdata=np.column_stack([edges[:-1], edges[1:], counts / n]),
        hovertemplate="%{customdata[0]:.0f} to %{customdata[1]:.0f} days<br>Orders: <b>%{y:,}</b> "
                      "(%{customdata[2]:.1%} of all)<extra></extra>"))
    ymax = float(counts.max()) if counts.size else 1.0
    for val, name, c, dash in ((mean, f"Mean ({mean:.1f} d)", "#0891B2", "dash"),
                               (median, f"Median ({median:.1f} d)", INK, "solid")):
        fig.add_shape(type="line", x0=val, x1=val, y0=0, y1=1, xref="x", yref="paper",
                      line=dict(color=c, width=2, dash=dash))
        _legend_proxy(fig, name, c, dash=dash)
    stats = (f"<b>n = {n:,}</b><br>Mean {mean:.1f} d · Median {median:.1f} d<br>"
             f"90th pct {p90:.1f} d · 95th pct {p95:.1f} d")
    fig.add_annotation(text=stats, xref="paper", yref="paper", x=0.99, y=0.62, xanchor="right", yanchor="top",   # the right-skewed tail leaves this area empty
                       showarrow=False, align="left", bgcolor="#FFFFFF", bordercolor=BASELINE, borderwidth=1,
                       borderpad=8, font=dict(size=12, color=INK_2))
    fig.update_xaxes(title=xtitle, range=[0, top])
    fig.update_yaxes(title="Orders (count)", tickformat=",", range=[0, ymax * 1.12])
    fig.update_layout(bargap=0.04)
    sub = subtitle
    if n_beyond:
        extra = f"{n_beyond:,} orders above {top:g} days not shown (axis capped at the {cap_quantile:.0%} quantile)"
        sub = f"{sub} · {extra}" if sub else extra
    return _finish(fig, title, sub, height=420, legend=True, wrap=_NARROW_WRAP, legend_px=_NARROW_LEGEND_PX)


# --------------------------------------------------------------------------- #
# Geography
# --------------------------------------------------------------------------- #
def state_late_rate_bar(state_perf: pd.DataFrame | None, *, min_orders: int = DEFAULT_MIN_ORDERS,
                        overall_rate: float | None = None, top_n: int | None = None,
                        include_small: bool = True, ascending: bool = False,
                        title: str = "Late-delivery rate by customer state") -> go.Figure:
    """Sorted horizontal bars of late rate per state (``analytics.state_performance`` output).

    Whiskers = Wilson interval; red/green bars = interval excludes the overall rate; grey hatched bars =
    fewer than ``min_orders`` orders (kept at the bottom, de-emphasised).  Tick labels carry ``n``.
    """
    return _segment_rate_bar(
        state_perf, "customer_state" if (state_perf is not None and "customer_state" in state_perf.columns)
        else "seller_state", title=title, xtitle="Late rate (% of orders)", ytitle="State",
        min_orders=min_orders, overall_rate=overall_rate, top_n=top_n, include_small=include_small,
        ascending=ascending, label_fn=lambda k: str(k), name_fn=lambda k: f"{k} · {_state_label(k)}",
        subtitle_extra="Whiskers = 95% Wilson interval; n is beside each state")


def category_late_rate_bar(cat_perf: pd.DataFrame | None, *, top_n: int = 15,
                           min_orders: int = DEFAULT_MIN_ORDERS, overall_rate: float | None = None,
                           include_small: bool = False, ascending: bool = False,
                           title: str | None = None) -> go.Figure:
    """Late rate per product category (``analytics.category_performance``); top/bottom ``top_n`` reliable ones."""
    t = title or (f"Late-delivery rate by product category ({'lowest' if ascending else 'highest'} {top_n})")
    return _segment_rate_bar(
        cat_perf, "product_category", title=t, xtitle="Late rate (% of orders)", ytitle="Product category",
        min_orders=min_orders, overall_rate=overall_rate, top_n=top_n, include_small=include_small,
        ascending=ascending, label_fn=lambda k: pretty_category(k, 30), name_fn=pretty_category,
        subtitle_extra="Whiskers = 95% Wilson interval; categories with few orders are hidden by default")


def _perf_metric_bar(df: pd.DataFrame | None, key: str, metric: str, *, top_n: int, min_orders: int,
                     ascending: bool, title: str, xtitle: str, ytitle: str, value_fmt: str, unit: str,
                     label_fn, name_fn, include_small: bool = False, color: str = PALETTE["blue"],
                     ref_label: str | None = None, ref_value: float | None = "weighted",
                     extra_cols: Sequence[tuple[str, str, str]] = (), max_small: int | None = None) -> go.Figure:
    if not _frame_ok(df, [key, "n_orders", metric], metric):
        return empty_figure(title=title)
    d = df.copy()
    d["_v"] = _num(d, metric)
    d["_n"] = _num(d, "n_orders").fillna(0)
    d = d[d["_v"].notna() & (d["_n"] > 0)]
    if d.empty:
        return empty_figure(title=title)
    w = d["_n"].to_numpy()
    ref = float(np.average(d["_v"], weights=w)) if (ref_value == "weighted" and w.sum() > 0) else (
        None if ref_value == "weighted" else ref_value)
    d["_small"] = d["_n"] < min_orders
    big = d[~d["_small"]].sort_values(["_v", "_n"], ascending=[ascending, False], kind="mergesort")
    small = d[d["_small"]].sort_values(["_v", "_n"], ascending=[ascending, False], kind="mergesort")
    hidden = max(len(big) - top_n, 0)
    big = big.head(top_n)
    n_small_hidden = 0
    if include_small:
        cap = max_small if max_small is not None else max(3, top_n // 3)
        n_small_hidden, small = max(len(small) - cap, 0), small.head(cap)
    else:
        n_small_hidden, small = len(small), small.iloc[0:0]
    d = pd.concat([big, small])
    if d.empty:
        return empty_figure(title=title, hint=f"All segments have fewer than {min_orders:,} orders.")
    labels = [f"{label_fn(k)}  ·  n={int(n):,}" for k, n in zip(d[key], d["_n"])]
    hover = []
    for _, r in d.iterrows():
        extra = "".join(f"<br>{lbl}: {fmt.format(r[c])}" for c, lbl, fmt in extra_cols if c in d.columns and pd.notna(r[c]))
        hover.append(f"<b>{name_fn(r[key])}</b><br>{xtitle.split(' (')[0]}: <b>{r['_v']:{value_fmt}} {unit}</b>"
                     f"<br>Orders: {int(r['_n']):,}{extra}")
    text = [f"{v:{value_fmt}}" for v in d["_v"]]
    notes = []
    if hidden:
        notes.append(f"{hidden} more reliable segments not shown")
    if n_small_hidden:
        notes.append(f"{n_small_hidden} segments with n < {min_orders:,} hidden")
    return _plain_hbar(labels, d["_v"], title=title, subtitle=" · ".join(notes) or None, xtitle=xtitle,
                       ytitle=ytitle, hover=hover, text=text, color=color, ref_value=ref,
                       ref_name=(f"{ref_label} ({ref:{value_fmt}} {unit})" if ref is not None and ref_label else None),
                       small=list(d["_small"]), small_label=f"Small sample (n < {min_orders:,})")


def category_duration_bar(cat_perf: pd.DataFrame | None, *, top_n: int = 15,
                          min_orders: int = DEFAULT_MIN_ORDERS, metric: str = "mean_duration",
                          ascending: bool = False, include_small: bool = False,
                          title: str | None = None) -> go.Figure:
    """Mean (or median) delivery time per category (``analytics.category_performance``)."""
    word = "Median" if metric == "median_duration" else "Mean"
    t = title or f"{word} delivery time by product category ({'shortest' if ascending else 'longest'} {top_n})"
    return _perf_metric_bar(
        cat_perf, "product_category", metric, top_n=top_n, min_orders=min_orders, ascending=ascending, title=t,
        xtitle=f"{word} delivery time (days)", ytitle="Product category", value_fmt=".1f", unit="days",
        label_fn=lambda k: pretty_category(k, 30), name_fn=pretty_category, include_small=include_small,
        ref_label="Order-weighted average",
        extra_cols=[("late_rate", "Late rate", "{:.1%}")])


def category_freight_bar(cost_df: pd.DataFrame | None, *, top_n: int = 15, metric: str = "mean_freight",
                         min_orders: int = DEFAULT_MIN_ORDERS, ascending: bool = False,
                         include_small: bool = False, title: str | None = None) -> go.Figure:
    """Freight cost per category (``analytics.category_cost_comparison``).

    ``metric`` is ``"mean_freight"`` / ``"median_freight"`` (R$ per order) or ``"mean_freight_ratio"`` /
    ``"median_freight_ratio"`` (freight as a share of item price).
    """
    ratio = metric.endswith("ratio")
    kind = "Median" if metric.startswith("median") else "Mean"
    what = "freight / item price" if ratio else "freight cost per order"
    t = title or f"{kind} {what} by product category ({'lowest' if ascending else 'highest'} {top_n})"
    d = cost_df
    if isinstance(d, pd.DataFrame) and metric in d.columns and ratio:
        d = d.copy()
        d[metric] = _num(d, metric) * 100.0
    return _perf_metric_bar(
        d, "product_category", metric, top_n=top_n, min_orders=min_orders, ascending=ascending, title=t,
        xtitle=(f"{kind} freight as % of item price" if ratio else f"{kind} freight cost per order (R$)"),
        ytitle="Product category", value_fmt=".1f", unit="%" if ratio else "R$", label_fn=lambda k: pretty_category(k, 30),
        name_fn=pretty_category, include_small=include_small, ref_label="Order-weighted average",
        extra_cols=[("mean_price", "Mean item value (R$)", "{:,.0f}"), ("mean_weight_g", "Mean weight (g)", "{:,.0f}"),
                    ("late_rate", "Late rate", "{:.1%}")])


def state_duration_bar(state_perf: pd.DataFrame | None, *, top_n: int = 27,
                       min_orders: int = DEFAULT_MIN_ORDERS, metric: str = "mean_duration",
                       ascending: bool = False, include_small: bool = True,
                       title: str | None = None, max_small: int | None = None) -> go.Figure:
    """Mean (or median) delivery time per customer state (``analytics.state_performance``).

    ``max_small`` caps how many small-sample states are drawn (default ``max(3, top_n // 3)``; pass a large
    number to draw every state).  Segments left out are counted in the subtitle.
    """
    word = "Median" if metric == "median_duration" else "Mean"
    t = title or f"{word} delivery time by customer state ({'shortest' if ascending else 'longest'} first)"
    return _perf_metric_bar(
        state_perf, "customer_state", metric, top_n=top_n, min_orders=min_orders, ascending=ascending, title=t,
        xtitle=f"{word} delivery time (days)", ytitle="Customer state", value_fmt=".1f", unit="days",
        label_fn=lambda k: str(k), name_fn=lambda k: f"{k} · {_state_label(k)}", include_small=include_small,
        ref_label="Order-weighted average", extra_cols=[("late_rate", "Late rate", "{:.1%}")], max_small=max_small)


def flagged_months_chart(monthly: pd.DataFrame | None, screen: pd.DataFrame | None = None, *,
                         min_orders: int = DEFAULT_MIN_ORDERS, alpha: float = 0.05,
                         title: str = "Monthly late rate, flagged months in red") -> go.Figure:
    """Late rate per purchase month (``analytics.monthly_performance``) with flagged months in red.

    ``screen`` = ``insights.month_screen`` output (each reliable month vs all other months, Bonferroni
    p).  Red = higher than the other months at the Bonferroni-adjusted ``alpha``; blue = reliable but not
    flagged; hatched grey = fewer than ``min_orders`` orders (not tested).  Dashed line = pooled late rate.
    """
    if not _frame_ok(monthly, ["purchase_year_month", "n_orders"], "n_orders"):
        return empty_figure(title=title)
    d = monthly.copy()
    d["_x"] = pd.to_datetime(d["purchase_year_month"].astype(str) + "-01", errors="coerce")
    d = d[d["_x"].notna()].sort_values("_x")
    d["_n"] = _num(d, "n_orders").fillna(0)
    d["_k"] = _num(d, "n_late").fillna(0) if "n_late" in d.columns else (_num(d, "late_rate") * d["_n"]).round()
    d = d[d["_n"] > 0]
    if d.empty:
        return empty_figure(title=title)
    d["_r"] = d["_k"] / d["_n"]
    scr = pd.DataFrame(columns=["month", "diff", "p_bonferroni", "rest_late_rate"]) if screen is None else screen
    sc = scr.set_index("month") if "month" in scr.columns and len(scr) else pd.DataFrame()
    flagged, extra = set(), {}
    for m, r in sc.iterrows():
        extra[m] = r
        if pd.notna(r.get("diff")) and r["diff"] > 0 and pd.notna(r.get("p_bonferroni")) and r["p_bonferroni"] < alpha:
            flagged.add(m)
    d["_state"] = np.where(d["_n"] < min_orders, "small",
                           np.where(d["purchase_year_month"].isin(flagged), "flag", "ok"))
    hover = []
    for _, r in d.iterrows():
        h = (f"<b>{r['_x']:%b %Y}</b><br>Late rate: <b>{r['_r']:.1%}</b><br>Late orders: {int(r['_k']):,} of {int(r['_n']):,}")
        e = extra.get(r["purchase_year_month"])
        if e is not None:
            pb = float(e["p_bonferroni"])
            h += (f"<br>Other months: {e['rest_late_rate']:.1%}<br>Adjusted p (Bonferroni): "
                  + ("< 0.001" if pb < 0.001 else f"{pb:.3f}"))
        elif r["_state"] == "small":
            h += f"<br><i>n < {min_orders:,}: too few orders to test</i>"
        hover.append(h)
    d["_h"] = hover
    fig = go.Figure()
    styles = {"flag": ("Significantly above other months", dict(color=PALETTE["red"])),
              "ok": ("Not flagged", dict(color=PALETTE["blue_light"])),
              "small": (f"Small sample (n < {min_orders:,}), not tested",
                        dict(color=NEUTRAL_LIGHT, pattern=dict(shape="/", fgcolor="#FFFFFF", bgcolor=NEUTRAL_LIGHT, size=6)))}
    for key in ("flag", "ok", "small"):
        sub = d[d["_state"] == key]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(x=sub["_x"], y=sub["_r"], name=styles[key][0], marker=styles[key][1],
                             customdata=sub["_h"], hovertemplate="%{customdata}<extra></extra>"))
    overall = float(d["_k"].sum() / d["_n"].sum())
    fig.add_shape(type="line", xref="paper", x0=0, x1=1, y0=overall, y1=overall,
                  line=dict(color=INK, width=1.5, dash="dash"))
    _legend_proxy(fig, f"Pooled late rate ({overall:.1%})", INK, dash="dash")
    testable = d[d["_state"] != "small"]
    top = float((testable if len(testable) else d)["_r"].max())   # tiny months must not stretch the axis
    fig.update_yaxes(title="Late rate (% of orders)", tickformat=".0%", range=[0, max(top * 1.15, overall * 1.5, 0.02)])
    fig.update_xaxes(title="Purchase month", tickformat="%b %Y", showline=True, ticks="outside")
    fig.update_layout(bargap=0.25)
    return _finish(fig, title, f"{len(flagged)} of {int((d['_state'] != 'small').sum())} testable months flagged "
                   f"(Bonferroni-adjusted, {int(round(100 * (1 - alpha)))}% confidence)", height=420, legend=True)


# --------------------------------------------------------------------------- #
# Weight / freight / processing
# --------------------------------------------------------------------------- #
def _log_ticks(lo: float, hi: float) -> list[float]:
    """Readable tick values for a log axis spanning ``lo``..``hi`` (1-2-5 steps; finer steps for narrow ranges)."""
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0 or hi <= lo:
        return []
    best: list[float] = []
    for mult in ((1,), (1, 2, 5), (1, 2, 3, 5, 7), (1, 1.5, 2, 3, 4, 5, 6, 8)):
        vals = [m * 10.0 ** e for e in range(int(math.floor(math.log10(lo))) - 1, int(math.ceil(math.log10(hi))) + 1)
                for m in mult]
        vals = [v for v in vals if lo <= v <= hi]
        best = vals
        if 3 <= len(vals) <= 7:
            break
    return best


def _vs_duration(binned: pd.DataFrame | None, scatter: pd.DataFrame | None, *, xcol: str, title: str,
                 xtitle: str, log_x: bool, unit: str, min_orders: int, subtitle: str) -> go.Figure:
    if not _frame_ok(binned, ["mean_value", "mean_duration", "n_orders"], "mean_duration"):
        return empty_figure(title=title)
    b = binned.copy()
    b["_x"], b["_y"], b["_n"] = _num(b, "mean_value"), _num(b, "mean_duration"), _num(b, "n_orders").fillna(0)
    b = b[b["_x"].notna() & b["_y"].notna() & (b["_n"] > 0)].sort_values("_x")
    if b.empty:
        return empty_figure(title=title)
    xmax_data = float(b["_x"].max())
    fig = go.Figure()
    have_sc = _frame_ok(scatter, [xcol, "delivery_duration_days"])
    if have_sc:
        s = scatter.copy()
        s["_x"], s["_y"] = _num(s, xcol), _num(s, "delivery_duration_days")
        s = s[s["_x"].notna() & s["_y"].notna()]
        if log_x:
            s = s[s["_x"] > 0]
        have_sc = not s.empty
    if have_sc:
        xmax_data = max(xmax_data, float(s["_x"].quantile(0.995)))
        late = _num(s, "is_late") == 1 if "is_late" in s.columns else pd.Series(False, index=s.index)
        for mask, name, c in ((~late, "Sampled order: on time", "rgba(37,99,235,0.28)"),
                              (late, "Sampled order: late", "rgba(220,38,38,0.40)")):
            sub = s[mask]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["_x"], y=sub["_y"], mode="markers", name=name, marker=dict(size=4, color=c),
                hovertemplate=f"{xtitle.split(' (')[0]}: %{{x:,.1f}} {unit}<br>Delivery time: %{{y:.1f}} days<extra></extra>"))
    small = b["_n"] < min_orders
    cd = np.column_stack([b["bin"].astype(str), b["_n"], _num(b, "late_rate")])
    fig.add_trace(go.Scatter(
        x=b["_x"], y=b["_y"], mode="lines+markers", name="Mean delivery time per bin",
        line=dict(color=INK, width=2.5),
        marker=dict(size=11, color=[NEUTRAL if sm else INK for sm in small], line=dict(color="#FFFFFF", width=2)),
        customdata=cd,
        hovertemplate=("Bin: <b>%{customdata[0]}</b> " + unit + "<br>Mean delivery time: <b>%{y:.1f} days</b><br>"
                       "Orders in bin: %{customdata[1]:,.0f}<br>Late rate: %{customdata[2]:.1%}<extra></extra>")))
    if bool(small.any()):
        sub = b[small]
        fig.add_trace(go.Scatter(x=sub["_x"], y=sub["_y"], mode="markers", name=f"Bin with n < {min_orders:,}",
                                 marker=dict(size=11, color="#FFFFFF", line=dict(color=NEUTRAL, width=2)),
                                 hoverinfo="skip"))
    xaxis = dict(title=xtitle + (" – log scale" if log_x else ""))
    xmin_data = float(b["_x"].min())
    ymax_data = float(b["_y"].max())
    if have_sc:
        xmin_data = min(xmin_data, float(s["_x"].quantile(0.005)))
        ymax_data = max(ymax_data, float(s["_y"].quantile(0.995)))
    if log_x:
        xaxis.update(type="log", range=[math.log10(max(xmin_data, 1e-3)) - 0.08, math.log10(xmax_data) + 0.08])
    else:
        xaxis.update(range=[0, xmax_data * 1.05])
    fig.update_xaxes(**xaxis, tickformat=",")
    if log_x:
        ticks = _log_ticks(10 ** xaxis["range"][0], 10 ** xaxis["range"][1])
        if ticks:
            fig.update_xaxes(tickmode="array", tickvals=ticks, ticktext=[f"{t:,.0f}" if t >= 10 else f"{t:g}" for t in ticks],
                             minor=dict(showgrid=False, ticks=""))
    fig.update_yaxes(title="Delivery time (days from purchase)", range=[0, ymax_data * 1.1])
    sub_t = subtitle + (" · scatter = random sample, axes trimmed at the 99.5th percentile" if have_sc else "")
    return _finish(fig, title, sub_t, height=460, legend=True)


def weight_vs_duration(weight_bins_df: pd.DataFrame | None, scatter: pd.DataFrame | None = None, *,
                       min_orders: int = DEFAULT_MIN_ORDERS, log_x: bool = True,
                       title: str = "Delivery time by order weight") -> go.Figure:
    """Binned mean delivery time vs weight (``analytics.weight_bins``) over an optional sampled scatter
    (``analytics.scatter_sample(df, "total_weight_g", "delivery_duration_days")``)."""
    return _vs_duration(weight_bins_df, scatter, xcol="total_weight_g", title=title, xtitle="Order weight (g)",
                        log_x=log_x, unit="g", min_orders=min_orders,
                        subtitle="Association only; heavier orders also differ in category, route and seller")


def freight_vs_duration(freight_bins_df: pd.DataFrame | None, scatter: pd.DataFrame | None = None, *,
                        min_orders: int = DEFAULT_MIN_ORDERS, log_x: bool = False,
                        title: str = "Delivery time by freight cost") -> go.Figure:
    """Binned mean delivery time vs freight cost (``analytics.freight_bins``) + optional sampled scatter."""
    return _vs_duration(freight_bins_df, scatter, xcol="total_freight", title=title, xtitle="Freight cost (R$)",
                        log_x=log_x, unit="R$", min_orders=min_orders,
                        subtitle="Freight is partly a function of distance, so this is an association, not a cause")


def _compact_edge(v: float) -> str:
    """``500`` -> ``500``, ``1000`` -> ``1k``, ``2800`` -> ``2.8k``, ``184400`` -> ``184k``."""
    if abs(v) < 1000:
        return f"{v:g}"
    k = v / 1000.0
    return f"{k:.0f}k" if (abs(k) >= 100 or abs(k - round(k)) < 0.05) else f"{k:.1f}k"


def _compact_bin_labels(d: pd.DataFrame) -> list[str]:
    """Short x tick labels (``0–500``, ``1k–2k``, ``> 30k``) from ``bin_low`` / ``bin_high``; falls back to ``bin``."""
    full = [str(b) for b in d["bin"]]
    if "bin_low" not in d.columns or "bin_high" not in d.columns:
        return full
    out = []
    for lo, hi in zip(_num(d, "bin_low"), d["bin_high"].astype(float)):
        if not np.isfinite(lo):
            return full
        out.append(f"> {_compact_edge(lo)}" if not np.isfinite(hi) else f"{_compact_edge(lo)}–{_compact_edge(hi)}")
    return out if len(set(out)) == len(out) else full


def _binned_late_bars(binned: pd.DataFrame | None, *, title: str, xtitle: str, unit: str, min_orders: int,
                      overall_rate: float | None, subtitle: str | None) -> go.Figure:
    if not _frame_ok(binned, ["bin", "n_orders", "late_rate"], "late_rate"):
        return empty_figure(title=title)
    d = binned.copy()
    d["_r"], d["_n"] = _num(d, "late_rate"), _num(d, "n_orders").fillna(0)
    d = d[d["_r"].notna() & (d["_n"] > 0)]
    if "bin_low" in d.columns and d["bin_low"].notna().any():
        d = d.sort_values("bin_low", kind="mergesort")
    if d.empty:
        return empty_figure(title=title)
    overall = _overall_rate(d, overall_rate)
    small = (d["_n"] < min_orders).to_numpy()
    ci = _ci_arrays(d)
    cats = [f"{b}<br><span style='font-size:10px'>n={int(n):,}</span>" for b, n in zip(_compact_bin_labels(d), d["_n"])]
    colors = [NEUTRAL_LIGHT if s else PALETTE["blue"] for s in small]
    cd = np.column_stack([d["bin"].astype(str), d["_n"], _num(d, "n_late") if "n_late" in d.columns else np.full(len(d), np.nan),
                          _num(d, "late_ci_low"), _num(d, "late_ci_high")])
    fig = go.Figure(go.Bar(
        x=cats, y=d["_r"], marker=dict(color=colors, pattern=dict(shape=["/" if s else "" for s in small],
                                                                 fgcolor="#FFFFFF", bgcolor=colors, size=6)),
        error_y=dict(type="data", symmetric=False, array=ci[0], arrayminus=ci[1], color=INK_2, thickness=1.2, width=3)
        if ci is not None else None,
        customdata=cd, showlegend=False,
        hovertemplate=("Bin: <b>%{customdata[0]} " + unit + "</b><br>Late rate: <b>%{y:.1%}</b><br>"
                       "Late orders: %{customdata[2]:,.0f} of %{customdata[1]:,.0f}<br>"
                       "95% CI: %{customdata[3]:.1%} to %{customdata[4]:.1%}<extra></extra>")))
    ytop = (d["_r"] + (ci[0] if ci is not None else 0)).to_numpy()
    fig.add_hline(y=overall, line=dict(color=INK, width=1.5, dash="dash"), layer="below")
    for cx, cy, r in zip(cats, ytop, d["_r"]):     # value labels sit on a white chip so the dashed line never cuts through them
        fig.add_annotation(x=cx, y=float(cy), text=_pct(r), showarrow=False, yanchor="bottom", yshift=3,
                           font=dict(size=11, color=INK_2), bgcolor="rgba(255,255,255,0.88)", borderpad=1)
    _legend_proxy(fig, f"Overall late rate ({_pct(overall)})", INK, dash="dash")
    legend = True
    if small.any():
        _legend_proxy(fig, f"Small sample (n < {min_orders:,})", NEUTRAL_LIGHT, pattern=True)
    ymax = float(np.nanmax((d["_r"] + (ci[0] if ci is not None else 0)).to_numpy()))
    fig.update_yaxes(title="Late rate (% of orders)", tickformat=".0%", range=[0, max(ymax, overall) * 1.2])
    fig.update_xaxes(title=xtitle, type="category", categoryorder="array", categoryarray=cats, tickangle=0)
    fig.update_layout(bargap=0.3)
    return _finish(fig, title, subtitle, height=440, legend=legend)


def weight_late_rate_bars(weight_bins_df: pd.DataFrame | None, *, overall_rate: float | None = None,
                          min_orders: int = DEFAULT_MIN_ORDERS,
                          title: str = "Late rate by order weight") -> go.Figure:
    """Late rate per weight bin (``analytics.weight_bins``)."""
    return _binned_late_bars(weight_bins_df, title=title, xtitle="Order weight bin (g)", unit="g",
                             min_orders=min_orders, overall_rate=overall_rate,
                             subtitle="Whiskers = 95% Wilson interval; n under each bin")


def freight_late_rate_bars(freight_bins_df: pd.DataFrame | None, *, overall_rate: float | None = None,
                           min_orders: int = DEFAULT_MIN_ORDERS,
                           title: str = "Late rate by freight cost") -> go.Figure:
    """Late rate per freight-cost bin (``analytics.freight_bins``)."""
    return _binned_late_bars(freight_bins_df, title=title, xtitle="Freight cost bin (R$)", unit="R$",
                             min_orders=min_orders, overall_rate=overall_rate,
                             subtitle="Whiskers = 95% Wilson interval; n under each bin")


def processing_histogram(df: pd.DataFrame | pd.Series | None, col: str = "processing_days", *,
                         title: str = "Seller processing time (purchase to carrier hand-over)",
                         cap_quantile: float = 0.99, bin_width: float = 1.0) -> go.Figure:
    """Histogram of ``processing_days`` (retrospective: only known after the seller has shipped)."""
    vals = _values_from(df, col)
    if vals is None:
        return empty_figure(title=title)
    return _histogram(vals, title=title, xtitle="Processing time (days from purchase to carrier hand-over)",
                      cap_quantile=cap_quantile, bin_width=bin_width, color=PALETTE["blue"],
                      subtitle="Retrospective: known only after the order has shipped, so it cannot be used to predict at checkout")


def late_rate_by_processing_bucket(binned: pd.DataFrame | None, *, overall_rate: float | None = None,
                                   min_orders: int = DEFAULT_MIN_ORDERS,
                                   title: str = "Late rate by seller processing time (retrospective)") -> go.Figure:
    """Late rate per processing-time bucket (``analytics.late_by_processing_bucket``)."""
    return _binned_late_bars(binned, title=title, xtitle="Processing time bucket (days)", unit="days",
                             min_orders=min_orders, overall_rate=overall_rate,
                             subtitle="Retrospective: processing time is only known after shipping; association, not cause")


# --------------------------------------------------------------------------- #
# Regional
# --------------------------------------------------------------------------- #
def state_pair_heatmap(matrices: Mapping[str, pd.DataFrame] | None, *, metric: str = "late_rate",
                       min_orders: int = DEFAULT_MIN_ORDERS, title: str | None = None) -> go.Figure:
    """Seller-state x customer-state heat map from ``analytics.state_matrix`` (``metric`` = ``late_rate``,
    ``mean_duration`` or ``n_orders``).  Cells with fewer than ``min_orders`` orders are masked grey."""
    names = {"late_rate": "Late rate", "mean_duration": "Mean delivery time", "n_orders": "Orders"}
    metric = metric if metric in names else "late_rate"
    t = title or f"{names[metric]} by seller state and customer state"
    if not isinstance(matrices, Mapping) or metric not in matrices or "n_orders" not in matrices:
        return empty_figure(title=t)
    z, n = matrices[metric], matrices["n_orders"]
    if not isinstance(z, pd.DataFrame) or z.empty or not isinstance(n, pd.DataFrame) or n.empty:
        return empty_figure(title=t)
    n = n.reindex(index=z.index, columns=z.columns).fillna(0)
    zv = z.astype("float64").where(n >= min_orders) if metric != "n_orders" else z.astype("float64").where(n > 0)
    if zv.isna().all().all():
        return empty_figure(title=t, hint=f"No seller/customer pair has at least {min_orders:,} orders.")
    keep_r, keep_c = zv.notna().any(axis=1), zv.notna().any(axis=0)
    z, n, zv = z.loc[keep_r, keep_c], n.loc[keep_r, keep_c], zv.loc[keep_r, keep_c]
    vals = zv.to_numpy()
    zmax = float(np.nanquantile(vals, 0.95)) if metric != "n_orders" else float(np.nanmax(vals))
    zmax = zmax if zmax > 0 else float(np.nanmax(vals)) or 1.0
    capped = metric != "n_orders" and float(np.nanmax(vals)) > zmax * 1.001
    if metric == "late_rate":
        scale, cb, fmt, hf = RED_SCALE, "Late rate", ".0%", ".1%"
    elif metric == "mean_duration":
        scale, cb, fmt, hf = BLUE_SCALE, "Mean days", ".0f", ".1f"
    else:
        scale, cb, fmt, hf = BLUE_SCALE, "Orders", ",", ","
    lr = matrices.get("late_rate")
    lr = lr.reindex(index=z.index, columns=z.columns) if isinstance(lr, pd.DataFrame) else None
    extra = lr.to_numpy() if lr is not None else np.full(vals.shape, np.nan)
    cd = np.dstack([n.to_numpy(), extra])
    fig = go.Figure(go.Heatmap(
        z=vals, x=list(z.columns), y=list(z.index), colorscale=scale, zmin=0, zmax=zmax, xgap=2, ygap=2,
        customdata=cd, hoverongaps=False,
        # value printed in every drawn cell (the grid is small), so the reader does not have to decode the colour scale
        texttemplate="%{z:" + hf + "}" if vals.size <= 330 else None, textfont=dict(size=10),
        colorbar=dict(title=dict(text=cb + (" (capped)" if capped else ""), side="right"), thickness=12, len=0.8,
                      tickformat=fmt),
        hovertemplate=("Seller state: <b>%{y}</b> → Customer state: <b>%{x}</b><br>"
                       + names[metric] + ": <b>%{z:" + hf + "}</b><br>Orders: %{customdata[0]:,.0f}<extra></extra>")))
    fig.update_xaxes(title="Customer state (destination)", side="bottom", showgrid=False, type="category",
                     tickangle=0)
    fig.update_yaxes(title="Seller state (origin)", autorange="reversed", showgrid=False, type="category")
    fig.update_layout(plot_bgcolor=MASKED_CELL)
    sub = f"Grey cells have fewer than {min_orders:,} orders and are masked" + (
        "; colour scale capped at the 95th percentile so a few extreme cells do not wash out the rest" if capped else "")
    h = max(420, 24 * len(z.index) + 190)
    return _finish(fig, t, sub, height=min(h, 760), left=16, right=16)


def state_pair_bar(pair_perf: pd.DataFrame | None, *, top_n: int = 15, metric: str = "late_rate",
                   min_orders: int = DEFAULT_MIN_ORDERS, overall_rate: float | None = None,
                   ascending: bool = False, title: str | None = None) -> go.Figure:
    """Top seller→customer routes (``analytics.state_pair_performance``) by ``late_rate``, ``n_orders``
    or ``mean_duration``."""
    if metric == "late_rate":
        t = title or f"Routes with the {'lowest' if ascending else 'highest'} late rate (top {top_n})"
        return _segment_rate_bar(
            pair_perf, "state_pair", title=t, xtitle="Late rate (% of orders)", ytitle="Route (seller → customer state)",
            min_orders=min_orders, overall_rate=overall_rate, top_n=top_n, include_small=False, ascending=ascending,
            label_fn=lambda k: str(k), name_fn=lambda k: str(k),
            subtitle_extra="Only routes with enough orders are ranked; whiskers = 95% Wilson interval")
    if metric == "mean_duration":
        t = title or f"Routes with the {'shortest' if ascending else 'longest'} delivery time (top {top_n})"
        return _perf_metric_bar(pair_perf, "state_pair", "mean_duration", top_n=top_n, min_orders=min_orders,
                                ascending=ascending, title=t, xtitle="Mean delivery time (days)",
                                ytitle="Route (seller → customer state)", value_fmt=".1f", unit="days",
                                label_fn=str, name_fn=str, ref_label="Order-weighted average",
                                extra_cols=[("late_rate", "Late rate", "{:.1%}")])
    t = title or f"Busiest routes (top {top_n} by order volume)"
    d = pair_perf
    if not _frame_ok(d, ["state_pair", "n_orders"], "n_orders"):
        return empty_figure(title=t)
    d = d.copy()
    d["_n"] = _num(d, "n_orders")
    d = d[d["_n"] > 0].sort_values("_n", ascending=ascending, kind="mergesort").head(top_n)
    labels = [str(k) for k in d["state_pair"]]
    hover = [f"<b>{k}</b><br>Orders: <b>{int(n):,}</b><br>Late rate: {_pct(r)}"
             for k, n, r in zip(d["state_pair"], d["_n"], _num(d, "late_rate"))]
    return _plain_hbar(labels, d["_n"], title=t, subtitle="Volume of delivered orders per seller → customer state route",
                       xtitle="Orders (count)", ytitle="Route (seller → customer state)", hover=hover,
                       text=[f"{int(n):,}" for n in d["_n"]], color=PALETTE["blue"], xformat=",")


def state_comparison_chart(state_perf: pd.DataFrame | None, states: Sequence[str] | None = None, *,
                           min_orders: int = DEFAULT_MIN_ORDERS, overall_rate: float | None = None,
                           title: str = "State comparison") -> go.Figure:
    """Selected states side by side: order volume, late rate with Wilson interval, mean delivery time."""
    if not _frame_ok(state_perf, ["customer_state", "n_orders"], "n_orders"):
        return empty_figure(title=title)
    d = state_perf.copy()
    if states:
        wanted = [str(s) for s in states]
        d = d[d["customer_state"].astype(str).isin(wanted)]
        d["_o"] = d["customer_state"].astype(str).map({s: i for i, s in enumerate(wanted)})
        d = d.sort_values("_o")
    else:
        d = d.sort_values("n_orders", ascending=False).head(8)
    d = d[_num(d, "n_orders") > 0]
    if d.empty:
        return empty_figure(title=title, hint="None of the selected states has orders under the current filters.")
    overall = _overall_rate(state_perf, overall_rate)
    labels = [f"{c} · {_state_label(c)}" for c in d["customer_state"]]
    n = _num(d, "n_orders")
    small = (n < min_orders).to_numpy()
    colors = [NEUTRAL_LIGHT if s else PALETTE["blue"] for s in small]
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.09)
    fig.add_trace(go.Bar(x=n, y=labels, orientation="h", marker=dict(color=colors), name="Orders",
                         text=[f"{int(v):,}" for v in n], textposition="outside", cliponaxis=False,
                         textfont=dict(size=11, color=INK_2),
                         hovertemplate="<b>%{y}</b><br>Orders: %{x:,}<extra></extra>", showlegend=False), row=1, col=1)
    rate = _num(d, "late_rate")
    ci = _ci_arrays(d)
    cd = np.column_stack([n, _num(d, "n_late") if "n_late" in d.columns else np.full(len(d), np.nan),
                          _num(d, "late_ci_low"), _num(d, "late_ci_high")])
    fig.add_trace(go.Scatter(
        x=rate, y=labels, mode="markers+text", text=[_pct(r) for r in rate], textposition="top center",
        textfont=dict(size=11, color=INK_2), marker=dict(size=11, color=colors, line=dict(color="#FFFFFF", width=1.5)),
        error_x=dict(type="data", symmetric=False, array=ci[0], arrayminus=ci[1], color=INK_2, thickness=1.4, width=4)
        if ci is not None else None, customdata=cd, showlegend=False,
        hovertemplate=("<b>%{y}</b><br>Late rate: <b>%{x:.1%}</b><br>Late orders: %{customdata[1]:,.0f} of "
                       "%{customdata[0]:,.0f}<br>95% CI: %{customdata[2]:.1%} to %{customdata[3]:.1%}<extra></extra>")),
        row=1, col=2)
    if np.isfinite(overall):
        fig.add_vline(x=overall, line=dict(color=INK, width=1.3, dash="dash"), row=1, col=2)
    dur = _num(d, "mean_duration")
    fig.add_trace(go.Bar(x=dur, y=labels, orientation="h", marker=dict(color=colors), name="Mean delivery time",
                         text=[_nice_float(v) for v in dur], textposition="outside", cliponaxis=False,
                         textfont=dict(size=11, color=INK_2), customdata=n,
                         hovertemplate="<b>%{y}</b><br>Mean delivery time: %{x:.1f} days<br>Orders: %{customdata:,}<extra></extra>",
                         showlegend=False), row=1, col=3)
    xmax_r = float(np.nanmax(np.append((rate + (ci[0] if ci is not None else 0)).to_numpy(), overall if np.isfinite(overall) else 0)))
    fig.update_xaxes(range=[0, float(n.max()) * 1.25], tickformat="~s", showgrid=True, gridcolor=GRID, showline=False, row=1, col=1)
    fig.update_xaxes(range=[0, xmax_r * 1.2], tickformat=".0%", showgrid=True, gridcolor=GRID, showline=False, row=1, col=2)
    fig.update_xaxes(range=[0, float(np.nanmax(dur)) * 1.35], showgrid=True, gridcolor=GRID, showline=False, row=1, col=3)
    for c_, t_ in enumerate(("Orders", "Late rate (%)", "Mean (days)"), start=1):
        fig.update_xaxes(title=t_, nticks=4, tickangle=0, row=1, col=c_)
    fig.update_yaxes(title="State", autorange="reversed", showgrid=False, categoryorder="array", categoryarray=labels, row=1, col=1)
    fig.update_yaxes(showgrid=False, row=1, col=2)
    fig.update_yaxes(showgrid=False, row=1, col=3)
    fig.update_annotations(font=dict(size=12, color=INK_2))
    fig.update_layout(bargap=0.4)
    sub = (f"Dashed line = overall late rate ({_pct(overall)}); grey = fewer than {min_orders:,} orders"
           if small.any() else f"Dashed line = overall late rate ({_pct(overall)}); whiskers = 95% Wilson interval")
    return _finish(fig, title, sub, height=max(300, 56 * len(d) + 170), top=84)


def delay_trend_chart(trend: pd.DataFrame | None, *, window: int = 1, min_orders: int = DEFAULT_MIN_ORDERS,
                      freq_label: str = "month", title: str | None = None) -> go.Figure:
    """Late-rate trend plus late / on-time order counts per period (``analytics.delay_trend``).

    ``window`` > 1 adds a rolling average over that many periods (pooled: sum late / sum orders) and
    draws the raw per-period rate faintly behind it.  Periods under ``min_orders`` are hollow grey.
    """
    t = title or f"Late-delivery trend by purchase {freq_label}"
    if not _frame_ok(trend, ["period", "n_orders"], "n_orders"):
        return empty_figure(title=t)
    d = trend.copy()
    d["_x"] = pd.to_datetime(d["period"], errors="coerce")
    d = d[d["_x"].notna()].sort_values("_x")
    if d.empty:
        return empty_figure(title=t)
    d["_n"] = _num(d, "n_orders").fillna(0)
    d["_late_n"] = _num(d, "n_late") if "n_late" in d.columns else (_num(d, "late_rate") * d["_n"]).round()
    d["_on_n"] = d["_n"] - d["_late_n"]
    d["_r"] = np.where(d["_n"] > 0, d["_late_n"] / d["_n"].where(d["_n"] > 0), np.nan)
    ok = d["_n"] >= min_orders
    window = max(int(window), 1)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4])
    step = _step(d["_x"])
    good = d[ok]
    raw_name = "Late rate per period" if window == 1 else "Late rate per period (raw)"
    if window == 1 and "late_ci_low" in d.columns:
        for k, run in enumerate(_contiguous_runs(good, step)):
            fig.add_trace(go.Scatter(x=pd.concat([run["_x"], run["_x"][::-1]]),
                                     y=pd.concat([_num(run, "late_ci_high"), _num(run, "late_ci_low")[::-1]]),
                                     fill="toself", fillcolor="rgba(220,38,38,0.12)", mode="lines", line=dict(width=0),
                                     hoverinfo="skip", name="95% interval (Wilson)", legendgroup="ci",
                                     showlegend=(k == 0)), row=1, col=1)
    gp = _gapped(good, step)
    fig.add_trace(go.Scatter(
        x=gp["_x"], y=gp["_r"], mode="lines+markers", name=raw_name,
        line=dict(color=PALETTE["red"] if window == 1 else "rgba(220,38,38,0.35)", width=2.5 if window == 1 else 1.5),
        marker=dict(size=7 if window == 1 else 5, color=PALETTE["red"] if window == 1 else "rgba(220,38,38,0.45)",
                    line=dict(color="#FFFFFF", width=1.5)),
        customdata=np.column_stack([gp["_n"], gp["_late_n"]]),
        hovertemplate="Late rate: <b>%{y:.1%}</b><br>Late orders: %{customdata[1]:,.0f} of %{customdata[0]:,.0f}<extra></extra>"),
        row=1, col=1)
    if window > 1:
        # pooled rolling rate over consecutive periods; windows touching a small period are left out
        full = pd.DataFrame({"_x": pd.date_range(d["_x"].min(), d["_x"].max(), freq=_freq_alias(step))})
        full = full.merge(d[["_x", "_n", "_late_n"]], on="_x", how="left")
        valid = (full["_n"] >= min_orders)
        rl = full["_late_n"].where(valid).rolling(window, min_periods=window).sum()
        rn = full["_n"].where(valid).rolling(window, min_periods=window).sum()
        roll = rl / rn.where(rn > 0)
        fig.add_trace(go.Scatter(
            x=full["_x"], y=roll, mode="lines", name=f"{window}-{freq_label} rolling late rate",
            line=dict(color=PALETTE["red"], width=3), customdata=rn,
            hovertemplate=f"Rolling ({window} {freq_label}s): <b>%{{y:.1%}}</b><br>Orders in window: %{{customdata:,.0f}}<extra></extra>"),
            row=1, col=1)
    fig.add_trace(go.Bar(x=d["_x"], y=d["_on_n"], name="On-time orders", marker=dict(color=PALETTE["green"]),
                         hovertemplate="On-time orders: %{y:,}<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Bar(x=d["_x"], y=d["_late_n"], name="Late orders", marker=dict(color=PALETTE["red"]),
                         hovertemplate="Late orders: %{y:,}<extra></extra>"), row=2, col=1)
    fig.update_yaxes(title="Late rate (% of orders)", tickformat=".0%", row=1, col=1,
                     range=_rate_range(good, "_r", True, "late_ci_high" if "late_ci_high" in good.columns else None) or [0, 0.1])
    fig.update_yaxes(title="Orders (count)", tickformat=",", row=2, col=1)
    fig.update_xaxes(title=f"Purchase {freq_label}", row=2, col=1, showline=True, ticks="outside")
    fmt = "%b %Y" if freq_label in ("month", "quarter") else "%d %b %Y"
    fig.update_xaxes(tickformat=fmt, row=2, col=1)
    fig.update_layout(barmode="stack", hovermode="x unified", bargap=0.2)
    if good.empty:
        fig.add_annotation(text=f"No period has {min_orders:,} or more orders, so no late rate is drawn",
                           xref="x domain", yref="y domain", x=0.5, y=0.5, showarrow=False,
                           font=dict(size=13, color=MUTED))
    n_small = int((~ok).sum())
    note = (f"{n_small} periods with fewer than {min_orders:,} orders are left out of the rate lines (too noisy); "
            if n_small else "")
    return _finish(fig, t, note + "stacked bars show order volume",
                   height=560, legend=True, wrap=_NARROW_WRAP, legend_px=_NARROW_LEGEND_PX)


def load_brazil_geojson(path: str | Path | None = None) -> dict | None:
    """Vendored Brazil states GeoJSON (``properties.sigla`` = UF code) or ``None`` if missing / unreadable."""
    return _load_geojson(str(path or GEOJSON_PATH))


@lru_cache(maxsize=4)
def _load_geojson(path: str) -> dict | None:
    try:
        gj = json.loads(Path(path).read_text(encoding="utf-8"))
        if gj.get("type") != "FeatureCollection" or not gj.get("features"):
            return None
        return gj
    except (OSError, ValueError, AttributeError):
        return None


LATE_RATE_COLOR_FLOOR = 0.10   # colour-scale floor (10% late) for the map; a display setting, not a statistic


def brazil_choropleth(state_perf: pd.DataFrame | None, metric: str = "late_rate", *,
                      min_orders: int = DEFAULT_MIN_ORDERS, title: str | None = None) -> go.Figure:
    """Map of Brazil's states coloured by ``metric`` (``late_rate``, ``mean_duration`` or ``n_orders``).

    Needs ``data/geo/brazil_states.geojson``; if it is missing the same data is drawn as a bar chart with
    a note instead.  States below ``min_orders`` are shown grey and flagged in the tooltip.
    """
    names = {"late_rate": "Late rate", "mean_duration": "Mean delivery time", "n_orders": "Orders"}
    metric = metric if metric in names else "late_rate"
    t = title or f"{names[metric]} by customer state"
    if not _frame_ok(state_perf, ["customer_state", "n_orders", metric], metric):
        return empty_figure(title=t, height=480)
    d = state_perf.copy()
    d["_v"], d["_n"] = _num(d, metric), _num(d, "n_orders").fillna(0)
    d = d[d["_v"].notna() & (d["_n"] > 0)]
    d["customer_state"] = d["customer_state"].astype(str)
    if d.empty:
        return empty_figure(title=t, height=480)
    gj = _load_geojson(str(GEOJSON_PATH))
    unit_fmt = {"late_rate": ".1%", "mean_duration": ".1f", "n_orders": ",.0f"}[metric]
    if gj is None:
        if metric == "late_rate":
            fig = state_late_rate_bar(state_perf, min_orders=min_orders, title=t)
        else:
            lab = [f"{c}  ·  n={int(n):,}" for c, n in zip(d["customer_state"], d["_n"])]
            d = d.sort_values("_v", ascending=False)
            lab = [f"{c}  ·  n={int(n):,}" for c, n in zip(d["customer_state"], d["_n"])]
            fig = _plain_hbar(lab, d["_v"], title=t, subtitle=None, xtitle=names[metric], ytitle="State",
                              hover=[f"<b>{_state_label(c)}</b><br>{names[metric]}: {v:{unit_fmt}}<br>Orders: {int(n):,}"
                                     for c, v, n in zip(d["customer_state"], d["_v"], d["_n"])],
                              text=[f"{v:{unit_fmt}}" for v in d["_v"]], color=PALETTE["blue"])
        fig.update_layout(title=_title_dict(t, "Map unavailable (data/geo/brazil_states.geojson not found): showing a bar chart instead"))
        return fig
    all_codes = [f["properties"].get("sigla") for f in gj["features"]]
    small = d["_n"] < min_orders
    big, sm = d[~small], d[small]
    scale, cbt = {"late_rate": (RED_SCALE, "Late rate"), "mean_duration": (BLUE_SCALE, "Mean days"),
                  "n_orders": (BLUE_SCALE, "Orders")}[metric]
    fig = go.Figure()
    fig.add_trace(go.Choropleth(geojson=gj, locations=all_codes, featureidkey="properties.sigla",
                                z=[0] * len(all_codes), colorscale=[[0, MASKED], [1, MASKED]], showscale=False,
                                marker=dict(line=dict(color="#FFFFFF", width=0.8)), hoverinfo="skip",
                                name="No orders"))
    cdfun = lambda x: np.column_stack([x["customer_state"].map(_state_label), x["_n"],
                                       _num(x, "late_rate") if "late_rate" in x.columns else np.full(len(x), np.nan)])
    hover = ("<b>%{customdata[0]}</b> (%{location})<br>" + names[metric] + ": <b>%{z:" + unit_fmt + "}</b><br>"
             "Orders: %{customdata[1]:,.0f}<extra></extra>")
    if not big.empty:
        zmax = float(big["_v"].max()) if big["_v"].max() > 0 else 1.0
        if metric == "late_rate":
            # a floor on the colour scale: a single low-rate state must not be painted the darkest red
            zmax = max(zmax, LATE_RATE_COLOR_FLOOR)
        fig.add_trace(go.Choropleth(
            geojson=gj, locations=big["customer_state"], featureidkey="properties.sigla", z=big["_v"],
            colorscale=scale, zmin=0, zmax=zmax, customdata=cdfun(big), hovertemplate=hover,
            marker=dict(line=dict(color="#FFFFFF", width=0.8)),
            colorbar=dict(title=dict(text=cbt, side="top"), thickness=12, len=0.5, orientation="h", x=0.5,
                          xanchor="center", y=-0.01, yanchor="top",
                          tickformat={"late_rate": ".0%", "mean_duration": ".0f", "n_orders": ","}[metric]),
            name="States"))
    if not sm.empty:
        fig.add_trace(go.Choropleth(
            geojson=gj, locations=sm["customer_state"], featureidkey="properties.sigla", z=sm["_v"],
            colorscale=[[0, NEUTRAL], [1, NEUTRAL]], showscale=False, customdata=cdfun(sm),
            hovertemplate=("<b>%{customdata[0]}</b> (%{location})<br>" + names[metric] + ": %{z:" + unit_fmt +
                           "}<br>Orders: <b>%{customdata[1]:,.0f}</b> (small sample, interpret with care)<extra></extra>"),
            marker=dict(line=dict(color="#FFFFFF", width=0.8)), name="Small sample"))
    fig.update_geos(fitbounds="locations", visible=False, bgcolor="rgba(0,0,0,0)", projection_type="mercator")
    sub = "Pale grey = no orders under the current filters"
    if not sm.empty:
        sub = f"Mid-grey = fewer than {min_orders:,} orders ({len(sm)} states); pale grey = no orders"
    fig.update_layout(margin=dict(l=0, r=0, t=78, b=0))
    return _finish(fig, t, sub, height=580, left=0, right=0, bottom=56)


# --------------------------------------------------------------------------- #
# Model charts (read the structure of models/model_metrics.json)
# --------------------------------------------------------------------------- #
_MODEL_SHORT = {"dummy_prior": "Dummy baseline", "logistic_regression": "Logistic regression",
                "random_forest": "Random forest", "hist_gradient_boosting": "Hist. gradient boosting"}


def _models(metrics: Any) -> dict:
    m = metrics.get("models") if isinstance(metrics, Mapping) else None
    return m if isinstance(m, Mapping) else {}


def _selected_name(metrics: Any) -> str | None:
    sel = metrics.get("selected_model") if isinstance(metrics, Mapping) else None
    return sel.get("name") if isinstance(sel, Mapping) else None


def _model_entry(metrics: Any, model: str | None) -> tuple[str | None, Mapping | None]:
    models = _models(metrics)
    name = model or _selected_name(metrics) or (next(iter(models), None))
    entry = models.get(name) if name else None
    return (name, entry) if isinstance(entry, Mapping) else (name, None)


def _model_label(name: str | None, entry: Mapping | None = None) -> str:
    if name in _MODEL_SHORT:
        return _MODEL_SHORT[name]
    if isinstance(entry, Mapping) and entry.get("label"):
        return str(entry["label"])
    return pretty_category(name or "model")


def _window_label(w: str) -> str:
    return {"validation": "Validation (tuned, in-sample)", "test": "Test (untouched later window)"}.get(w, w.title())


_WINDOW_STYLE = {"test": dict(color=PALETTE["blue"], dash="solid"),
                 "validation": dict(color=VALIDATION_BLUE, dash="dash")}


def _prevalence(entry: Mapping, window: str) -> float | None:
    v = entry.get(window, {}).get("positive_rate") if isinstance(entry.get(window), Mapping) else None
    return float(v) if isinstance(v, (int, float)) and np.isfinite(v) else None


def model_comparison_chart(metrics: Mapping | None, *, windows: Sequence[str] = ("validation", "test"),
                           metrics_shown: Sequence[str] = ("pr_auc", "roc_auc", "recall"),
                           title: str = "Model comparison: validation vs untouched test window") -> go.Figure:
    """Grouped bars (one panel per metric) of each candidate model on the validation and test windows.

    Test bars carry the bootstrap 95% interval when ``test_ci95`` exists.  Dashed reference = no-skill level
    (ROC-AUC 0.5; PR-AUC = late-order share of the test window).  The selected model is starred.
    """
    models = _models(metrics)
    labels_m = {"pr_auc": "PR-AUC (higher = better)", "roc_auc": "ROC-AUC (0.5 = no skill)",
                "recall": "Recall at chosen threshold"}
    shown = [m for m in metrics_shown if m in labels_m and any(
        isinstance(e.get(w), Mapping) and np.isfinite(_f(e[w].get(m))) for e in models.values() for w in windows)]
    if not models or not shown:
        return empty_figure(title=title, hint="Model metrics are missing. Train the model: python -m src.train_model")
    sel = _selected_name(metrics)
    names = list(models)
    ticks = [_wrap(_model_label(n, models[n]), 14) + ("<br>★ selected" if n == sel else "") for n in names]
    fig = make_subplots(rows=1, cols=len(shown), horizontal_spacing=0.07,
                        subplot_titles=[labels_m[m] for m in shown])
    colors = {"validation": VALIDATION_BLUE, "test": PALETTE["blue"]}
    ref_names: list[str] = []
    for ci_, m in enumerate(shown, start=1):
        ymax = 0.0
        for w in windows:
            ys = [_f(models[n].get(w, {}).get(m)) if isinstance(models[n].get(w), Mapping) else np.nan for n in names]
            err = None
            if w == "test" and m in ("roc_auc", "pr_auc"):
                plus, minus = [], []
                for n, y in zip(names, ys):
                    ci = (models[n].get("test_ci95") or {}).get(m) if isinstance(models[n].get("test_ci95"), Mapping) else None
                    if ci and len(ci) == 2 and np.isfinite(y):
                        plus.append(max(float(ci[1]) - y, 0)); minus.append(max(y - float(ci[0]), 0))
                    else:
                        plus.append(0); minus.append(0)
                err = dict(type="data", symmetric=False, array=plus, arrayminus=minus, color=INK_2, thickness=1.3, width=3)
            ymax = max(ymax, float(np.nanmax(ys)) if np.isfinite(ys).any() else 0)
            fig.add_trace(go.Bar(
                x=ticks, y=ys, name=_window_label(w), marker=dict(color=colors.get(w, NEUTRAL)),
                text=[f"{y:.2f}" if np.isfinite(y) else "" for y in ys], textposition="inside", insidetextanchor="start",
                textfont=dict(size=10, color=INK if w == "validation" else "#FFFFFF"), constraintext="none", error_y=err, legendgroup=w, showlegend=(ci_ == 1),
                customdata=[_model_label(n, models[n]) for n in names],
                hovertemplate="<b>%{customdata}</b><br>" + _window_label(w) + "<br>" + labels_m[m].split(" (")[0] +
                              ": <b>%{y:.3f}</b><extra></extra>"), row=1, col=ci_)
        ref = None
        if m == "roc_auc":
            ref = (0.5, "ROC-AUC 0.5")
        elif m == "pr_auc":
            prev = [p for p in (_prevalence(e, "test") for e in models.values()) if p is not None]
            if prev:
                ref = (float(np.mean(prev)), f"PR-AUC {_pct(float(np.mean(prev)))}")
        if ref:
            ymax = max(ymax, ref[0])
            ref_names.append(ref[1])
            fig.add_hline(y=ref[0], line=dict(color=INK, width=1.3, dash="dash"), row=1, col=ci_)
        fig.update_yaxes(range=[0, ymax * 1.28], tickformat=".1f" if m != "pr_auc" else ".2f", row=1, col=ci_,
                         title="Score (0–1)" if ci_ == 1 else None)
        fig.update_xaxes(tickfont=dict(size=10), row=1, col=ci_)
    fig.update_annotations(font=dict(size=12, color=INK_2))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.06)
    if ref_names:
        _legend_proxy(fig, "No skill: " + ", ".join(ref_names), INK, dash="dash")
    sub = ("Validation scores are tuned / in-sample for the selected model; the test window is the honest check. "
           "Whiskers = bootstrap 95% interval")
    _finish(fig, title, sub, height=500, legend=True, top=110 + 22 * _legend_rows(fig))
    fig.update_layout(legend=dict(y=1.0 + (0.12 if _legend_rows(fig) == 1 else 0.10 + 0.04 * _legend_rows(fig))))
    return fig


def _f(x: Any) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def confusion_matrix_chart(metrics: Mapping | None, model: str | None = None, window: str = "test", *,
                           title: str | None = None) -> go.Figure:
    """2x2 confusion matrix: counts and row-normalised %, labelled On-time / Late, at the model's threshold."""
    name, entry = _model_entry(metrics, model)
    t = title or f"Confusion matrix ({window} window)"
    cm = entry.get(f"confusion_matrix_{window}") if entry else None
    try:
        arr = np.asarray(cm, dtype="float64")
        if arr.shape != (2, 2) or not np.isfinite(arr).all():
            raise ValueError
    except (TypeError, ValueError):
        return empty_figure(title=t, hint="No confusion matrix stored for this model/window.")
    rows = arr.sum(axis=1, keepdims=True)
    pct = np.divide(arr, rows, out=np.zeros_like(arr), where=rows > 0)
    cls = ["On-time", "Late"]
    text = [[f"<b>{int(arr[i, j]):,}</b><br>{pct[i, j]:.1%} of actual {cls[i].lower()}" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=pct, x=[f"Predicted {c.lower()}" for c in cls], y=[f"Actual {c.lower()}" for c in cls],
        colorscale=BLUE_SCALE, zmin=0, zmax=1, showscale=False, xgap=3, ygap=3, customdata=arr,
        hovertemplate="%{y} → %{x}<br>Orders: <b>%{customdata:,.0f}</b><br>Share of row: %{z:.1%}<extra></extra>"))
    for i in range(2):
        for j in range(2):
            fig.add_annotation(x=j, y=i, xref="x", yref="y", text=text[i][j], showarrow=False,
                               font=dict(size=14, color="#FFFFFF" if pct[i, j] >= 0.45 else INK))
    fig.update_xaxes(title="Model prediction", side="bottom", showgrid=False, type="category")
    fig.update_yaxes(title="What actually happened", autorange="reversed", showgrid=False, type="category")
    thr = entry.get("threshold") if entry else None
    tp = arr[1, 1]; fn = arr[1, 0]; fp = arr[0, 1]
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    sub = (f"{_model_label(name, entry)} · flags an order as late when probability ≥ {thr:.3f} · "
           f"recall {_pct(rec)} · precision {_pct(prec)} · n = {int(arr.sum()):,}") if isinstance(thr, (int, float)) else \
        f"{_model_label(name, entry)} · recall {_pct(rec)} · precision {_pct(prec)} · n = {int(arr.sum()):,}"
    return _finish(fig, t, sub, height=420, left=16)


def _curve_traces(metrics: Any, model: str | None, windows: Sequence[str], kind: str):
    name, entry = _model_entry(metrics, model)
    out = []
    if not entry:
        return name, entry, out
    for w in windows:
        c = entry.get(f"{kind}_curve_{w}")
        if not isinstance(c, Mapping):
            continue
        keys = ("fpr", "tpr") if kind == "roc" else ("recall", "precision")
        try:
            x, y = np.asarray(c[keys[0]], dtype=float), np.asarray(c[keys[1]], dtype=float)
        except (KeyError, TypeError, ValueError):
            continue
        if x.size and x.size == y.size:
            out.append((w, x, y))
    return name, entry, out


def roc_curve_chart(metrics: Mapping | None, model: str | None = None, *,
                    windows: Sequence[str] = ("test", "validation"), title: str = "ROC curve") -> go.Figure:
    """ROC curve of the selected (or given) model for the test and validation windows vs the no-skill diagonal."""
    name, entry, curves = _curve_traces(metrics, model, windows, "roc")
    if not curves:
        return empty_figure(title=title, hint="No ROC curve stored. Train the model: python -m src.train_model")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="No skill (random ranking)",
                             line=dict(color=NEUTRAL, width=1.5, dash="dot"), hoverinfo="skip"))
    for w, x, y in curves:
        auc = _f(entry.get(w, {}).get("roc_auc")) if isinstance(entry.get(w), Mapping) else float("nan")
        st = _WINDOW_STYLE.get(w, dict(color=NEUTRAL, dash="solid"))
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=st["color"], width=2.5, dash=st["dash"],
                                                                shape="hv"),
                                 name=f"{_window_label(w)}, AUC {auc:.3f}" if np.isfinite(auc) else _window_label(w),
                                 hovertemplate="False positive rate: %{x:.1%}<br>True positive rate: <b>%{y:.1%}</b><extra></extra>"))
    fig.update_xaxes(title="False positive rate (share of on-time orders flagged late)", range=[0, 1], tickformat=".0%",
                     showgrid=True, gridcolor=GRID)
    fig.update_xaxes(constrain="domain")
    fig.update_yaxes(title="True positive rate (share of late orders caught)", range=[0, 1.0], tickformat=".0%",
                     scaleanchor="x", scaleratio=1, constrain="domain")
    return _finish(fig, title, f"{_model_label(name, entry)}; the closer to the top-left corner the better, the dotted diagonal is a coin flip",
                   height=520, legend=True, top=None)


def pr_curve_chart(metrics: Mapping | None, model: str | None = None, *,
                   windows: Sequence[str] = ("test", "validation"), title: str = "Precision-recall curve") -> go.Figure:
    """Precision-recall curve with the no-skill line (= late share of each window)."""
    name, entry, curves = _curve_traces(metrics, model, windows, "pr")
    if not curves:
        return empty_figure(title=title, hint="No precision-recall curve stored. Train the model: python -m src.train_model")
    fig = go.Figure()
    ymax = 0.0
    for w, x, y in curves:
        w_entry = entry.get(w, {}) if isinstance(entry.get(w), Mapping) else {}
        ap = _f(w_entry.get("pr_auc"))
        st = _WINDOW_STYLE.get(w, dict(color=NEUTRAL, dash="solid"))
        ymax = max(ymax, float(np.nanmax(y)))
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=st["color"], width=2.5, dash=st["dash"]),
                                 name=f"{_window_label(w)}, PR-AUC {ap:.3f}" if np.isfinite(ap) else _window_label(w),
                                 hovertemplate="Recall: %{x:.1%}<br>Precision: <b>%{y:.1%}</b><extra></extra>"))
    for w, _, _ in curves:
        prev = _prevalence(entry, w)
        if prev is not None:
            st = _WINDOW_STYLE.get(w, dict(color=NEUTRAL))
            ymax = max(ymax, prev)
            fig.add_trace(go.Scatter(x=[0, 1], y=[prev, prev], mode="lines", line=dict(color=INK if w == "test" else NEUTRAL, width=1.5, dash="dot"),
                                     name=f"No skill, {w} window ({_pct(prev)} late)", hovertemplate=f"No-skill precision = late share = {prev:.1%}<extra></extra>"))
    fig.update_xaxes(title="Recall (share of late orders caught)", range=[0, 1], tickformat=".0%", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title="Precision (share of flagged orders that are late)", range=[0, min(1.0, ymax * 1.15)], tickformat=".0%")
    return _finish(fig, title, f"{_model_label(name, entry)}; the dotted line is what a model with no skill would achieve",
                   height=480, legend=True, top=None, right=44)


def calibration_chart(metrics: Mapping | None, model: str | None = None, *,
                      windows: Sequence[str] = ("test", "validation"),
                      title: str = "Calibration: predicted vs observed late rate") -> go.Figure:
    """Reliability diagram (binned predicted probability vs observed late share, bin size in the tooltip)."""
    name, entry = _model_entry(metrics, model)
    series = []
    for w in windows:
        c = entry.get(f"calibration_{w}") if entry else None
        if isinstance(c, Mapping):
            try:
                x, y = np.asarray(c["mean_predicted"], float), np.asarray(c["fraction_positive"], float)
                n = np.asarray(c.get("bin_counts", np.full(x.shape, np.nan)), float)
            except (KeyError, TypeError, ValueError):
                continue
            if x.size and x.size == y.size and n.size == x.size:
                series.append((w, x, y, n))
    if not series:
        return empty_figure(title=title, hint="No calibration data stored. Train the model: python -m src.train_model")
    top = max(float(np.nanmax(np.concatenate([x, y]))) for _, x, y, _ in series) * 1.15
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="Perfect calibration",
                             line=dict(color=NEUTRAL, width=1.5, dash="dot"), hoverinfo="skip"))
    for w, x, y, n in series:
        st = _WINDOW_STYLE.get(w, dict(color=NEUTRAL, dash="solid"))
        size = 8 + 18 * np.sqrt(np.where(np.isfinite(n), n, 0) / max(float(np.nanmax(n)), 1.0))
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers", name=_window_label(w), customdata=n,
            line=dict(color=st["color"], width=2, dash=st["dash"]),
            marker=dict(size=size, color=st["color"], line=dict(color="#FFFFFF", width=1.5), opacity=0.95),
            hovertemplate="Mean predicted: %{x:.1%}<br>Observed late rate: <b>%{y:.1%}</b><br>Orders in bin: %{customdata:,.0f}<extra></extra>"))
    fig.update_xaxes(title="Mean predicted probability of late delivery (per bin)", range=[0, top], tickformat=".0%",
                     showgrid=True, gridcolor=GRID)
    fig.update_xaxes(constrain="domain")
    fig.update_yaxes(title="Observed late rate (% of orders in bin)", range=[0, top], tickformat=".0%", scaleanchor="x",
                     scaleratio=1, constrain="domain")
    return _finish(fig, title, f"{_model_label(name, entry)}; points on the dotted line would mean predicted = observed; marker size = orders in bin",
                   height=480, legend=True, top=None)


def feature_importance_chart(metrics: Mapping | None, *, top_n: int = 12, window: str = "validation",
                             title: str = "What the model relies on most") -> go.Figure:
    """Horizontal bars of permutation importance (mean ± std over repeats), human-readable names."""
    pi = metrics.get("permutation_importance") if isinstance(metrics, Mapping) else None
    items = None
    if isinstance(pi, Mapping):
        items = pi.get("test_items") if window == "test" else pi.get("items")
    if not isinstance(items, list) or not items:
        return empty_figure(title=title, hint="No importance values stored. Train the model: python -m src.train_model")
    d = pd.DataFrame(items)
    if "feature" not in d.columns or "importance_mean" not in d.columns:
        return empty_figure(title=title)
    d["_m"] = _num(d, "importance_mean")
    d["_s"] = _num(d, "importance_std").fillna(0)
    d = d[d["_m"].notna()].sort_values("_m", ascending=False).head(top_n)
    if d.empty:
        return empty_figure(title=title)
    labels = [pretty_feature_name(f) for f in d["feature"]]
    colors = [PALETTE["blue"] if m > 0 else NEUTRAL for m in d["_m"]]
    fig = go.Figure(go.Bar(
        x=d["_m"], y=labels, orientation="h", marker=dict(color=colors),
        error_x=dict(type="data", array=d["_s"], color=INK_2, thickness=1.3, width=3),
        customdata=np.column_stack([d["_s"], d["feature"]]),
        hovertemplate="<b>%{y}</b><br>Drop in PR-AUC when shuffled: <b>%{x:.4f}</b> (±%{customdata[0]:.4f})<extra></extra>"))
    xmax = float((d["_m"] + d["_s"]).max())
    fig.add_trace(go.Scatter(x=d["_m"] + d["_s"], y=labels, mode="text", text=["  " + f"{abs(m) if round(m, 3) == 0 else m:.3f}" for m in d["_m"]],
                             textposition="middle right", textfont=dict(size=11, color=INK_2), hoverinfo="skip",
                             showlegend=False, cliponaxis=False))
    fig.update_xaxes(title="Drop in PR-AUC", range=[min(0, float(d["_m"].min()) * 1.2), xmax * 1.25],
                     showgrid=True, gridcolor=GRID, showline=False, ticks="")
    fig.update_yaxes(title="Feature", autorange="reversed", categoryorder="array", categoryarray=labels, showgrid=False)
    fig.update_layout(bargap=0.35)
    n_rep = pi.get("n_repeats") if isinstance(pi, Mapping) else None
    sub = f"{window.capitalize()} window. Permutation importance (drop in PR-AUC when one input is shuffled): bigger = the model's accuracy depends more on that input. Association in the model, not a cause of lateness" + \
          (f" · whiskers ±1 sd over {n_rep} shuffles" if n_rep else "")
    return _finish(fig, title, sub, height=max(340, 30 * len(d) + 170), wrap=_NARROW_WRAP)


def risk_band_chart(metrics: Mapping | None, *, title: str = "Observed late rate by predicted risk band") -> go.Figure:
    """Observed late rate in the Low / Moderate / High bands, validation vs test window (n and 95% CI shown)."""
    rb = metrics.get("risk_bands") if isinstance(metrics, Mapping) else None
    wins = [w for w in ("validation", "test") if isinstance(rb, Mapping) and isinstance(rb.get(w), list) and rb.get(w)]
    if not wins:
        return empty_figure(title=title, hint="No risk-band results stored. Train the model: python -m src.train_model")
    split = metrics.get("split", {}) if isinstance(metrics, Mapping) else {}
    short = {"validation": "Validation", "test": "Test"}

    def facet_title(w: str) -> str:
        lr = split[w].get("late_rate") if isinstance(split, Mapping) and isinstance(split.get(w), Mapping) else None
        return short.get(w, w.title()) + (f" ({_pct(float(lr))} late)" if isinstance(lr, (int, float)) and np.isfinite(lr) else "")

    fig = make_subplots(rows=1, cols=len(wins), shared_yaxes=True, horizontal_spacing=0.05,
                        subplot_titles=[facet_title(w) for w in wins])
    ymax = 0.0
    for ci_, w in enumerate(wins, start=1):
        d = pd.DataFrame(rb[w])
        if not {"band", "n", "late_rate"} <= set(d.columns):
            continue
        d["_r"] = _num(d, "late_rate")
        ci = _ci_arrays(d)
        cols = [RISK_COLORS.get(b, NEUTRAL) for b in d["band"]]
        fig.add_trace(go.Bar(
            x=[f"{b}<br><span style='font-size:10px'>n={int(n):,}</span>" for b, n in zip(d["band"], _num(d, "n"))],
            y=d["_r"], marker=dict(color=cols), showlegend=False,
            error_y=dict(type="data", symmetric=False, array=ci[0], arrayminus=ci[1], color=INK_2, thickness=1.3, width=4)
            if ci is not None else None,
            customdata=np.column_stack([d["band"], _num(d, "n"), _num(d, "n_late") if "n_late" in d.columns else np.full(len(d), np.nan),
                                        _num(d, "late_ci_low"), _num(d, "late_ci_high")]),
            hovertemplate=("<b>%{customdata[0]} risk</b><br>Observed late rate: <b>%{y:.1%}</b><br>"
                           "Late orders: %{customdata[2]:,.0f} of %{customdata[1]:,.0f}<br>"
                           "95% CI: %{customdata[3]:.1%} to %{customdata[4]:.1%}<extra></extra>")), row=1, col=ci_)
        tops = (d["_r"] + (ci[0] if ci is not None else 0)).to_numpy()
        base_w = (split.get(w, {}) or {}).get("late_rate") if isinstance(split, Mapping) else None
        label_y = tops.copy()
        if isinstance(base_w, (int, float)) and np.isfinite(base_w):
            span = max(float(tops.max()), float(base_w)) * 1.3     # label sits above its bar; lift it clear of the dashed baseline
            hit = (base_w >= label_y - 0.01 * span) & (base_w <= label_y + 0.11 * span)
            label_y = np.where(hit, float(base_w) + 0.005 * span, label_y)
        fig.add_trace(go.Scatter(x=[f"{b}<br><span style='font-size:10px'>n={int(n):,}</span>" for b, n in zip(d["band"], _num(d, "n"))],
                                 y=label_y, mode="text", text=[_pct(r) for r in d["_r"]], textposition="top center",
                                 textfont=dict(size=12, color=INK_2), hoverinfo="skip", showlegend=False, cliponaxis=False),
                      row=1, col=ci_)
        ymax = max(ymax, float(tops.max()))
        base = (split.get(w, {}) or {}).get("late_rate") if isinstance(split, Mapping) else None
        if isinstance(base, (int, float)) and np.isfinite(base):
            ymax = max(ymax, float(base))
            fig.add_hline(y=float(base), line=dict(color=INK, width=1.3, dash="dash"), row=1, col=ci_)
    fig.update_yaxes(range=[0, ymax * 1.3 if ymax > 0 else 1], tickformat=".0%", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title="Observed late rate (% of orders)", col=1)
    fig.update_xaxes(title="Predicted risk band", showline=True)
    fig.update_annotations(font=dict(size=12, color=INK_2))
    fig.update_layout(bargap=0.35)
    test_diag = (rb.get("test_diagnostics") or {}) if isinstance(rb, Mapping) else {}
    overlap = (test_diag.get("adjacent_intervals_overlap") or {}) if isinstance(test_diag, Mapping) else {}
    sub = "Whiskers = 95% Wilson interval; dashed line = late rate of the whole window"
    if overlap.get("Moderate-High"):
        sub += " · on the test window the Moderate and High bands are barely separated (intervals overlap)"
    n_sub = len(_wrap(sub, _NARROW_WRAP).split("<br>"))
    return _finish(fig, title, sub, height=440, top=78 + 15 * (n_sub - 1) + 28, wrap=_NARROW_WRAP)


def probability_gauge(probability: float | None, thresholds: Mapping[str, float] | None = None, *,
                      title: str = "Predicted probability of a late delivery", height: int = 190) -> go.Figure:
    """Compact horizontal risk meter: green / amber / red zones split at the model's thresholds."""
    try:
        p = float(probability)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return empty_figure("No prediction yet", title=title, height=height)
    if not np.isfinite(p) or p < 0:
        return empty_figure("No prediction yet", title=title, height=height)
    p = min(p, 1.0)
    thr = thresholds or {}
    mod, high = _f(thr.get("moderate")), _f(thr.get("high"))
    have_thr = np.isfinite(mod) and np.isfinite(high) and 0 < mod < high
    xmax = min(1.0, max(2.0 * high if have_thr else 0.0, p * 1.15, 0.05))
    xmax = math.ceil(xmax * 20) / 20
    fig = go.Figure()
    if have_thr:
        segs = [("Low risk", 0.0, mod, PALETTE["green"], "#FFFFFF", f"below {mod:.1%}"),
                ("Moderate risk", mod, high, PALETTE["amber"], INK, f"{mod:.1%} to {high:.1%}"),
                ("High risk", high, xmax, PALETTE["red"], "#FFFFFF", f"{high:.1%} and above")]
    else:
        segs = [("Probability scale", 0.0, xmax, NEUTRAL_LIGHT, INK, "no thresholds supplied")]
    for name, a, b, c, tc, rng in segs:
        fig.add_trace(go.Bar(x=[b - a], y=[0.35], base=[a], orientation="h", width=0.5, marker=dict(color=c),
                             name=name, hovertemplate=f"<b>{name}</b><br>Probability {rng}<extra></extra>", showlegend=False))
        if have_thr:   # band names sit just BELOW the bar, centred, so the marker needle never covers them
            fig.add_annotation(x=(a + b) / 2, y=-0.1, xref="x", yref="y", yanchor="middle", showarrow=False,
                               text=name.split(" ")[0], font=dict(size=12, color=INK_2))
    label = ("Low" if p < mod else ("Moderate" if p < high else "High")) if have_thr else None
    # marker: a dark needle through the bar plus a triangle above it
    fig.add_shape(type="line", x0=p, x1=p, y0=0.08, y1=0.7, xref="x", yref="y", line=dict(color=INK, width=3))
    fig.add_trace(go.Scatter(x=[p], y=[0.82], mode="markers", marker=dict(symbol="triangle-down", size=16, color=INK),
                             hovertemplate=f"Predicted probability: <b>{p:.1%}</b><extra></extra>", showlegend=False))
    fig.add_annotation(x=p, y=1.05, xref="x", yref="y", yanchor="bottom",
                       xanchor="left" if p / xmax < 0.14 else ("right" if p / xmax > 0.86 else "center"), showarrow=False,
                       text=f"<b>{p:.1%}</b>" + (f"  ·  {label} risk" if label else ""), font=dict(size=14, color=INK))
    ticks = [0.0] + ([mod, high] if have_thr else []) + [xmax]
    fig.update_xaxes(range=[0, xmax], tickvals=ticks, ticktext=[f"{t:.1%}" if 0 < t < xmax else f"{t:.0%}" for t in ticks],
                     title=None, showgrid=False, showline=False, ticks="outside", ticklen=4)
    fig.update_yaxes(visible=False, showgrid=False, range=[-0.28, 1.45])
    fig.update_layout(barmode="overlay", bargap=0)
    _finish(fig, title, None, height=height, top=40, bottom=30, left=12, right=40)
    fig.update_layout(title=dict(font=dict(size=14)))
    return fig


def scenario_comparison_chart(scenarios: Sequence[Mapping] | None, thresholds: Mapping[str, float] | None = None, *,
                              title: str = "Pinned scenarios: estimated probability of a late delivery") -> go.Figure:
    """Bars comparing the model's late probability for up to a few pinned what-if orders.

    Each scenario is a mapping with ``label``, ``probability`` (0-1), ``risk_label`` and optionally
    ``description``.  Dashed lines mark the Moderate / High thresholds.  Model estimates, not outcomes.
    """
    rows = []
    for s in scenarios or []:
        if not isinstance(s, Mapping):
            continue
        p = _f(s.get("probability"))
        if np.isfinite(p) and 0 <= p <= 1:
            rows.append((str(s.get("label", f"Scenario {len(rows) + 1}")), p, str(s.get("risk_label", "")), str(s.get("description", ""))))
    if not rows:
        return empty_figure("No scenarios pinned yet", title=title, hint="Predict an order, then pin it to compare.")
    labels = [r[0] for r in rows]
    probs = [r[1] for r in rows]
    fig = go.Figure(go.Bar(
        x=labels, y=probs, marker=dict(color=[RISK_COLORS.get(r[2], NEUTRAL) for r in rows]), width=0.45,
        customdata=np.column_stack([[r[2] or "n/a" for r in rows], [r[3] for r in rows]]),
        hovertemplate="<b>%{x}</b><br>Estimated probability: <b>%{y:.1%}</b><br>Risk band: %{customdata[0]}<br>%{customdata[1]}<extra></extra>"))
    fig.add_trace(go.Scatter(x=labels, y=probs, mode="text", text=[f"{p:.1%}" for p in probs], textposition="top center",
                             textfont=dict(size=12, color=INK_2), hoverinfo="skip", showlegend=False, cliponaxis=False))
    thr = thresholds or {}
    ymax = max(probs)
    for key, name in (("moderate", "Moderate from"), ("high", "High from")):
        t = _f(thr.get(key))
        if np.isfinite(t) and t > 0:
            ymax = max(ymax, t)
            fig.add_hline(y=t, line=dict(color=NEUTRAL, width=1.2, dash="dash"),
                          annotation=dict(text=f"{name} {t:.1%}", font=dict(size=11, color=MUTED), xanchor="right", x=1.0))
    fig.update_yaxes(title="Estimated probability of a late delivery", tickformat=".0%", range=[0, ymax * 1.3 if ymax > 0 else 1],
                     showgrid=True, gridcolor=GRID)
    fig.update_xaxes(title="Pinned scenario", showline=True)
    return _finish(fig, title, "Model estimates, not outcomes; bar colour = risk band", height=380)
