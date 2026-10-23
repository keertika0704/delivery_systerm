"""Export byte-builders and report helpers for the Reports & Insights page (pure Python, no Streamlit).

* **Downloads** -- the shipment CSV (a column *whitelist*: ``order_id`` plus analytical columns,
  never customer / seller identifiers), the regional CSV, the Markdown summary and its HTML
  rendering.  CSVs are written as UTF-8 *with* a byte-order mark so spreadsheet programs show
  accents (``São Paulo``) correctly; read them back with ``encoding="utf-8-sig"``.
* **Explorer** -- :func:`explorer_view` sorts and paginates a frame (never more than one page of
  rows is handed to the UI).
* **Presentation helpers** -- everything that turns :class:`src.insights.Insight` objects and
  ``models/model_metrics.json`` into labelled, number-driven text and tables.  All wording is built
  from the values passed in; nothing is hard-coded.
"""
from __future__ import annotations

import html as _html
import io
import math
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import analytics as an
from .config import STATE_NAMES

# --------------------------------------------------------------------------- #
# Column whitelist (what may leave the app)
# --------------------------------------------------------------------------- #
# kind: text | int | float | money | days | date | datetime | flag | status
COLUMN_INFO: dict[str, tuple[str, str, str]] = {
    "order_id": ("Order ID", "Order key (the only identifier exported; customers and sellers are not identified).", "text"),
    "order_purchase_timestamp": ("Purchased at", "Date and time the customer placed the order.", "datetime"),
    "purchase_year_month": ("Purchase month", "Purchase month as YYYY-MM.", "text"),
    "order_estimated_delivery_date": ("Estimated delivery", "Delivery date promised at checkout.", "datetime"),
    "order_delivered_customer_date": ("Delivered at", "Date and time the customer received the order.", "datetime"),
    "is_late": ("Late (1/0)", "1 = delivered on a calendar date after the estimate; 0 = on time or early.", "flag"),
    "delivery_status": ("Delivery status", "Readable version of is_late: late / on_time.", "status"),
    "delivery_delay_days": ("Delay (days)", "Delivered date minus estimated date in whole days (negative = early).", "int"),
    "delivery_duration_days": ("Delivery time (days)", "Purchase to customer delivery, in days.", "days"),
    "estimated_delivery_days": ("Promised time (days)", "Delivery promise shown at checkout, in days.", "int"),
    "processing_days": ("Processing (days)", "Purchase to carrier hand-over, in days (known only after the event).", "days"),
    "customer_state": ("Customer state", "Brazilian state (UF) of the customer.", "text"),
    "customer_city": ("Customer city", "City of the customer (city level only).", "text"),
    "seller_state": ("Seller state", "State of the representative seller (largest item value).", "text"),
    "seller_city": ("Seller city", "City of the representative seller.", "text"),
    "state_pair": ("Route", "Seller state to customer state, e.g. SP→RJ.", "text"),
    "same_state": ("Same state (1/0)", "1 = seller and customer are in the same state.", "flag"),
    "product_category": ("Product category", "Category of the highest-priced item (English).", "text"),
    "n_items": ("Items", "Number of items in the order.", "int"),
    "n_sellers": ("Sellers", "Number of distinct sellers in the order.", "int"),
    "total_price": ("Order value (R$)", "Sum of item prices.", "money"),
    "total_freight": ("Freight (R$)", "Sum of item freight charges.", "money"),
    "freight_ratio": ("Freight / price", "Freight divided by item value.", "float"),
    "total_weight_g": ("Weight (g)", "Total product weight; empty if any item lacks a weight.", "float"),
    "total_volume_cm3": ("Volume (cm³)", "Total product volume; empty if any item lacks dimensions.", "float"),
    "review_score": ("Review score", "1-5 customer rating where a review exists (descriptive only).", "float"),
}
EXPORT_COLUMNS: tuple[str, ...] = tuple(COLUMN_INFO)
DERIVED_COLUMNS = ("delivery_status",)
DEFAULT_EXPLORER_COLUMNS = (
    "order_id", "order_purchase_timestamp", "customer_state", "product_category", "seller_state",
    "delivery_status", "delivery_delay_days", "delivery_duration_days", "estimated_delivery_days",
    "total_price", "total_freight", "n_items",
)
EXCLUDED_NOTE = (
    "Never exported: customer and seller identifiers (none are stored), order status flags (constant for "
    "delivered orders), carrier / approval timestamps, data-quality flags and leakage-prone model columns."
)
CSV_ENCODING = "utf-8-sig"
_STATUS_TEXT = {1: "late", 0: "on_time"}


def column_label(col: str) -> str:
    """Readable column name (falls back to the raw name)."""
    return COLUMN_INFO.get(col, (col,))[0]


def data_dictionary() -> pd.DataFrame:
    """Documentation of the whitelist: one row per exportable column."""
    return pd.DataFrame([{"column": c, "label": v[0], "description": v[1]} for c, v in COLUMN_INFO.items()])


# --------------------------------------------------------------------------- #
# Shipment CSV
# --------------------------------------------------------------------------- #
def export_frame(df: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Subset ``df`` to whitelisted columns (in whitelist order unless ``columns`` orders them).

    ``delivery_status`` is derived from ``is_late``.  Asking for a column outside the whitelist
    raises ``ValueError`` so an identifier can never be exported by accident; whitelisted columns
    missing from ``df`` are skipped.
    """
    want = list(EXPORT_COLUMNS if columns is None else columns)
    bad = [c for c in want if c not in COLUMN_INFO]
    if bad:
        raise ValueError(f"Columns not allowed in exports: {bad}. Allowed: {list(EXPORT_COLUMNS)}")
    out = df[[c for c in want if c in df.columns]].copy()
    if "delivery_status" in want and "is_late" in df.columns:
        late = pd.to_numeric(df["is_late"], errors="coerce")
        status = pd.Series(pd.NA, index=df.index, dtype="string")
        status[late == 1] = "late"
        status[late == 0] = "on_time"
        out["delivery_status"] = status
        out = out[[c for c in want if c in out.columns]]
    return out


def csv_bytes(df: pd.DataFrame) -> bytes:
    """Any frame as UTF-8 (with BOM) CSV bytes: header always written, no index, ISO timestamps."""
    text = df.to_csv(index=False, date_format="%Y-%m-%d %H:%M:%S", lineterminator="\n")
    return text.encode(CSV_ENCODING)


def shipments_csv_bytes(df: pd.DataFrame, columns: Sequence[str] | None = None) -> bytes:
    """The filtered shipment table as CSV bytes (whitelisted columns only; header-only when empty)."""
    return csv_bytes(export_frame(df, columns))


def read_csv_bytes(data: bytes, **kwargs: Any) -> pd.DataFrame:
    """Read bytes produced by :func:`csv_bytes` (BOM-aware); convenience for tests and round trips."""
    return pd.read_csv(io.BytesIO(data), encoding=CSV_ENCODING, **kwargs)


def regional_csv_bytes(report: pd.DataFrame) -> bytes:
    """``insights.regional_report`` output as CSV bytes, exactly the table shown on screen."""
    return csv_bytes(report)


# --------------------------------------------------------------------------- #
# Summary report: Markdown -> standalone HTML
# --------------------------------------------------------------------------- #
def _inline(text: str) -> str:
    s = _html.escape(text, quote=False)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    return re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s)


def markdown_to_html(md: str) -> str:
    """Tiny Markdown renderer for the summary report (headings, tables, lists, emphasis, paragraphs).

    Input is HTML-escaped first, so report text can never inject markup.
    """
    out: list[str] = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{_inline(m.group(2))}</h{lvl}>")
            i += 1
        elif line.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            rows = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c) for c in r)]
            if rows:
                head, body = rows[0], rows[1:]
                out.append("<table><thead><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in head) + "</tr></thead><tbody>"
                           + "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in body)
                           + "</tbody></table>")
        elif re.match(r"^\s*[-*]\s+", line) or re.match(r"^\s*\d+\.\s+", line):
            ordered = bool(re.match(r"^\s*\d+\.\s+", line))
            items = []
            pat = r"^\s*\d+\.\s+(.*)$" if ordered else r"^\s*[-*]\s+(.*)$"
            while i < len(lines) and re.match(pat, lines[i]):
                items.append(re.match(pat, lines[i]).group(1))
                i += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{_inline(t)}</li>" for t in items) + f"</{tag}>")
        else:
            para = [line.strip()]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r"^(#{1,4}\s|\s*\||\s*[-*]\s|\s*\d+\.\s)", lines[i]):
                para.append(lines[i].strip())
                i += 1
            out.append(f"<p>{_inline(' '.join(para))}</p>")
    return "\n".join(out)


_HTML_STYLE = (
    "body{font-family:Inter,-apple-system,'Segoe UI',Roboto,Arial,sans-serif;max-width:900px;margin:2rem auto;"
    "padding:0 1rem;color:#1E293B;line-height:1.55}h1{color:#0F172A;font-size:1.7rem}h2{color:#0F172A;margin-top:2rem;"
    "border-bottom:1px solid #E2E8F0;padding-bottom:.3rem}h3{color:#0F172A;font-size:1.05rem}"
    "table{border-collapse:collapse;margin:1rem 0;width:100%}th,td{border:1px solid #E2E8F0;padding:.4rem .6rem;"
    "text-align:left;font-size:.92rem}th{background:#F1F5F9}code{background:#F1F5F9;padding:.05rem .3rem;border-radius:4px}"
)


def summary_html_bytes(markdown_text: str, title: str = "DeliveryIQ - Delivery Performance Summary") -> bytes:
    """Standalone HTML page (UTF-8) rendering the Markdown summary."""
    doc = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
           "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
           f"<title>{_html.escape(title)}</title><style>{_HTML_STYLE}</style></head><body>\n"
           f"{markdown_to_html(markdown_text)}\n</body></html>")
    return doc.encode("utf-8")


# --------------------------------------------------------------------------- #
# Small formatting helpers
# --------------------------------------------------------------------------- #
def human_size(n_bytes: int | float) -> str:
    """Bytes as a readable size: ``1536`` -> ``1.5 KB``."""
    try:
        n = float(n_bytes)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(n) or n < 0:
        return "—"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"  # pragma: no cover


def _slug(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")


def filter_slug(states: Sequence[str] | None = None, categories: Sequence[str] | None = None,
                extra: Sequence[str] | None = None) -> str:
    """Short, file-name-safe description of the state / category filters (``""`` when none is active).

    One or two states are listed (``RJ-SP``), more become ``5-states``; one category is named
    (``health-beauty``), more become ``3-categories``; ``extra`` parts (e.g. ``late``) are appended.
    """
    parts: list[str] = []
    st_ = sorted({str(x) for x in (states or []) if str(x).strip()})
    if st_:
        parts.append("-".join(_slug(x).upper() for x in st_) if len(st_) <= 2 else f"{len(st_)}-states")
    ca_ = sorted({str(x) for x in (categories or []) if str(x).strip()})
    if ca_:
        parts.append(_slug(ca_[0]) if len(ca_) == 1 else f"{len(ca_)}-categories")
    parts += [_slug(x) for x in (extra or []) if _slug(x)]
    return "_".join(p for p in parts if p)


def export_filename(kind: str, date_range: tuple | None = None, ext: str = "csv", states: Sequence[str] | None = None,
                    categories: Sequence[str] | None = None, extra: Sequence[str] | None = None) -> str:
    """Predictable file name that records the filters, e.g.
    ``deliveryiq_shipments_SP_health-beauty_2017-01-05_to_2018-08-29.csv``.

    The state / category filters (see :func:`filter_slug`) sit between the kind and the date span so a
    filtered export can never be mistaken for the full one.
    """
    safe = re.sub(r"[^a-z0-9]+", "_", str(kind).lower()).strip("_") or "export"
    flt = filter_slug(states, categories, extra)
    span = ""
    if date_range and len(date_range) == 2 and all(d is not None for d in date_range):
        s, e = (pd.Timestamp(d) for d in date_range)
        span = f"_{s:%Y-%m-%d}_to_{e:%Y-%m-%d}"
    return f"deliveryiq_{safe}{'_' + flt if flt else ''}{span}.{ext.lstrip('.')}"


def _fin(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _pct(x: Any, nd: int = 1) -> str:
    v = _fin(x)
    return "n/a" if v is None else f"{100 * v:.{nd}f}%"


def _pp(x: Any, nd: int = 1) -> str:
    v = _fin(x)
    return "n/a" if v is None else f"{100 * v:+.{nd}f} pp"


def _int(x: Any) -> str:
    v = _fin(x)
    return "n/a" if v is None else f"{int(round(v)):,}"


def _num(x: Any, nd: int = 2) -> str:
    v = _fin(x)
    return "n/a" if v is None else f"{v:,.{nd}f}"


def _state(code: Any) -> str:
    name = STATE_NAMES.get(str(code))
    return f"{code} ({name})" if name else str(code)


def _cat(c: Any) -> str:
    s = str(c).replace("_", " ").strip()
    return s[:1].upper() + s[1:]


def _p(p: Any) -> str:
    v = _fin(p)
    if v is None:
        return "p = n/a"
    if v <= 1e-300:  # 0, or an underflowed value clipped to P_FLOOR (or a Bonferroni multiple of it)
        return "p < 0.001"
    return f"p = {v:.2g}" if v < 0.001 else f"p = {v:.3f}"


# --------------------------------------------------------------------------- #
# Insight presentation (type label, severity order, supporting metrics)
# --------------------------------------------------------------------------- #
OBSERVED = "Observed"
ASSOCIATION = "Statistical association"
DATA_NOTE = "Data sufficiency"
_ASSOCIATION_KINDS = {"region_test", "month_high", "same_state", "weight_freight", "processing_time"}
_DATA_KINDS = {"insufficient_data", "no_data"}
_SEVERITY_RANK = {"alert": 0, "watch": 1, "info": 2}


def insight_type_label(insight: Any) -> str:
    """``Observed`` (rates / rankings with intervals), ``Statistical association`` (hypothesis tests,
    correlations) or ``Data sufficiency``."""
    kind = getattr(insight, "kind", "")
    if kind in _DATA_KINDS:
        return DATA_NOTE
    return ASSOCIATION if kind in _ASSOCIATION_KINDS else OBSERVED


def sort_insights(insights: Iterable[Any]) -> list[Any]:
    """Overview first, data-sufficiency notes last, otherwise alert > watch > info (stable)."""
    items = list(insights)

    def key(ix_i):
        ix, i = ix_i
        if getattr(i, "kind", "") == "overview":
            return (0, 0, ix)
        if getattr(i, "kind", "") in _DATA_KINDS:
            return (3, 0, ix)
        return (1, _SEVERITY_RANK.get(getattr(i, "severity", "info"), 2), ix)

    return [i for _, i in sorted(enumerate(items), key=key)]


SINGLE_REGION_SECTION = "regional comparison (only one region in view)"


def comparable_insights(insights: Iterable[Any], min_orders: int) -> list[Any]:
    """Drop findings that compare a segment with itself.

    With a single customer state in view the "region vs overall" screen is vacuous (the one region IS the
    overall), so its "none of the 1 regions is above" card is removed and the comparison is listed among the
    skipped sections of the data-sufficiency note instead.  Everything else passes through unchanged.
    """
    from .insights import Insight  # local import: keeps this module importable without the insights stack

    items = list(insights)
    vacuous = [i for i in items if getattr(i, "kind", "") == "region_high"
               and int((getattr(i, "metrics", None) or {}).get("n_screened", 2) or 0) < 2
               and not (getattr(i, "metrics", None) or {}).get("n_flagged")]
    if not vacuous:
        return items
    out = [i for i in items if not any(i is v for v in vacuous)]
    overview = next((i for i in out if getattr(i, "kind", "") == "overview"), None)
    n = int(getattr(overview, "n", 0) or 0)
    old = next((i for i in out if getattr(i, "kind", "") == "insufficient_data"), None)
    sections = list((getattr(old, "metrics", None) or {}).get("sections", [])) if old is not None else []
    if SINGLE_REGION_SECTION not in sections:
        sections.insert(0, SINGLE_REGION_SECTION)
    note = Insight(
        "insufficient_data", "Insufficient data for some comparisons",
        f"With {n:,} orders and a minimum of {int(min_orders):,} orders per segment, there is not enough data to report: "
        + "; ".join(sections) + ". Widen the filters or collect more orders before drawing conclusions about these.",
        {"sections": sections, "min_orders": int(min_orders), "n_orders": n}, n)
    return [note if i is old else i for i in out] if old is not None else out + [note]


def _seg_pair(seg: Mapping, key: str, label_fn=str) -> tuple[str, str]:
    return label_fn(seg.get(key, "?")), f"{_pct(seg.get('late_rate'))} late (n = {_int(seg.get('n_orders'))})"


def insight_metric_pairs(insight: Any) -> list[tuple[str, str]]:
    """Up to ~5 supporting ``(label, value)`` pairs for an insight card, taken from ``insight.metrics``."""
    m = getattr(insight, "metrics", None) or {}
    kind = getattr(insight, "kind", "")
    out: list[tuple[str, str]] = []
    segs = m.get("segments") or []
    if kind == "overview":
        out = [("Late rate", _pct(m.get("late_rate"))),
               ("95% interval" if m.get("alpha", 0.05) == 0.05 else "Interval",
                f"{_pct(m.get('late_ci_low'))} to {_pct(m.get('late_ci_high'))}"),
               ("Late orders", _int(m.get("n_late"))), ("Mean delivery", f"{_num(m.get('mean_duration'), 1)} days"),
               ("Mean days late", f"{_num(m.get('mean_late_days'), 1)} days")]
    elif kind in ("region_high", "region_low", "pair_high", "pair_low"):
        key = "state_pair" if kind.startswith("pair") else "customer_state"
        out = [("Flagged / screened", f"{_int(m.get('n_flagged'))} / {_int(m.get('n_screened'))}"),
               ("Overall late rate", _pct(m.get("overall_late_rate")))]
        for s in segs[:3]:
            lab, val = _seg_pair(s, key, str if kind.startswith("pair") else _state)
            out.append((lab, val))
    elif kind == "region_test":
        out = [("Cramér's V", f"{_num(m.get('cramers_v'), 3)} ({m.get('effect_label', 'n/a')})"),
               ("Chi-square", f"{_num(m.get('chi2'), 1)} (df {_int(m.get('dof'))}), {_p(m.get('p_value'))}"),
               ("Lowest region rate", _pct(m.get("min_late_rate"))), ("Highest region rate", _pct(m.get("max_late_rate")))]
    elif kind == "month_high":
        out = [("Flagged / tested", f"{_int(m.get('n_flagged'))} / {_int(m.get('n_tested'))}")]
        for r in (m.get("months") or [])[:3]:
            out.append((str(r.get("month")), f"{_pct(r.get('late_rate'))} late (n = {_int(r.get('n_orders'))})"))
    elif kind == "month_extremes":
        out = [(f"Worst: {m.get('worst_month')}", f"{_pct(m.get('worst_late_rate'))} (n = {_int(m.get('worst_n'))})"),
               (f"Best: {m.get('best_month')}", f"{_pct(m.get('best_late_rate'))} (n = {_int(m.get('best_n'))})")]
    elif kind == "category_duration":
        out = [("Overall mean delivery", f"{_num(m.get('overall_mean_duration'), 1)} days")]
        out += [(_cat(s.get("product_category")), f"{_num(s.get('mean_duration'), 1)} days (n = {_int(s.get('n_orders'))})")
                for s in segs[:3]]
    elif kind == "category_delay":
        out = [("Overall late rate", _pct(m.get("overall_late_rate"))),
               ("Reliably above overall", f"{_int(len(m.get('flagged') or []))} of top {_int(len(segs))} shown "
                                          f"({_int(m.get('n_reliably_above'))} of {_int(m.get('n_screened'))} screened)")]
        out += [(_cat(s.get("product_category")), f"{_pct(s.get('late_rate'))} late (n = {_int(s.get('n_orders'))})")
                for s in segs[:3]]
    elif kind == "category_freight":
        out = [("Overall mean freight", f"R$ {_num(m.get('overall_mean_freight'), 2)}")]
        out += [(_cat(s.get("product_category")), f"R$ {_num(s.get('mean_freight'), 2)} (n = {_int(s.get('n_orders'))})")
                for s in segs[:3]]
    elif kind == "same_state":
        out = [("Cross-state", f"{_pct(m.get('cross_late_rate'))} late (n = {_int(m.get('cross_n'))})"),
               ("Same-state", f"{_pct(m.get('same_late_rate'))} late (n = {_int(m.get('same_n'))})"),
               ("Difference", f"{_pp(m.get('diff'))} ({_pp(m.get('diff_ci_low'))} to {_pp(m.get('diff_ci_high'))})"),
               ("Test", _p(m.get("p_value")))]
    elif kind == "weight_freight":
        for lab, key in (("Weight vs delivery time", "weight_vs_duration"), ("Freight vs delivery time", "freight_vs_duration"),
                         ("Weight vs freight", "weight_vs_freight")):
            r = m.get(key)
            if isinstance(r, Mapping):
                out.append((lab, f"rho {_num(r.get('rho'), 2)} ({r.get('strength_label', 'n/a')}), n = {_int(r.get('n'))}"))
    elif kind == "processing_time":
        out = [("Slowest quarter late", f"{_pct(m.get('slow_late_rate'))} (n = {_int(m.get('slow_n'))})"),
               ("Others late", f"{_pct(m.get('rest_late_rate'))} (n = {_int(m.get('rest_n'))})"),
               ("Difference", f"{_pp(m.get('diff'))} ({_pp(m.get('diff_ci_low'))} to {_pp(m.get('diff_ci_high'))})"),
               ("Median processing", f"{_num(m.get('median_processing_days'), 1)} days")]
    elif kind == "insufficient_data":
        out = [("Orders in view", _int(m.get("n_orders"))), ("Minimum per segment", _int(m.get("min_orders"))),
               ("Skipped", f"{_int(len(m.get('sections') or []))} comparison(s)")]
    return out


# --------------------------------------------------------------------------- #
# Data explorer
# --------------------------------------------------------------------------- #
def explorer_view(df: pd.DataFrame, sort_by: str | None = None, ascending: bool = True, page: int = 1,
                  page_size: int = 50) -> tuple[pd.DataFrame, dict]:
    """Sort ``df`` and return one page of it plus paging info.

    The info dict has ``total`` (rows), ``page`` (clamped, 1-based), ``n_pages``, ``start`` / ``end``
    (1-based, inclusive row numbers shown; 0 when empty) and ``page_size``.  Missing values sort last.
    """
    page_size = max(1, int(page_size))
    total = int(len(df))
    n_pages = max(1, math.ceil(total / page_size))
    page = min(max(1, int(page)), n_pages)
    d = df
    if sort_by and sort_by in df.columns and total:
        d = df.sort_values(sort_by, ascending=ascending, kind="mergesort", na_position="last")
    lo = (page - 1) * page_size
    view = d.iloc[lo:lo + page_size]
    return view, {"total": total, "page": page, "n_pages": n_pages, "page_size": page_size,
                  "start": lo + 1 if total else 0, "end": lo + len(view)}


def explorer_filter(df: pd.DataFrame, date_range: tuple | None = None, states: Sequence[str] | None = None,
                    categories: Sequence[str] | None = None, status: str | None = None,
                    search: str | None = None) -> pd.DataFrame:
    """Explorer-level filters layered on an already (sidebar-)filtered frame.

    ``status`` is ``"all"`` / ``None`` (no filter), ``"on_time"`` or ``"late"``.
    """
    st_ = None if status in (None, "", "all") else [status]
    return an.apply_filters(df, date_range=date_range, states=states or None, categories=categories or None,
                            statuses=st_, search=search)


# --------------------------------------------------------------------------- #
# Model-evaluation helpers (read models/model_metrics.json)
# --------------------------------------------------------------------------- #
def _g(d: Any, *path: str, default: Any = None) -> Any:
    for p in path:
        if not isinstance(d, Mapping) or p not in d:
            return default
        d = d[p]
    return d


def selected_model_name(metrics: Mapping | None) -> str | None:
    """Name of the selected model in the metrics file (None if unknown)."""
    name = _g(metrics, "selected_model", "name")
    return name if isinstance(name, str) else None


def model_label(metrics: Mapping | None, name: str | None) -> str:
    """Human label of a model key (``label`` field, else prettified key)."""
    lab = _g(metrics, "models", name, "label") if name else None
    return str(lab) if lab else _cat(name or "model")


def split_table(metrics: Mapping | None) -> pd.DataFrame:
    """One row per chronological window: purchase period, orders, late orders, late rate."""
    cols = ["Window", "Role", "From", "To", "Orders (n)", "Late orders", "Late rate"]
    roles = {"train": "Fits the model", "validation": "Tuning, calibration, thresholds, risk-band cuts",
             "test": "Untouched until the end: honest estimate"}
    rows = []
    for w in ("train", "validation", "test"):
        s = _g(metrics, "split", w)
        if isinstance(s, Mapping):
            rows.append({"Window": w.capitalize(), "Role": roles[w], "From": s.get("start"), "To": s.get("end"),
                         "Orders (n)": s.get("n"), "Late orders": s.get("n_late"), "Late rate": s.get("late_rate")})
    return pd.DataFrame(rows, columns=cols)


def split_notes(metrics: Mapping | None) -> list[str]:
    """Notes about the split written by the training script (selection bias, drift, censoring)."""
    notes = _g(metrics, "split", "notes", default=[])
    return [str(n) for n in notes] if isinstance(notes, list) else []


_METRIC_COLS = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("precision", "Precision"), ("recall", "Recall"),
                ("f1", "F1"), ("brier", "Brier"), ("accuracy", "Accuracy"), ("lift_top10pct", "Lift (top 10%)")]


def model_metrics_table(metrics: Mapping | None) -> pd.DataFrame:
    """Every model on both windows, test window first.  ``No-skill PR-AUC`` = late share of that window."""
    cols = ["Model", "Window", "Selected", "Orders (n)", "No-skill PR-AUC"] + [c for _, c in _METRIC_COLS]
    models = _g(metrics, "models", default={})
    sel = selected_model_name(metrics)
    rows = []
    if isinstance(models, Mapping):
        for window in ("test", "validation"):
            for name, entry in models.items():
                w = _g(entry, window)
                if not isinstance(w, Mapping):
                    continue
                row = {"Model": model_label(metrics, name), "Window": window.capitalize(),
                       "Selected": "yes" if name == sel else "", "Orders (n)": w.get("n"),
                       "No-skill PR-AUC": w.get("positive_rate")}
                row.update({lab: w.get(k) for k, lab in _METRIC_COLS})
                rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def risk_band_table(metrics: Mapping | None) -> pd.DataFrame:
    """Observed late rate per predicted risk band for the validation and test windows."""
    cols = ["Window", "Band", "Orders (n)", "Late orders", "Late rate", "CI low", "CI high"]
    rows = []
    for w in ("validation", "test"):
        for b in _g(metrics, "risk_bands", w, default=[]) or []:
            if isinstance(b, Mapping):
                rows.append({"Window": w.capitalize(), "Band": b.get("band"), "Orders (n)": b.get("n"),
                             "Late orders": b.get("n_late"), "Late rate": b.get("late_rate"),
                             "CI low": b.get("late_ci_low"), "CI high": b.get("late_ci_high")})
    return pd.DataFrame(rows, columns=cols)


def confusion_counts(metrics: Mapping | None, model: str | None = None, window: str = "test") -> dict | None:
    """tn / fp / fn / tp, recall, precision and false-alarm numbers (None when unavailable)."""
    name = model or selected_model_name(metrics)
    cm = _g(metrics, "models", name, f"confusion_matrix_{window}") if name else None
    try:
        a = np.asarray(cm, dtype="float64")
        if a.shape != (2, 2) or not np.isfinite(a).all():
            return None
    except (TypeError, ValueError):
        return None
    tn, fp, fn, tp = (int(x) for x in a.ravel())
    pos, neg, flagged = tp + fn, tn + fp, tp + fp
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "n": tn + fp + fn + tp, "n_late": pos, "n_on_time": neg,
            "flagged": flagged, "recall": tp / pos if pos else float("nan"),
            "precision": tp / flagged if flagged else float("nan"),
            "prevalence": pos / (pos + neg) if (pos + neg) else float("nan"),
            "false_alarm_rate": fp / neg if neg else float("nan"),
            "threshold": _g(metrics, "models", name, "threshold"), "name": name}


def confusion_reading(metrics: Mapping | None, model: str | None = None, window: str = "test") -> str | None:
    """Plain-language reading of the confusion matrix, written from its counts."""
    c = confusion_counts(metrics, model, window)
    if c is None:
        return None
    lift = c["precision"] / c["prevalence"] if c["prevalence"] and np.isfinite(c["precision"]) else float("nan")
    thr = f" (probability of at least {c['threshold']:.3f})" if _fin(c["threshold"]) is not None else ""
    return (f"On the {window} window the model marked {c['flagged']:,} of {c['n']:,} orders as likely late{thr}. "
            f"It caught {c['tp']:,} of the {c['n_late']:,} orders that really were late (recall {_pct(c['recall'])}) "
            f"and missed {c['fn']:,}. Only {_pct(c['precision'])} of the flagged orders were late "
            f"(precision), against a late share of {_pct(c['prevalence'])} in all orders"
            + (f", i.e. {lift:.1f}x the base rate" if np.isfinite(lift) else "")
            + f". It also flagged {c['fp']:,} on-time orders ({_pct(c['false_alarm_rate'])} of all on-time orders). "
            "A flag is therefore a weak hint, not a reliable alarm.")


def model_headline(metrics: Mapping | None) -> dict | None:
    """Key numbers of the selected model on the test and validation windows (None if unavailable)."""
    name = selected_model_name(metrics)
    entry = _g(metrics, "models", name) if name else None
    if not isinstance(entry, Mapping) or not isinstance(entry.get("test"), Mapping):
        return None
    t, v = entry["test"], entry.get("validation") or {}
    base = _fin(t.get("positive_rate"))
    pr = _fin(t.get("pr_auc"))
    vbase, vpr = _fin(v.get("positive_rate")), _fin(v.get("pr_auc"))
    roc_ci = _g(entry, "test_ci95", "roc_auc") or [None, None]
    pr_ci = _g(entry, "test_ci95", "pr_auc") or [None, None]
    return {
        "name": name, "label": model_label(metrics, name),
        "test_n": t.get("n"), "test_late_rate": base,
        "test_roc_auc": _fin(t.get("roc_auc")), "test_roc_ci": tuple(_fin(x) for x in roc_ci[:2]),
        "test_pr_auc": pr, "test_pr_ci": tuple(_fin(x) for x in pr_ci[:2]),
        "test_lift": pr / base if pr is not None and base else None,
        "val_roc_auc": _fin(v.get("roc_auc")), "val_pr_auc": vpr, "val_late_rate": vbase,
        "val_lift": vpr / vbase if vpr is not None and vbase else None,
        "test_brier": _fin(t.get("brier")), "test_precision": _fin(t.get("precision")),
        "test_recall": _fin(t.get("recall")), "test_accuracy": _fin(t.get("accuracy")),
        "test_flagged_share": _fin(t.get("predicted_positive_rate")),
        "degraded": bool(_g(metrics, "selected_model", "test_performance_degraded", default=False)),
    }


def signal_phrase(roc_auc: float | None) -> tuple[str, str]:
    """``(noun phrase, comparison phrase)`` describing a ROC-AUC without overstating it.

    Example: 0.62 -> ``("a modest signal", "only modestly better than chance")``.
    """
    v = _fin(roc_auc)
    if v is None:
        return "an unknown signal", "of unknown quality"
    if v < 0.55:
        return "almost no signal", "barely better than chance"
    if v < 0.65:
        return "a modest signal", "only modestly better than chance"
    if v < 0.75:
        return "a moderate signal", "moderately better than chance"
    return "a clear but imperfect signal", "clearly better than chance, though far from perfect"


def honest_summary(metrics: Mapping | None) -> list[str]:
    """Bulleted, number-driven statement of what the model can and cannot do (never overstates)."""
    h = model_headline(metrics)
    if h is None:
        return []
    out = []
    ci = h["test_roc_ci"]
    ci_txt = f" (95% bootstrap interval {_num(ci[0], 2)} to {_num(ci[1], 2)})" if ci[0] is not None and ci[1] is not None else ""
    noun = signal_phrase(h["test_roc_auc"])[0]
    out.append(
        f"**Ranking power: {noun}.** On the untouched test window ({_int(h['test_n'])} orders, {_pct(h['test_late_rate'])} late) "
        f"the {h['label'].lower()} reaches ROC-AUC {_num(h['test_roc_auc'], 2)}{ci_txt} (0.5 = no skill) and PR-AUC "
        f"{_num(h['test_pr_auc'], 3)} against a no-skill level of {_num(h['test_late_rate'], 3)}"
        + (f", a lift of {h['test_lift']:.1f}x." if h["test_lift"] else "."))
    weaker = (h["val_roc_auc"] is not None and h["test_roc_auc"] is not None and h["val_roc_auc"] - h["test_roc_auc"] > 0.02)
    out.append(
        f"**{'Weaker than on validation' if weaker else 'Compared with validation'}.** The validation window gave ROC-AUC "
        f"{_num(h['val_roc_auc'], 2)} and a PR-AUC lift of {_num(h['val_lift'], 1)}x, but that window was also used for tuning, "
        "calibration, the threshold and the risk-band cuts, so those figures are optimistic. The test window is the honest estimate.")
    sp = {w: _g(metrics, "split", w, "late_rate") for w in ("train", "validation", "test")}
    if all(_fin(x) is not None for x in sp.values()):
        out.append(
            f"**Drift between windows.** The late rate moved from {_pct(sp['train'])} (train) to {_pct(sp['validation'])} "
            f"(validation) and {_pct(sp['test'])} (test). Relationships learned on the earlier period may not hold later, and "
            "purchase-time attributes cannot anticipate episodes of unusual delay.")
    cav = _g(metrics, "selected_model", "test_window_caveat") or {}
    if _fin(cav.get("brier")) is not None and _fin(cav.get("brier_baseline")) is not None:
        better = not cav.get("brier_not_better_than_baseline", False)
        out.append(
            f"**{'Probabilities are not better than the base rate' if not better else 'Probabilities on the test window'}.** Mean predicted risk was {_pct(cav.get('mean_predicted'))} "
            f"against an observed {_pct(cav.get('observed_late_rate'))}; Brier score {_num(cav['brier'], 4)} vs {_num(cav['brier_baseline'], 4)} "
            + ("for always quoting the base rate (better than it)." if better
               else "for always quoting the base rate (no better than it). Treat probabilities as relative risk, not as exact chances."))
    td = _g(metrics, "risk_bands", "test_diagnostics") or {}
    bands = {b.get("band"): b for b in _g(metrics, "risk_bands", "test", default=[]) or [] if isinstance(b, Mapping)}
    if {"Low", "Moderate", "High"} <= set(bands):
        ov = _g(td, "adjacent_intervals_overlap", "Moderate-High")
        lm = _g(td, "adjacent_intervals_overlap", "Low-Moderate")
        out.append(
            f"**Risk bands on the test window.** Observed late rate: Low {_pct(bands['Low']['late_rate'])} "
            f"(n = {_int(bands['Low']['n'])}), Moderate {_pct(bands['Moderate']['late_rate'])} (n = {_int(bands['Moderate']['n'])}), "
            f"High {_pct(bands['High']['late_rate'])} (n = {_int(bands['High']['n'])}). "
            + (("Moderate and High cannot be told apart statistically (their intervals overlap)"
                + ("; only Low is clearly lower." if not lm else "; neither can Low and Moderate be separated."))
               if ov else ("Low and Moderate cannot be told apart statistically (their intervals overlap)." if lm
                           else "The adjacent bands are statistically separated at this sample size.")))
    cs = _g(metrics, "split", "covariate_support", "test") or {}
    if _fin(cs.get("share_outside")) is not None and cs["share_outside"] >= 0.01:
        by_feat = {k: v for k, v in (cs.get("share_by_feature") or {}).items() if _fin(v) is not None}
        top_feat = max(by_feat, key=by_feat.get) if by_feat else None
        out.append(
            f"**Out-of-range inputs.** {_pct(cs['share_outside'])} of test orders ({_int(cs.get('n_outside'))}) have a numeric feature "
            "outside the range seen in training"
            + (f" (mostly {_cat(top_feat).lower()})" if top_feat else "") + ", where the model can only extrapolate.")
    return out


def metric_guidance_points(metrics: Mapping | None) -> list[str]:
    """Why accuracy misleads at this prevalence and which metrics to read instead (from the test numbers)."""
    h = model_headline(metrics)
    if h is None or h["test_late_rate"] is None:
        return []
    base = h["test_late_rate"]
    return [
        f"**Accuracy is misleading.** Only {_pct(base)} of test orders are late, so answering \"on time\" for every order is "
        f"{_pct(1 - base)} accurate while finding no late order. The model's own accuracy is {_pct(h['test_accuracy'])} because it "
        f"flags {_pct(h['test_flagged_share'])} of orders.",
        f"**PR-AUC (average precision)** rewards finding the rare late orders with few false alarms. The no-skill level equals the late "
        f"share ({_num(base, 3)}); the model reaches {_num(h['test_pr_auc'], 3)}.",
        f"**Recall and precision at the chosen threshold** describe the real trade-off: recall {_pct(h['test_recall'])}, "
        f"precision {_pct(h['test_precision'])} on the test window.",
        "**ROC-AUC** is the chance that a random late order scores higher than a random on-time one (0.5 = no skill); it is less "
        "sensitive to the rare-event problem but can look flattering when precision is poor.",
        "**Calibration (Brier score, calibration curve)** shows whether a quoted probability matches the observed share of late orders."]


dig = _g  # public alias: nested-dict lookup that returns ``default`` instead of raising
