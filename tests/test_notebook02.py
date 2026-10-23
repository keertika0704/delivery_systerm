"""Consistency tests for the executed notebook notebooks/02_statistical_analysis.ipynb.

They read the saved notebook (no re-execution) and check it against the processed data: the structural
guarantees that the review findings (time-instability of the same-state gap, the missing-category bucket,
the bootstrap interval, the freight-bin pattern) depend on.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import nbformat
import pandas as pd
import pytest

from src import analytics as A
from src.config import MODEL_METRICS_PATH, ORDERS_DELIVERED_PATH, ROOT

NB_PATH = ROOT / "notebooks" / "02_statistical_analysis.ipynb"

pytestmark = [
    pytest.mark.skipif(not NB_PATH.exists(), reason="notebook 02 not built"),
    pytest.mark.skipif(not ORDERS_DELIVERED_PATH.exists(), reason="processed data not built"),
]


@pytest.fixture(scope="module")
def nb():
    return nbformat.read(NB_PATH, as_version=4)


@pytest.fixture(scope="module")
def key(nb) -> dict:
    found = None
    for c in nb.cells:
        if c.cell_type == "code":
            for o in c.get("outputs", []):
                m = re.search(r"KEY_RESULTS (\{.*\})", o.get("text", "") if o.get("output_type") == "stream" else "")
                if m:
                    found = json.loads(m.group(1))
    assert found is not None, "KEY_RESULTS missing from the executed notebook"
    return found


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    d = pd.read_parquet(ORDERS_DELIVERED_PATH)
    d["late"] = d["is_late"].astype(int)
    return d


def _text(nb) -> str:
    return "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")


def test_notebook_has_no_errors_or_unfilled_placeholders(nb):
    for c in nb.cells:
        if c.cell_type == "code":
            assert not any(o.get("output_type") == "error" for o in c.get("outputs", []))
        else:
            assert "«" not in c.source and "»" not in c.source, "unfilled narrative placeholder"


# -- finding 1: the same-state gap is not stable over time ---------------------------------------------------
def test_same_state_gap_by_window_matches_data_and_is_discussed(nb, key, df):
    wins = json.loads(MODEL_METRICS_PATH.read_text())["split"]["windows"]
    ts = pd.to_datetime(df["order_purchase_timestamp"])
    for tag, name in [("train", "train"), ("val", "validation"), ("test", "test")]:
        w0, w1 = wins[name]
        m = (ts >= pd.Timestamp(w0)) & ((ts < pd.Timestamp(w1)) if w1 else True)
        assert key[f"stab_{tag}_same"] == pytest.approx(df.loc[m & (df.same_state == 1), "late"].mean(), abs=1e-9)
        assert key[f"stab_{tag}_cross"] == pytest.approx(df.loc[m & (df.same_state == 0), "late"].mean(), abs=1e-9)
    # the sign change reported by the model card
    assert key["stab_train_same"] < key["stab_train_cross"] and key["stab_val_same"] < key["stab_val_cross"]
    assert key["stab_test_same"] > key["stab_test_cross"]
    assert 0 < key["stab_n_months_same_higher"] < key["stab_n_months"]
    text = _text(nb)
    assert "not stable" in text and "test window" in text
    # the instability is stated in the RQ3 interpretation, in the limitations and in the conclusions table
    rq3 = next(c.source for c in nb.cells if c.cell_type == "markdown" and c.source.startswith("### Interpretation of RQ3"))
    limits = next(c.source for c in nb.cells if c.cell_type == "markdown" and c.source.startswith("## 10. Limitations"))
    concl = next(c.source for c in nb.cells if c.cell_type == "markdown" and c.source.startswith("## 11. Conclusions"))
    assert "not stable" in rq3 and "reverses" in rq3
    assert "same-state advantage" in limits and "positive in the test window" in limits
    assert re.search(r"RQ3 Same vs cross state.*not stable over time", concl)


# -- finding 2: the missing-category bucket is not a product category ----------------------------------------
def test_unknown_category_excluded_from_rq2_and_counts_match_insights(nb, key, df):
    unk = df["product_category"].astype(str) == "unknown"
    assert key["n_unknown_cat"] == int(unk.sum()) == int(df["category_missing"].sum())
    assert key["n_cat_known"] == int((~unk).sum())
    cp = A.category_performance(df[~unk], min_orders=100)
    assert "unknown" not in set(cp["product_category"].astype(str))
    assert key["n_cat_all"] == len(cp) == 73
    assert key["n_cat_reliable"] == int(cp["reliable"].sum()) == key["n_cat_tested"]
    text = _text(nb)
    assert "left out of every RQ2 test and ranking" in text


# -- finding 3: bootstrap intervals contain their point estimate -----------------------------------------------
def test_bootstrap_intervals_contain_point_estimates(key):
    for v in ("state", "cat"):
        assert key[f"v_{v}_lo"] <= key[f"v_{v}_corr"] <= key[f"v_{v}_hi"], v


def test_effect_size_table_intervals_contain_corrected_v(nb):
    """Parse the printed Cramer's V table: every 'corrected V, bootstrap 95% CI' must bracket 'V (bias-corrected)'."""
    checked = 0
    for c in nb.cells:
        for o in c.get("outputs", []) if c.cell_type == "code" else []:
            plain = o.get("data", {}).get("text/plain", "") if o.get("output_type") == "execute_result" else ""
            if "V (bias-corrected)" not in plain:
                continue
            for line in plain.splitlines():
                m = re.search(r"\s(\d\.\d+)\s+(\d\.\d+) to (\d\.\d+)\s+\w+\s*$", line)
                if m:
                    corr, lo, hi = map(float, m.groups())
                    assert lo - 5e-4 <= corr <= hi + 5e-4, line
                    checked += 1
    assert checked == 4


# -- finding 4: freight bins ----------------------------------------------------------------------------------------
def test_freight_bin_pattern_is_quoted_from_the_data(nb, key, df):
    fb = A.freight_bins(df, min_orders=100)
    assert key["fb_first_rate"] == pytest.approx(float(fb.late_rate.iloc[0]), abs=1e-12)
    assert key["fb_last_rate"] == pytest.approx(float(fb.late_rate.iloc[-1]), abs=1e-12)
    assert key["fb_monotone"] and key["fb_last_rate"] >= 1.5 * key["fb_first_rate"]
    text = _text(nb)
    assert "understates a stepwise pattern" in text
    assert "neither says much about lateness" not in text
    assert f"{100 * key['fb_first_rate']:.1f}%" in text and f"{100 * key['fb_last_rate']:.1f}%" in text


# -- review round 2: time-section wording must match the data, and overclaims must stay out -----------------------
def test_time_section_text_matches_monthly_data(nb, df):
    ym = pd.to_datetime(df["order_purchase_timestamp"]).dt.to_period("M").astype(str)
    counts = ym.value_counts()
    tiny = sorted(counts[counts < 100].index)
    assert tiny == ["2016-09", "2016-12"] and (counts[tiny] == 1).all()   # only these two are excluded by n >= 100
    assert counts["2016-10"] == 265 and "2016-11" not in counts.index
    intro = next(c.source for c in nb.cells if c.cell_type == "markdown" and c.source.startswith("## 6. RQ5"))
    assert "2016-09 and 2016-12" in intro and "2016-10, with n = 265, is kept" in intro
    assert "(2016) are excluded" not in intro
    interp = next(c.source for c in nb.cells if c.cell_type == "markdown" and c.source.startswith("### Interpretation of RQ5"))
    assert "Only January to August appear in both years" not in interp
    assert "October also appears in 2016 and 2017" in interp
    # months with data in both 2017 and 2018 are exactly Jan..Aug
    y = pd.to_datetime(df["order_purchase_timestamp"])
    both = set(y[y.dt.year == 2017].dt.month) & set(y[y.dt.year == 2018].dt.month)
    assert both == set(range(1, 9))


def test_monthly_figure_cell_breaks_line_over_missing_months(nb):
    src = next(c.source for c in nb.cells if c.cell_type == "code" and "26_monthly_late_rate" in c.source)
    assert "period_range" in src and ".reindex(cal)" in src      # calendar axis -> NaN gaps, no line across 2016-11/12


def test_independence_and_prespecification_claims_are_not_overstated(nb):
    text = _text(nb)
    assert "every result is checked a second time" not in text
    assert "chosen *before* looking at results" not in text
    assert "not independently recomputed here" in text
    assert "a choice was made after seeing the data" in text
