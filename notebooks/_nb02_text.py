"""Narrative (Markdown) cells for notebooks/02_statistical_analysis.ipynb.

Numbers appear as ``«name»`` placeholders; ``_build_nb02.py`` fills them from values it recomputes from the
processed data (and then checks them against the executed notebook output).  Qualitative statements (e.g.
"small", "negligible", "reverses") are guarded by ``check(V)`` below, which fails the build if the data
no longer supports them.
"""
from __future__ import annotations

MD: dict[str, str] = {}


def check(V: dict) -> None:
    """Fail loudly if a qualitative claim made in the narrative is not true for the computed values."""
    claims = {
        "state V is 'small' (0.1 <= V < 0.3)": 0.1 <= V["v_state"] < 0.3,
        "category V is negligible (< 0.1) even before bias correction": V["v_cat"] < 0.1,
        "same-state V is negligible (< 0.1)": V["v_same"] < 0.1,
        "month-of-year and year-month V are small": 0.1 <= V["v_month"] < 0.3 and 0.1 <= V["v_ym"] < 0.3,
        "weekday V negligible": V["v_dow"] < 0.1,
        "RJ late rate well above SP's": V["rj_rate"] > 2 * V["sp_rate"],
        "MH odds ratio closer to 1 than crude and its CI excludes 1": abs(1 - V["or_mh"]) < abs(1 - V["or_crude"]) and V["or_mh_hi"] < 1,
        "within SP the same-state rate is HIGHER (reversal)": V["sp_same_rate"] > V["sp_cross_rate"],
        "most strata have OR < 1": V["n_strata_or_below1"] > V["n_strata_or_above1"],
        "cross-state durations longer": V["median_cross"] > V["median_same"] and V["mw_rank_biserial"] > 0.5,
        "weight~duration negligible (< 0.1)": abs(V["rho_weight_dur"]) < 0.1,
        "freight~duration moderate (0.3 <= rho < 0.5)": 0.3 <= V["rho_freight_dur"] < 0.5,
        "freight_ratio~duration weak (0.1 <= rho < 0.3)": 0.1 <= V["rho_fratio_dur"] < 0.3,
        "promise~duration strong (0.5 <= rho < 0.7)": 0.5 <= V["rho_promise_dur"] < 0.7,
        "weight~freight moderate": 0.3 <= V["rho_weight_freight"] < 0.5,
        "partial freight|weight about equal to raw rho (within 0.05)": abs(V["partial_freight_given_weight"] - V["rho_freight_dur"]) < 0.05,
        "partial weight|freight is not positive": V["partial_weight_given_freight"] < 0,
        "freight~duration weaker within cross-state than within same-state": V["rho_freight_dur_cross"] < V["rho_freight_dur_same"],
        "spike months carry a larger share of late orders than of orders": V["spike_share_late"] > V["spike_share_orders"],
        "spike months detected include 2017-11 and 2018-02 and 2018-03": set(V["spike_list"].split(",")) == {"2017-11", "2018-02", "2018-03"},
        "V shrinks when spike months are removed for state, same_state, month": V["v_state_ns"] < V["v_state"] and V["v_same_ns"] < V["v_same"] and V["v_month_ns"] < V["v_month"],
        "category V unchanged by removing spikes (within 0.005)": abs(V["v_cat_ns"] - V["v_cat"]) < 0.005,
        "same-state V stays below 0.1 without spikes": V["v_same_ns"] < 0.1,
        "processing: late rate rises from first to last bucket": V["proc_rate_last"] > 3 * V["proc_rate_first"],
        "processing~is_late weak (0.1 <= rho < 0.3)": 0.1 <= V["rho_proc_late"] < 0.3,
        "processing~duration moderate": 0.3 <= V["rho_proc_dur"] < 0.5,
        "processing and carrier->customer days nearly unrelated (|rho| < 0.1)": abs(V["rho_proc_c2c"]) < 0.1,
        "carrier->customer days~is_late at least as strong as processing~is_late": V["rho_c2c_late"] > V["rho_proc_late"],
        "Holm changes no verdict in the primary family": V["n_sig_raw"] == V["n_sig_holm"] if "n_sig_raw" in V else True,
        "state-level late rate positively related to median duration": V["rho_state_dur"] > 0.3,
        "Kruskal epsilon-squared for duration by state is 'medium' (0.08..0.26)": 0.08 <= V["kw_state_eps2"] < 0.26,
        "only a few categories survive Holm": V["n_cat_sig_holm"] <= 5 < V["n_cat_sig_raw"],
        "most states differ from the rest after Holm": V["n_state_sig_holm"] > V["n_state_tested"] / 2,
        "state V stays 'small' (>= 0.1) without spike months": V["v_state_ns"] >= 0.1,
        "state p-value underflowed (< 1e-300)": V["p_state"] <= 1.0001 * 2.2250738585072014e-308,
        "state-level late rate vs same-state share not significant": V["p_state_same"] > 0.05,
        "cross-state promise longer than same-state": V["est_cross"] > V["est_same"],
        "all four rank correlations with is_late negligible": V["max_abs_rho_late"] < 0.1,
        "promise slightly negatively related to lateness": -0.1 < V["rho_promise_late"] < 0,
        "freight_ratio~is_late not significant": V["p_fratio_late"] > 0.05,
        "March 2017 below overall, March 2018 the highest month": V["r_2017_03"] < V["BASE"] and V["r_2018_03"] == V["m_hi_rate"],
        "February differs a lot between years": V["r_2018_02"] > 3 * V["r_2017_02"],
        "processing rank-biserial near 0.3": 0.25 <= V["proc_rb"] <= 0.35,
        "category chi-square significant at 5%": V["p_cat"] < 0.05,
        "SP below the overall rate and RJ above": V["sp_rate"] < V["BASE"] < V["rj_rate"],
        "missing-category bucket is left out of RQ2 (known-category orders = all - unknown, unknown > 0)": 0 < V["n_unknown_cat"] and V["n_cat_known"] == V["N"] - V["n_unknown_cat"],
        "bootstrap intervals for corrected V contain their point estimates": all(
            V[f"v_{k}_lo"] <= V[f"v_{k}_corr"] <= V[f"v_{k}_hi"] for k in ("state", "cat")),
        "same-state late rate is LOWER in the train and validation windows but HIGHER in the test window": (
            V["stab_train_same"] < V["stab_train_cross"] and V["stab_val_same"] < V["stab_val_cross"] and V["stab_test_same"] > V["stab_test_cross"]),
        "test-window same-vs-cross gap interval excludes 0": V["stab_test_gap_lo_pp"] > 0,
        "same-state is the higher-late group in some but fewer than half of the months": 0 < V["stab_n_months_same_higher"] < V["stab_n_months"] / 2,
        "late rate rises monotonically across freight bins and the last bin is at least 1.5x the first": (
            V["fb_monotone"] and V["fb_last_rate"] >= 1.5 * V["fb_first_rate"] and V["fb_first_median_dur"] < V["fb_last_median_dur"]),
        "freight~is_late rank correlation negligible": abs(V["rho_freight_late"]) < 0.1,
    }
    bad = [k for k, ok in claims.items() if not ok]
    assert not bad, f"narrative claims no longer supported by the data: {bad}"
    print(f"narrative qualitative claims verified ({len(claims)})")


# =============================================================================
MD["title"] = """
# DeliveryIQ — Notebook 02: Statistical Analysis

**Project:** DeliveryIQ — Intelligent Shipment Analytics & Delay Prediction (Fundamentals of Data Science)
**Notebook:** `notebooks/02_statistical_analysis.ipynb`  |  **Data:** Olist Brazilian E-Commerce Public Dataset, 2016-2018 (delivered orders)

## Purpose
Notebook 01 explored the data descriptively. This notebook asks the *inferential* questions: which of the patterns we see are
larger than chance would produce, **how large** are they, and how much should we trust them? Every claim comes with
an effect size and an interval, and every rate with its sample size `n`. The chi-square, correlation, Kruskal-Wallis, Mann-Whitney and same-state
proportion results are also recomputed with plain `scipy` / `numpy` code, independently of the project module `src/statistical_analysis.py` that the dashboard uses;
the state-vs-rest and category-vs-rest comparisons (Newcombe intervals, pooled z-tests, Holm adjustment), the spike-month contrast and the correlation-strength labels use that module only.

## Research questions
| # | Question | Variables |
|---|---|---|
| **RQ1** | Does the late-delivery rate differ between customer regions (states)? | `customer_state` x `is_late` |
| **RQ2** | Does it differ between product categories? | `product_category` x `is_late` |
| **RQ3** | Are same-state (seller and customer in one state) orders delivered later / more often late than cross-state orders? | `same_state` x `is_late`, `delivery_duration_days` |
| **RQ4** | How are package weight and freight cost related to delivery duration? | `total_weight_g`, `total_freight`, `freight_ratio` vs `delivery_duration_days` |
| **RQ5** | Is there seasonality or another time pattern in lateness? | `purchase_month`, year-month, weekday |
| **RQ6** | *(Retrospective)* Is seller processing time related to lateness? | `processing_days` vs `is_late` |

**Definitions** (see `docs/ARCHITECTURE.md`): an order is *late* when it reaches the customer on a calendar date after the estimated
delivery date shown at checkout; the analysis set is the «N» delivered orders with complete timestamps, of which «K_LATE» are late
(overall late rate **«BASE»**, Wilson 95% CI «BASE_CI»). Lateness is measured **against Olist's own promise**, not against an absolute speed standard.

## Conventions
* Two-sided tests at alpha = 0.05; Holm correction when many tests are read together. Exact p-values below about 2e-308 underflow in floating point and are printed as `< 1e-300`.
* "Effect size" labels follow Cohen-style conventions (Cramer's V, Spearman |rho|, rank-biserial r); they are rules of thumb, not laws.
* A segment with fewer than 100 orders is flagged as a **small sample**.
* All statements are **associations in observational data**. Nothing here shows that one factor *causes* another.
* Analyses that use information only available after the purchase are labelled **RETROSPECTIVE**.

## How to run
From the project root or from `notebooks/`, with the project virtual environment (the processed data must exist: `.venv/bin/python -m src.preprocessing`):
```bash
.venv/bin/jupyter lab notebooks/02_statistical_analysis.ipynb
```
The notebook finds the project root by itself, runs top-to-bottom in about a minute, and writes figures `21_*.png` ... `29_*.png` to `reports/figures/`.

## Contents
1. Data and test plan  &nbsp;·&nbsp; 2. RQ1 regions  &nbsp;·&nbsp; 3. RQ2 categories  &nbsp;·&nbsp; 4. RQ3 same vs cross state  &nbsp;·&nbsp; 5. RQ4 weight and freight
&nbsp;·&nbsp; 6. RQ5 time  &nbsp;·&nbsp; 7. RQ6 processing time  &nbsp;·&nbsp; 8. Robustness: sensitivity and sample size  &nbsp;·&nbsp; 9. Cross-checks, multiple testing, summary
&nbsp;·&nbsp; 10. Limitations and conclusions
"""

MD["s1_intro"] = """
## 1. Data and test plan

The analysis set is `data/processed/orders_delivered.parquet` (one row per delivered order, built by `src/preprocessing.py`).
The cell below loads it, defines a plain-integer target `late` (0 = on time or early, 1 = late), and shows how complete each variable used here is.
"""

MD["s1_plan"] = """
### Test plan and hypotheses

Each question gets a null hypothesis **H0** (no association / no difference) and an alternative **H1** (an association or difference exists; always two-sided).
The core procedures follow the standard test for each type of question. Where a choice was made after seeing the data (the spike-month definition, the median-shifted bootstrap interval, excluding the missing-category bucket from RQ2) this is stated where it is used.
The last column of the plan lists H0 in words.

* **Chi-square test of independence** (categorical x late). Assumption: every cell's *expected* count should be at least 5 and observations should be independent; rare levels are pooled into "Other" to satisfy the first.
  Effect size: **Cramer's V**, with the Bergsma bias correction and a parametric-bootstrap interval for the corrected V (shifted so that its median equals the point estimate, see section 2). **Adjusted standardized residuals** show *which* levels drive a significant result (|z| > 1.96 is notable, but see multiplicity).
* **Two-proportion comparison**: difference of late rates with a **Newcombe** score interval (built from Wilson intervals; better behaved than the textbook Wald interval), a pooled z-test, risk ratio and odds ratio.
* **Mantel-Haenszel** odds ratio: a state-adjusted ("stratified") comparison, used to ask whether a crude association survives holding the customer's state fixed.
* **Spearman rank correlation** for monotone relationships (robust to the heavy right skew of weight, freight and duration). Confidence interval by the Fisher z-transform with the Bonett-Wright standard error, cross-checked with a **bootstrap** interval; a **partial** rank correlation where confounding is plausible.
* **Mann-Whitney U / Kruskal-Wallis** for durations across groups (no normality assumption); effect sizes rank-biserial r and epsilon-squared.
* **Holm** step-down adjustment for the family of tests in this notebook.
"""

MD["s1_helpers"] = """
### Independent re-implementations

The next cell defines the helper functions used from here on. They deliberately use only `pandas`, `numpy` and `scipy` (no code from `src/statistical_analysis.py`)
so that the project module can be **confirmed rather than trusted** for the chi-square, correlation, Kruskal-Wallis, Mann-Whitney and same-state proportion results: there the notebook recomputes the number and asserts agreement.
Some steps (the state-vs-rest and category-vs-rest comparisons with their Holm adjustment, the spike-month difference, the correlation-strength labels) call the project module directly and are not independently recomputed here.
"""

# -----------------------------------------------------------------------------
MD["s2_intro"] = """
## 2. RQ1 — Do late-delivery rates differ between regions?

**H0:** the probability of a late delivery is the same in every customer state. **H1:** at least one state differs.

First the descriptive picture: late rate per customer state with its sample size and Wilson 95% interval.
States with fewer than 100 orders are kept in the table (nothing is dropped silently) but are flagged and drawn with hollow markers.
"""

MD["s2_assump"] = """
### Checking the chi-square assumptions

The Pearson chi-square test needs expected counts of at least 5 in every cell. The table shows how many cells violate this *before* pooling, how many levels had to be merged,
and that after pooling **no** cell is below 5. Pooling changes the question slightly (the pooled levels are tested as one group), so the pooled levels are listed explicitly.
Independence of observations is the other assumption: orders from the same seller or customer region are not strictly independent, which makes the p-values somewhat optimistic (see section 10).
"""

MD["s2_interp"] = """
### Interpretation of RQ1

**Result.** Late rates differ clearly between regions: chi-square test, V = «v_state» (bias-corrected «v_state_corr»), a **small** effect by Cohen's convention, and far beyond chance (p «p_state»). Among states with at least 100 orders
the rate runs from **«state_lo»** to **«state_hi»**, against an overall rate of «BASE». Rio de Janeiro (RJ, «rj», n = «rj_n») is the large state that stands out most, versus São Paulo (SP, «sp», n = «sp_n»): a difference of «rj_sp_diff» percentage points
(Newcombe 95% CI «rj_sp_ci»).
Of the «n_state_tested» states with n >= 100, «n_state_sig_raw» differ from the rest of the data at the unadjusted 5% level and **«n_state_sig_holm»** remain after Holm's correction for «n_state_tested» comparisons; the states that lose significance
after correction are «state_notholm_names» (their intervals are wide or their rates close to the overall rate). The adjusted residuals show that the chi-square statistic is driven mostly by SP, MG and PR (fewer late orders than expected) and by RJ, BA and the north-eastern states (more).

**How to read it.** Delivery *duration* differs substantially by state (Kruskal-Wallis epsilon-squared = «kw_eps2», a **medium** effect), while *lateness*, which compares the duration with the promise shown at checkout, differs less (V = «v_state», small; the two effect sizes are on different scales, so this is a qualitative contrast).
At state level, late rate and median duration are positively rank-correlated (rho = «rho_state_dur», only «n_state_tested» points), and the correlation of late rate with the share of same-state orders (rho = «rho_state_same», p «p_state_same») is not distinguishable from zero with so few points.
This is an ecological comparison of states, not of orders, so it can only be suggestive.

**Caution.** (1) The interval for each state is an *unadjusted* Wilson interval; with «n_state_tested» states, a few will exclude the overall rate by chance alone. (2) The state differences are mixed with **time**: the three spike months of section 6 hold a disproportionate share of late orders, and the regional V falls to «v_state_ns» when they are removed (section 8) but remains a small, clearly non-zero effect.
(3) This is association only: the data contain no distance, carrier, road or weather variables, so *why* a region is later cannot be read from them.
"""

# -----------------------------------------------------------------------------
MD["s3_intro"] = """
## 3. RQ2 — Does the late-delivery rate differ between product categories?

**H0:** the late probability is the same in every product category. **H1:** at least one category differs.

Each order is assigned the category of its highest-priced item. With many categories, small ones give noisy rates, so the ranking uses only categories with n >= 100 and
the chi-square test pools categories whose expected counts would fall below 5. The chart shows the ten highest and ten lowest rates with their intervals and sample sizes.

**Missing categories are not a category.** «n_unknown_cat» orders («pct_unknown_cat» of all, late rate «rate_unknown_cat») have no product category in the source data; the data layer labels them `unknown`. That label is a missing-data bucket, not a product, so it is **left out of every RQ2 test and ranking**
(chi-square, the n >= 100 ranking and the category-vs-rest comparisons, whose comparison group is all other orders of known category). RQ2 therefore rests on «n_cat_known» orders, and the category counts below (and in the dashboard's insights) exclude it. All other sections keep these orders.
"""

MD["s3_interp"] = """
### Interpretation of RQ2

**Result.** The association is statistically detectable (chi-square p «p_cat») but **negligible in size**: V = «v_cat», or «v_cat_corr» after bias correction (the plain V is inflated when a table has many cells; bootstrap 95% CI for the corrected V «v_cat_ci»). Among the «n_cat_reliable» categories with n >= 100 (of «n_cat_all» observed product categories, not counting the missing-category bucket) late rates range from «cat_rate_min» to «cat_rate_max» — but most of that range is noise-width:
only «n_cat_sig_raw» categories differ from all other orders at the unadjusted 5% level, and just **«n_cat_sig_holm»** survive Holm's correction for «n_cat_tested» comparisons: **«cat_holm_names»**. For the rest the data cannot distinguish the category's rate from the overall rate.

**Confounding.** Categories differ in who sells and ships them: the share of same-state orders ranges from «cat_same_min» to «cat_same_max» across categories, so a category comparison partly compares geographies. The category effect does not disappear inside the largest destination state (São Paulo customers only: V = «sp_cat_v», still negligible), but we cannot separate a product effect from seller, route and period effects with this data.

**Practical meaning.** Product category alone is a weak guide to delivery risk. In the dashboard, category rankings should be read together with n and the interval, and the categories flagged in section 3 are a *shortlist to investigate*, not a verdict.
"""

# -----------------------------------------------------------------------------
MD["s4_intro"] = """
## 4. RQ3 — Same-state versus cross-state orders

**H0:** the late probability is the same for same-state and cross-state orders (and delivery durations come from the same distribution). **H1:** they differ.

*Same-state* means the representative seller (the seller with the largest item value in the order) and the customer are in the same state. We estimate the size of the difference with an interval, test it three ways (Newcombe interval, pooled z-test, chi-square) and confirm
that they agree, then look at delivery duration.
"""

MD["s4_confound"] = """
### Is the same-state difference a geography artefact? Stratifying by customer state

A crude comparison of same-state and cross-state orders mixes very different customer regions: «same_sp_share» of all same-state orders are São Paulo to São Paulo, and São Paulo is also the state with the lowest late rate in section 2.
To ask whether the difference survives comparing like with like, the comparison is repeated **inside each customer state** and combined with the **Mantel-Haenszel** odds ratio (Robins-Breslow-Greenland interval). Only «n_strata_both» states have at least 100 orders in both groups, so the stratum-level results are shown for those.
"""

MD["s4_interp"] = """
### Interpretation of RQ3

**Result — the crude association is real but mostly not what it looks like.**
* Same-state orders are late **«rate_same»** of the time (n = «n_same»), cross-state orders **«rate_cross»** (n = «n_cross»): a difference of **«diff_pp» percentage points** (Newcombe 95% CI «diff_ci»; crude odds ratio «or_crude»). Cramer's V is only «v_same» — a **negligible** effect size despite p «p_same».
* Durations differ much more: the median is **«median_same» days** for same-state versus **«median_cross» days** for cross-state orders (difference of the medians «med_diff» days; bootstrap interval in the output above), rank-biserial r = «mw_rb» (**large**). Orders shipped within a state arrive much sooner. That is expected, and it is not the same as being late: the estimated delivery time for cross-state orders is also much longer (mean «est_cross» days against «est_same» days for same-state orders).
* **Adjusting for customer state shrinks the lateness association a lot**: the Mantel-Haenszel odds ratio is **«or_mh»** (95% CI «or_mh_ci») against a crude «or_crude». It stays below 1 — same-state orders are somewhat less often late even within a state — but in the largest stratum, São Paulo customers, the direction **reverses**: «sp_same_rate» late for same-state against «sp_cross_rate» for cross-state orders.
  The odds ratio is below 1 in «n_strata_or_below1» of the «n_strata_both» strata with enough data and above 1 in «n_strata_or_above1», so a single pooled number summarises heterogeneous behaviour; the pooled estimate should not be treated as a constant "same-state effect".
* Removing the three spike months of section 6 cuts the same-state V from «v_same» to «v_same_ns»: part of the crude difference is the *timing* of orders, not geography alone (section 8).
* **The pooled lateness gap is not stable over time, and it reverses at the end of the data.** The table and chart above split the same comparison by the model's chronological windows: same-state versus cross-state late rate is «stab_train_same» versus «stab_train_cross» in the training window (gap «stab_train_gap» pp), «stab_val_same» versus «stab_val_cross» in the validation window («stab_val_gap» pp), but **«stab_test_same» versus «stab_test_cross» in the test window (July-August 2018; gap «stab_test_gap» pp, Newcombe 95% CI «stab_test_ci»)**: the same-state orders are *later* there. Month by month from 2017-01, same-state orders have the higher late rate in «stab_n_months_same_higher» of «stab_n_months» months («stab_months_same_higher»). This is the same non-stationarity as in RQ5, and it agrees with the model card in the app (the gap "changes sign between windows"). The pooled 2016-2018 gap is an average over regimes, not a constant property of same-state shipping.

**Take-away.** Same-state shipping is clearly faster. Over the full 2016-2018 period same-state orders were also somewhat less often late, but that headline gap is largely explained by *where* the customers are (and when they ordered), it varies a lot between states, and it is **not stable in time**: it reverses in the most recent months of the data, so it should not be read as a lasting "same-state advantage". Anyone reading the dashboard's same-state card should also see the duration gap, the effect size, this confounding and the time instability.
"""

# -----------------------------------------------------------------------------
MD["s5_intro"] = """
## 5. RQ4 — How are weight and freight related to delivery duration?

**H0:** the Spearman rank correlation between the variable and delivery duration is zero. **H1:** it is not.

Weight, freight, and duration are strongly right-skewed (a few very heavy, expensive or slow orders), so the rank-based Spearman correlation is used instead of Pearson's. For each correlation the table compares:
scipy's `spearmanr`, the project module's value, Pearson's correlation of the ranks (identical by definition), the Fisher-z interval, and a **bootstrap** interval from resampling orders. Then we look at the binary outcome, a *partial* correlation, and the pattern inside same-state and cross-state orders.
`estimated_delivery_days` (the promise shown at checkout) is added as a reference because it is known at purchase time, and `total_weight_g ~ total_freight` as a sanity check of how much freight reflects weight.
"""

MD["s5_interp"] = """
### Interpretation of RQ4

* **Weight** is only **negligibly** related to delivery duration: rho = «rho_weight_dur» (a visible but tiny upward drift in the binned plot). Heavier parcels are not noticeably slower in this data.
* **Freight cost** is **moderately** related to duration: rho = «rho_freight_dur» (95% CI «rho_freight_ci»). The Fisher-z and bootstrap intervals agree closely and are very narrow, because n is large. Freight is only moderately related to weight (rho = «rho_weight_freight»), and the partial correlation of freight with duration *controlling for weight* is «partial_fw», essentially unchanged. So the freight-duration link is not a weight effect.
  The data contain no distance, so "freight is a proxy for how far the parcel travels" is a plausible reading, not a tested one. The pattern is consistent with it: inside same-state orders freight and duration are more correlated (rho = «rho_fd_same») than inside cross-state orders (rho = «rho_fd_cross»), i.e. part of the pooled association comes from mixing short and long routes.
* **Freight ratio** (freight as a share of item price) has a weak association with duration (rho = «rho_fratio_dur»), and **no detectable** association with being late (rank correlation about zero, p «p_fratio_late»).
* The **promised** delivery time tracks actual duration strongly (rho = «rho_promise_dur»): Olist's estimate encodes the expected route length. Yet the promise is *slightly negatively* related to lateness (rho = «rho_promise_late», negligible), which would be consistent with a longer promise leaving more slack.
* With the binary outcome, all four rank correlations with `is_late` are negligible (largest |rho| = «max_abs_rho_late»); where they are statistically significant, that is a matter of n, not of strength.
  **But a rank correlation with a rare 0/1 outcome understates a stepwise pattern, and the binned chart shows one:** across the «fb_n_bins» freight bins (Fig. 24, bottom right) the late rate rises steadily from **«fb_first_rate»** in the cheapest bin (freight «fb_first_label», n = «fb_first_n») to **«fb_last_rate»** in the dearest (freight «fb_last_label», n = «fb_last_n»), against «BASE» overall, while the median duration climbs from «fb_first_dur» to «fb_last_dur» days over the same bins (rho of freight with `is_late` = «rho_freight_late»). This is descriptive and confounded: freight is probably a proxy for distance, and region and period move with it (see RQ1 and RQ5); it is not an effect of freight itself and it does not mean that cheaper shipping prevents lateness.

**Meaning.** Weight and freight describe how long a delivery takes far better than whether it is late, although orders in the highest freight bin are late about «fb_ratio» times as often as those in the lowest (a descriptive, confounded pattern). Duration is mostly geography; lateness is the *excess over the promise*, and the promise already absorbs most of the geography. These are descriptive associations: they do not say that raising or lowering freight changes delivery time.
"""

# -----------------------------------------------------------------------------
MD["s6_intro"] = """
## 6. RQ5 — Seasonality and other time patterns

**H0:** the late probability is the same in every month (and every weekday). **H1:** it differs.

The data cover only 2016-09 to 2018-08 and the monthly volume grows strongly over the period (see the order-count bars in the monthly chart), so two traps have to be avoided: (i) tiny early months are excluded by the n >= 100 rule (the two single-order months 2016-09 and 2016-12 drop out; 2016-10, with n = 265, is kept, and 2016-11 has no orders at all, so the monthly chart shows gaps there), and (ii) "month of year" pools *different years*, mixing seasonality with year and with one-off events.
We therefore report the monthly series with intervals first, look at the month x year heatmap, and only then run the chi-square tests. Spike months are defined from the series (Wilson interval entirely above 1.5 times the overall late rate); this is a **post-hoc, descriptive** definition, so the comparison with other months carries no confirmatory p-value.
"""

MD["s6_interp"] = """
### Interpretation of RQ5

**Result.** Lateness is strongly *time-varying*, but this is better described as **episodes** than as seasonality.
* From 2017-01 to 2018-08 the monthly late rate ranges from **«m_lo»** to **«m_hi»**. Three months stand out, with intervals entirely above 1.5 times the overall rate: **«spike_list»**. Together they hold «spike_share_orders» of all orders but **«spike_share_late»** of all late orders (late rate «spike_rate», n = «spike_n», versus «other_rate», n = «other_n», in the other months from 2017-01).
  The data contain no information on *why* these months are different (demand peaks, carrier capacity, weather, or a change in how the delivery estimate was set), so no explanation can be tested here.
* The chi-square test on **month of the year** is significant with V = «v_month» (**small**), and on **year-month** V = «v_ym» (**small**). The month-of-year result must not be read as "March is a bad month": the heatmap shows that March appears twice, in 2017 (**«r_2017_03»**) and in 2018 (**«r_2018_03»**, the highest of all), and February likewise («r_2017_02» in 2017, «r_2018_02» in 2018). January to August appear in both 2017 and 2018 (October also appears in 2016 and 2017, but 2016 holds only that one month with meaningful volume), and the same calendar month can differ enormously between them; section 8 shows that the pooled month-of-year V falls from «v_month» to «v_month_ns» without the spike months.
  The honest conclusion is **non-stationarity**: rates move between regimes, and a model trained on one period may not transfer to the next.
* **Weekday** differences are statistically detectable but **negligible**: V = «v_dow». This is the clearest example in the notebook of significance without practical importance.

**Practical meaning.** Dashboards should show monthly lateness with n and intervals, flag the episodes, and avoid seasonal claims from only two years of data.
"""

# -----------------------------------------------------------------------------
MD["s7_intro"] = """
## 7. RQ6 — Seller processing time (retrospective)

**H0:** processing time (purchase to carrier hand-over) is unrelated to lateness. **H1:** it is related.

> **RETROSPECTIVE.** The hand-over date is only known after the seller ships, so this analysis explains what happened; it **cannot** be used to predict a delivery at the moment of purchase, and it is deliberately excluded from the model's inputs.

Besides the rate by processing bucket we need to be careful about a **mechanical relationship**: `delivery_duration_days` *contains* processing time (purchase to hand-over, then hand-over to customer), so a positive correlation with duration is partly built in.
Lateness is the better outcome here because it compares the whole journey with a promise fixed at purchase. Orders with a missing or impossible carrier timestamp are left out of this section only.
"""

MD["s7_interp"] = """
### Interpretation of RQ6

* Late rates climb steadily with processing time: **«proc_rate_first»** for orders handed over within a day (n = «proc_n_first») up to **«proc_rate_last»** when hand-over took more than 14 days (n = «proc_n_last»).
* Late orders had a median processing time of «proc_median_late» days versus «proc_median_ok» days for on-time orders (Mann-Whitney, rank-biserial r = «proc_rb», at the small/medium boundary). The rank correlation with lateness is **weak** (rho = «rho_proc_late»), because most orders have short processing times and the relationship is concentrated in the long tail.
* Processing time with delivery duration: rho = «rho_proc_dur» (moderate), as expected given the mechanical overlap. For comparison, the *transit* leg (carrier to customer) has the stronger rank correlation with lateness (rho = «rho_c2c_late»), and processing time is almost unrelated to transit time (rho = «rho_proc_c2c»): a slow hand-over does not slow down the transit, but both add up to a late arrival.

**Meaning.** Slow seller processing is **associated with** a higher late rate, most visibly beyond a week. Whether a lower processing time would reduce lateness cannot be answered from this data (sellers who are slow to ship may also differ in region, product or load). Because it is retrospective, processing time is useful for diagnosing seller performance after the fact, not for predicting at checkout.
"""

# -----------------------------------------------------------------------------
MD["s8_intro"] = """
## 8. Robustness: sensitivity to the spike months, and why n matters

Two checks that guard against over-reading the p-values.

**(a) Sensitivity.** Repeat the chi-square battery without the spike months from section 6 (n = «n_ns» orders, overall late rate «base_ns»). An association that is mostly produced by those episodes should shrink sharply.
**(b) Effect size versus sample size.** Draw random subsamples of the data and test same-state x late each time. If significance is mostly a matter of n, the *share of subsamples reaching p < 0.05* should rise with n while the *effect size* stays put.

The sensitivity result: Cramer's V for the customer state falls from «v_state» to «v_state_ns», for same-state from «v_same» to «v_same_ns», and for the month of the year from «v_month» to «v_month_ns»;
the category effect is unchanged («v_cat» to «v_cat_ns»). So regional, same-state and seasonal patterns depend partly on the spike episodes, whereas the (small) category pattern does not.
"""

MD["s9_intro"] = """
## 9. Cross-checks, multiple testing and summary

**Cross-check.** All chi-square, Spearman and Kruskal-Wallis results in the project's `run_all_tests` battery are recomputed directly with scipy; the maximum absolute difference is printed.

**Multiple testing.** The notebook runs a family of related hypothesis tests, listed below; each is Holm-adjusted for the number run. With a few exceptions the p-values are so small that adjustment changes no verdict, which is a statement about the sample size,
not about the importance of the effects. The more informative columns are the **effect sizes**, drawn in the figure that follows. (The «n_state_tested» state-vs-rest and «n_cat_tested» category-vs-rest comparisons in sections 2-3 were adjusted separately within their own families, where adjustment *did* matter.)
"""

MD["s9_interp"] = """
### What the cross-checks and the effect-size figure show

* Wherever the notebook recomputed a statistic with plain scipy (chi-square, Spearman, Kruskal-Wallis, Mann-Whitney, same-state proportion tests), it agrees with the project module to numerical precision; the state-vs-rest, category-vs-rest and spike-month comparisons were not independently recomputed here.
* **«n_sig_raw» of «n_tests_family»** tests are significant before adjustment and **«n_sig_holm»** after Holm adjustment; the exception is **«p_holm_max_nonsig_name»** (not significant either way). Nearly everything is "significant" because n is about 96,000 — that cannot distinguish important from unimportant effects.
* The subsample experiment makes the point: with n = 200 only «sub_share_200» of random subsamples detect the same-state difference, with n = 2,000 «sub_share_2000», with n = 20,000 all of them (share = «sub_share_20000»), while the median Cramer's V stays near «sub_v_full» throughout.
* Judged by **effect size**, the associations with lateness are small at best: customer state, calendar month and processing time reach "small" (V between 0.1 and 0.3); product category, weekday, and the rank correlations of weight, freight, freight ratio and the promise with lateness are negligible. Larger correlations appear only with delivery *duration* (the promise, freight and, mechanically, processing time).
"""

# -----------------------------------------------------------------------------
MD["s10_limits"] = """
## 10. Limitations

1. **Huge n makes trivial effects significant.** At n ~ 96,000 almost any difference gives p < 0.05. We therefore read effect sizes and intervals, not p-values; weekday, product category and most weight/freight correlations are significant but negligible.
2. **Confounding.** Region, distance, seller location, product mix and period are entangled. The same-state analysis showed how a crude association changes size after adjusting for customer state (and even reverses inside São Paulo). Adjusting for one variable does not remove all confounding; the data contain no distance, carrier or capacity variable.
3. **Observational data, no causal claims.** Every statement is an association. "Late rates are higher in region X" does not say that the region *causes* lateness; "slow processing is associated with lateness" does not say that speeding it up would help.
4. **Delivered orders only (selection).** Cancelled, unavailable and never-delivered orders are excluded because they have no delivery date; if undelivered orders were disproportionately problematic, the late rates here understate the true failure rate.
5. **Lateness is measured against Olist's own estimate.** An order can be "on time" and slow, or "late" and quick. Changes in how the estimate was set over time (visible as shifts in the late rate) are therefore a possible alternative explanation for time patterns.
6. **Non-stationarity.** Late rates move between regimes (2017-11, 2018-02/03 are far above the rest; 2018-06 far below). Results from pooled data describe an average over very different periods, and two years are far too few to estimate seasonality. This applies to the pooled associations of the other questions too: for example, the same-state advantage of RQ3 (pooled gap «diff_pp» pp) is negative in the training and validation windows and **positive in the test window** (July-August 2018), as the model card also reports; the pooled same-state gap is not a stable relationship.
7. **Independence assumptions.** Chi-square, Wilson and Fisher-z intervals assume independent orders; orders from the same seller or region and period are not independent, so intervals are somewhat too narrow. Nothing here clusters the errors (no seller or customer identifier is used by design).
8. **Multiplicity.** Holm adjustment was applied within each family, but the whole exploration (dozens of looks at the data) is still a forking-paths exercise: treat every single "significant" finding as a hypothesis for confirmation on new data rather than as established.
9. **Pooling and binning choices.** Rare levels were pooled, categories below n = 100 were ranked out, orders with a missing category («n_unknown_cat») were left out of RQ2, month-of-year pools different years, and weight/freight/processing bins are conventions; other choices would give somewhat different numbers.
10. **Historical data.** These are 2016-2018 orders from one Brazilian marketplace. They do not describe current logistics performance and need not generalise to other platforms or periods.
11. **Retrospective variable.** Processing time (RQ6) is known only after shipping and is excluded from any purchase-time prediction.
"""

MD["s10_conclusion"] = """
## 11. Conclusions and what they mean for the dashboard

| Question | Evidence (effect size, n) | Plain-language conclusion |
|---|---|---|
| **RQ1 Region** | V = «v_state» (small); «state_lo» to «state_hi»; «n_state_sig_holm» of «n_state_tested» states differ after Holm | Region clearly matters, though modestly for *lateness*; RJ, BA and the north-east are above, SP, MG and PR below the overall rate. Show n and intervals; flag small states. |
| **RQ2 Category** | V = «v_cat» (negligible), «n_cat_reliable» categories with n >= 100 (missing-category bucket excluded); only «cat_holm_names» survive Holm | Category is a weak signal; do not rank categories without n and intervals. |
| **RQ3 Same vs cross state** | late «rate_same» vs «rate_cross»; V = «v_same» (negligible); MH OR «or_mh»; duration median «median_same» vs «median_cross» days | Same-state orders arrive much sooner. Their lower pooled late rate is largely explained by customer state, reverses in SP, and is **not stable over time**: it reverses in the Jul-Aug 2018 test window (same-state «stab_test_same» vs cross-state «stab_test_cross»), in line with the model card. |
| **RQ4 Weight / freight** | rho(weight) = «rho_weight_dur» (negligible); rho(freight) = «rho_freight_dur» (moderate) | Freight, probably as a proxy for distance, relates to duration; weight barely does. Rank correlations with lateness are negligible, but late rate rises from «fb_first_rate» to «fb_last_rate» across freight bins (descriptive, confounded by region and period). |
| **RQ5 Time** | V(year-month) = «v_ym» (small); «spike_list» hold «spike_share_late» of late orders | Lateness comes in episodes; avoid seasonal claims from two years, and expect drift. |
| **RQ6 Processing (retrospective)** | late «proc_rate_first» (<= 1 d) to «proc_rate_last» (> 14 d); rho = «rho_proc_late» (weak) | Slow processing is associated with lateness, mainly in the long tail; not usable at purchase time. |

**Overall.** Purchase-time and descriptive variables show real but **modest** associations with lateness, much weaker than their associations with delivery *duration*. This is the honest statistical context for the prediction page of the app: purchase-time information separates risk groups only weakly, so
expectations should be modest; the app's model card reports the measured performance on a later test window. **Historical 2016-2018 patterns are not current logistics performance, and none of these results shows causation.**

*Figures written by this notebook:* `reports/figures/21_regional_late_rate_and_residuals.png` ... `29_effect_sizes_and_sample_size.png`.
"""
