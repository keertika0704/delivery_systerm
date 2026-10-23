"""Narrative (Markdown) cells for notebooks/03_model_training.ipynb.

Every number quoted here was read from the executed notebook outputs.  The build script (_build_nb03.py) checks that each value
the notebook registers in its QUOTED table appears in this text, so the prose cannot silently drift from the computed results.
"""

MD: dict[str, str] = {}

MD["title"] = """
# DeliveryIQ — Notebook 03: Model Training, Evaluation and Honest Conclusions

**Project:** DeliveryIQ — Intelligent Shipment Analytics & Delay Prediction (Fundamentals of Data Science)
**Notebook:** `notebooks/03_model_training.ipynb`  |  **Data:** Olist Brazilian E-Commerce Public Dataset (2016–2018, historical)

## What this notebook does
It rebuilds, step by step and with the *same code the app uses* (`src/train_model.py`, `src/prediction.py`), the delay-risk model that powers
the **Delay Prediction** page, evaluates it honestly, and explains what it can and cannot do.

1. Frame the problem and define the target (`is_late`) and its class imbalance.
2. Choose predictors: a **purchase-time allow-list** versus a **leakage list**, with a leakage experiment that shows why the line matters.
3. Split the data **chronologically** (train → validation → test) and show why the late rate is not stationary.
4. Show the preprocessing pipeline (imputation, scaling, one-hot) that is shared by training and the app.
5. Train a dummy baseline, **logistic regression**, **random forest** and **histogram gradient boosting**, and reconcile every number with `models/model_metrics.json`.
6. Evaluate with the metrics that suit a rare outcome (PR-AUC, recall/precision, ROC-AUC, Brier, calibration) and explain why accuracy misleads.
7. Choose the decision threshold and the Low / Moderate / High risk bands on **validation**, then check them on **test**.
8. Permutation importance and coefficients: predictive associations, **not causes**.
9. Select the final model, diagnose why test performance is much weaker than validation, and state the conclusion plainly.
10. Load the saved bundle and score three real test-window orders with `prediction.predict_risk`.

## Headline, stated up front
The selected model is a **logistic regression** with a **modest but real** ranking signal. On the untouched test window it reaches ROC-AUC 0.617 and PR-AUC 0.073
against a no-skill PR-AUC of 0.048. That is **far weaker than on the validation window** (ROC-AUC 0.781, PR-AUC 0.140), where the model was tuned. The late rate moved
between windows (train 8.0%, validation 4.2%, test 4.8%), the Moderate and High risk bands are barely separated on test, and the probabilities are not better calibrated
than quoting the base rate. **The model is a rough triage aid, not an accurate or reliable predictor**, and it says nothing about causes or about today's logistics performance.

## How to run
From the project root, with the project virtual environment (the processed data and the saved model must exist; if not, run the two commands in the comments):

```bash
# .venv/bin/python -m src.preprocessing     # only if data/processed/*.parquet are missing
# .venv/bin/python -m src.train_model       # only if models/ is missing (NOT needed to run this notebook's training cells)
.venv/bin/jupyter lab notebooks/03_model_training.ipynb
```
The notebook finds the project root by itself. It trains the models itself (about 1.5 minutes in total; the slow random-forest / boosting grids are not re-run, see section 5)
and saves its figures as PNG files `30_…` to `39_…` in `reports/figures/`.

**Conventions.** Rates always come with their sample size `n`. "Associated with" never means "caused by". Validation numbers are labelled *tuned / in-sample* wherever the
validation window influenced a choice; only the **test window** is an out-of-sample estimate. No fitting, tuning, calibration, threshold, band or model-choice step uses the test window; it is used only for evaluation and for clearly labelled diagnostics (the leakage and random-split demonstrations, bootstrap intervals, permutation importance and the drift analysis), none of which feeds back into a decision.
"""

# =============================================================================
MD["s1_intro"] = """
## 1. Problem framing and target definition

**Question.** *At the moment a customer places an order, how likely is it that the order will arrive later than the delivery date promised at checkout?*

**Use case.** A seller-support or operations team could review the highest-risk new orders first. This is a **triage / ranking** task: a useful model puts truly late
orders higher in the list than on-time ones. Because the outcome is rare, "how many late orders are in the top of the list" matters more than overall accuracy.

**Target.** `is_late = 1` when the order was delivered **after the estimated delivery calendar date**, else 0.

* The rule is **calendar-date based**: `delivery_delay_days = delivered date − estimated date` (both truncated to the day) and `is_late = delay > 0`. Arriving at any time on the promised day is on time.
* Only **delivered** orders with all required timestamps are labelled (the analysis set). Cancelled, unavailable and still-in-transit orders have no label; they are never treated as on time.
* A customer's *promised* date is the Olist estimate shown at checkout, so "late" means "later than promised", not "slow in absolute terms".
"""

MD["s1_interp"] = """
**Interpretation.**

* The analysis set holds **96,470** delivered orders; **6,534** are late, a prevalence of **6.77%**, i.e. **13.8** on-time orders for every late one.
* The calendar-date rule matters: comparing full timestamps instead would label **8.11%** of orders late. The difference is exactly **1,292** orders that arrived on the promised
  day but after its midnight timestamp. Counting those as late would punish deliveries that kept the promise.
* The delay histogram shows why the target is imbalanced: most orders arrive days or weeks *before* the promised date, and lateness is a thin right tail.
* **Consequence for evaluation.** A model that never predicts "late" is right for about 93% of these orders. Accuracy therefore cannot be the headline metric; the no-skill
  value of PR-AUC equals the prevalence, and every later comparison is made against that reference (see sections 6 and 7).
"""

# =============================================================================
MD["s2_intro"] = """
## 2. Predictive feature selection and the leakage list

A delay-risk model must be usable **at purchase time**. `src/config.py` therefore partitions every processed column into exactly one of four groups, and the training script
refuses to run if a model input is outside the allow-list (this is also checked by unit tests and recorded in the model's `leakage_audit`):

1. **Purchase-time allow-list** (`PURCHASE_TIME_FEATURES`): order size, price, freight, weight and volume aggregates, the checkout promise `estimated_delivery_days`, purchase month / weekday / quarter,
   customer state, seller state, representative product category and `same_state`.
2. **Known at purchase but left out**: high-cardinality cities and `state_pair`, secondary counts and data-quality flags.
3. **Leakage / outcome** (`LEAKAGE_COLUMNS`): final status, post-purchase timestamps, durations and delays, processing time, review score, eligibility bookkeeping — they either *are* the outcome or only exist after it.
4. **Non-feature**: keys, raw timestamps used for the split, and time-drift proxies (year, date, year-month).
"""

MD["s2_leak_intro"] = """
### Leakage experiment — a demonstration, not a model

To make the danger concrete, the next cell trains the **same** logistic regression on the **same** training window but adds outcome-side columns to the allow-list.
These columns do not exist when a customer clicks "buy", so **none of these variants can ever be deployed**. They are shown only to demonstrate how leakage inflates scores and
why the honest score must never be compared with them.
"""

MD["s2_interp"] = """
**Interpretation.**

* The honest, purchase-time-only version reaches a test ROC-AUC of **0.619** (this is an uncalibrated logistic regression with `C = 1`; the deployed model adds a calibrator and is evaluated in section 6).
* Adding **`processing_days`** (how long the seller took to hand the parcel to the carrier — known only after the fact) lifts the test ROC-AUC to **0.807** and PR-AUC to **0.186**.
  Adding **`carrier_to_customer_days`** gives **0.894**, and adding **`delivery_duration_days`** gives **1.000** (PR-AUC **0.994**; validation ROC-AUC 1.000), because `is_late` is just
  `delivery_duration_days` compared with the promise, and the promise is already a feature. `delivery_delay_days` would be perfect by construction.
* A near-perfect score on this task is therefore a **red flag for leakage, not a success**. The honest model sits far lower, and the gap between the bars is the price of not knowing the future.
* The leakage audit stored with the model (all checks PASS, including "no feature appears in `config.LEAKAGE_COLUMNS`" and "imputer medians equal the train-window medians") is re-checked at the end of this notebook.
* **Feature selection details.** Starting from the allow-list, the training script drops one of every pair of numeric features with |Spearman ρ| ≥ 0.9 on the training window (`purchase_quarter` duplicates `purchase_month`, ρ = 0.96),
  and then tests calendar month: removing it did not lower the mean validation PR-AUC across two model families, so `purchase_month` is dropped (a month effect cannot be separated from one-off events when training covers about 1.7 years).
  The final model has **16** inputs: 12 numeric and 4 categorical (table above). Note that the month decision itself used the validation window.
"""

# =============================================================================
MD["s3_intro"] = """
## 3. Chronological splitting strategy

A delay model will always be used on **future** orders, so it is evaluated on a future window. Orders are split by `order_purchase_timestamp`:

| window | rule (start inclusive, end exclusive) | role |
|---|---|---|
| train | 2017-01-01 → 2018-04-01 | fit preprocessing and models |
| validation | 2018-04-01 → 2018-07-01 | tune hyper-parameters, choose calibration, threshold, risk bands and the final model |
| test | 2018-07-01 → end of data | evaluation and labelled diagnostics only; never used to fit, tune, calibrate, set thresholds or bands, or choose the model |

Orders before 2017-01-01 (marketplace almost empty) are excluded. No order is in two windows, and the whole preprocessing pipeline is fitted on the training window only.
The final model is **not** refitted on train + validation, so validation-based calibration, thresholds and bands stay honest (the cost: the weeks just before the test window are not used for fitting).
"""

MD["s3_interp"] = """
**Interpretation — windows, selection bias and non-stationarity.**

* **Counts.** Train **64,053** orders (**5,110** late, **8.0%**), validation **19,643** (**820** late, **4.2%**), test **12,507** (**601** late, **4.8%**). **267** earlier orders are excluded as too sparse.
  Each window has hundreds of late orders, enough for rates with intervals but not enough for fine segment-level conclusions.
* **The late rate is not stationary.** Inside the training window it ranges from about 3% (mid-2017) to **12.4% (Nov 2017)**, **14.1% (Feb 2018)** and **19.0% (Mar 2018)**, then falls back to 4.5% in April 2018.
  Those episodes are not explained by order attributes available at purchase, so any purchase-time model can at best capture a rough average level plus structural differences (promise length, geography).
  The training window deliberately contains the disrupted months so the model has seen both regimes, which also means its *average* training late rate (8.0%) is much higher than in the later windows.
* **Selection bias / right-censoring.** Labels exist only for delivered orders. In the test window **145** of 12,824 orders (**1.13%**) were still in an open status at the end of the data and cannot be labelled; if those are
  disproportionately late, the late rate of the latest months is slightly understated. Cancelled and unavailable orders are also unlabelled, so every rate describes *delivered* orders only.
* **Covariate / concept drift.** The median promise was **24 days** in train, **25 days** in validation and only **17 days** in test. The gap between same-state and other-state late rates is **-5.1** percentage points in train and
  **-3.1** in validation but **+4.4** in test, so the association with `same_state` **changes sign**. In the test window **10.1%** of orders (**1,263**) have a feature outside the training range (almost all: a shorter promise than ever seen in training),
  and those orders are late **16.9%** of the time versus **3.5%** for the rest. The model has to extrapolate on exactly the orders that are most often late.
* **Why not a random split?** Shuffling the same 2017-01 … 2018-08 orders into 80% / 20% gives a ROC-AUC of **0.712** and a PR-AUC lift of **2.4**× over no skill, against **0.619** and **1.7**× when the model must predict a later window.
  Random splitting lets the model see neighbouring days of the same disruption episodes, which flatters it. This is a demonstration of optimism, not an evaluation we rely on.
"""

# =============================================================================
MD["s4_intro"] = """
## 4. The preprocessing pipeline

`src.train_model.build_pipeline` is the single source of truth for the model structure: the notebook, the training script and the saved bundle all use it, so the app scores new orders with exactly the transformations used in training.
Everything is *inside* the scikit-learn `Pipeline`, so nothing leaks from validation or test data into the fitted preprocessing.
"""

MD["s4_interp"] = """
**Interpretation.**

* 16 input columns become **114 design-matrix columns**: the 12 numeric columns, 5 "was missing" indicator columns (only the product weight and dimension columns ever have missing values, 16 training orders), and 97 one-hot columns.
* The imputer medians are the training medians (checked above); the test-window medians differ (for example the median weight is lower), which would be a form of leakage if they had been used.
* Rare categories (fewer than 50 training orders) are pooled into an "infrequent" level per feature: this affects only a few hundred training orders and keeps the one-hot design stable. Unseen categories at prediction time become all-zero rows instead of errors.
* Scaling is applied only for logistic regression; the tree models are scale-free.
"""

# =============================================================================
MD["s5_intro"] = """
## 5. Baseline and candidate models

| model | role | tuning (validation PR-AUC, small grids) |
|---|---|---|
| Dummy (class prior) | the **no-skill baseline**: always predicts the training late rate | none |
| Logistic regression | simple, interpretable, strong on sparse one-hot data | `C` × `class_weight` (10 settings) |
| Random forest (300 trees) | non-linear, handles interactions | `min_samples_leaf` × `max_features` (6 settings) |
| Histogram gradient boosting | strong non-linear baseline | `max_iter` × `max_leaf_nodes` (4 settings) |

After tuning, each model is **probability-calibrated** (sigmoid or isotonic, fitted on the earlier half of validation and kept only if it improves Brier score and log-loss on the later half; the final calibrator is refitted on all of validation).
Exact-zero isotonic scores are floored at a small positive probability because no order is certain to be on time.

**Runtime note.** The logistic-regression grid is cheap and is **re-run in full** here. The random-forest and boosting grids dominate the training time, so for those two the notebook refits only the **saved best setting**
and shows the grid results stored in `models/model_metrics.json`. The notebook then reconciles every model, window and metric with that file.
"""

MD["s5_interp"] = """
**Interpretation.**

* The re-run logistic-regression grid reproduces the saved grid exactly and picks the same setting (`C = 1.0`, no class weighting). Class weighting (`balanced`) was not better than no weighting at any `C` of 0.03 or more (it was marginally higher only at `C` = 0.01).
* **Reconciliation with `models/model_metrics.json`:** the dummy, logistic regression and histogram gradient boosting match the saved file to floating-point precision (deviation 0).
  The **random forest deviates by about 1e-04** (largest difference over eight metrics). This is not a code difference: the forest sums tree outputs across threads, so each probability varies in its last bits (about 1e-16),
  and the isotonic calibrator, which pools tied scores, amplifies this to about 1e-4. Re-running the *same* calibration on the *same* forest three times gives validation PR-AUCs that differ among themselves by about 2e-4 (cell above).
  The selected model, logistic regression, is deterministic and matches exactly.
* Calibration was kept for all three tuned models (isotonic for logistic regression and the forest, sigmoid for boosting) because it improved the Brier score on the later half of validation (for logistic regression 0.0345 → 0.0333).
  Calibration was judged on validation, not on test: later we will see that it does not protect against a base-rate shift.
"""

# =============================================================================
MD["s6_intro"] = """
## 6. Evaluation

### Which metrics matter for delay-risk prediction, and why accuracy misleads

* **Accuracy** counts correct labels. With about 5% late orders, predicting "on time" for everything scores about 95% and finds no late order at all.
* **Recall** (share of late orders that are flagged) and **precision** (share of flagged orders that really are late) describe the trade-off in using the model for triage. They depend on the threshold.
* **PR-AUC (average precision)** summarises precision across all thresholds and focuses on the rare class; its **no-skill value equals the prevalence**, so lift over that baseline is easy to read. It is the primary selection metric here.
* **ROC-AUC** measures ranking quality independent of prevalence (0.5 = no skill) but can look comfortable even when precision is poor.
* **Brier score / log-loss and the calibration curve** check whether a predicted "10%" really means about 10%. A model can rank well and still be badly calibrated after the base rate shifts.
* **Lift in the top 10%** compares the late rate among the 10% highest scores with the overall late rate. The scores are a step function with few distinct levels, so tied scores at the 10% cut are split in expectation (as if broken at random); a plain `argsort` would break the ties by row order and give a somewhat different number.

All thresholds below were fixed on the **validation** window (F1-optimal cut for each model) and then applied unchanged to test.
"""

MD["s6_interp"] = """
**Interpretation (test window unless stated; all figures from the tables and plots above).**

* **Accuracy misleads.** The dummy that never flags an order scores **95.2%** accuracy on test and finds **0 of 601** late orders; logistic regression scores only **79.9%** accuracy yet finds **155** of the **601** late orders (recall **25.8%**).
  The accuracy ranking is the reverse of the usefulness ranking, which is why accuracy is shown but never used to choose a model.
* **Logistic regression, test:** ROC-AUC **0.617** (95% bootstrap interval 0.599 - 0.637), PR-AUC **0.073** (0.064 - 0.083) against a no-skill PR-AUC of **0.048**; precision **7.0%**, recall **25.8%**, F1 **0.110**, with **17.8%** of orders flagged.
  The top-10% lift is only **1.7**. This is a **modest but real** signal: the interval excludes no skill, but the model finds late orders only somewhat better than chance.
* **Validation, tuned / in-sample:** accuracy **91.7%**, Brier **0.0376** (dummy 0.0414), ROC-AUC **0.781**, PR-AUC **0.140** (no skill 0.042), precision **19.3%**, recall **31.0%**, F1 **0.238**, **6.7%** flagged, top-10% lift **3.9**. These look much better and are **not** an honest estimate: every choice was made on this window.
* **Other models.** Random forest: test ROC-AUC **0.595**, PR-AUC **0.059** (validation 0.742 / 0.117, saved file). Histogram gradient boosting: test ROC-AUC **0.507** (indistinguishable from no skill), PR-AUC **0.050** (validation 0.745 / 0.138).
  On validation, gradient boosting matches logistic regression on PR-AUC (0.138 vs 0.140) but is lower on ROC-AUC (0.745 vs 0.781); the random forest is lower on both (0.117 / 0.742). Neither flexible model transfers better to the later window; they transfer **worse**.
  One possible reason is that the more flexible models fit the earlier regime more closely; this notebook does not test that explanation.
* **Calibration and Brier.** Test Brier is **0.0487** for logistic regression versus **0.0468** for the constant dummy: *the model's probabilities are no better than quoting a base rate*, because the mean predicted risk (7.5%) is well above the observed 4.8%.
  The test reliability curve shows the highest-scored bin (**n=235** orders, 1.9% of the test window; the score bins are uneven because many orders share the same score, so they are not equal deciles) predicted at **28.3%** but observed at **14.0%** (Wilson 95% interval **10.2% - 19.1%**, roughly 33 late orders). That bin is small, so treat it as an indication of over-prediction at the top of the score range, not a precise figure. On validation (where calibration was fitted) the curve follows the diagonal, a reminder that calibration is only as good as the stability of the base rate.
* **What the plots add.** The test ROC curve for logistic regression is mostly above the diagonal, but not by much; on the test precision-recall curve precision stays around 6–7% over most of the recall range (no skill 4.8%), and the validation-versus-test comparison shows a clear loss of ranking signal and calibration.
"""

# =============================================================================
MD["s7_intro"] = """
## 7. Threshold and risk-band selection

Two decisions are taken on the **validation** window and then evaluated on test:

1. **Decision threshold** (flag an order as late) — the cut that maximises F1 on validation. F1 balances precision and recall, which suits a triage list; a team with a different review capacity may prefer another cut.
2. **Risk bands** — Low / Moderate / High probability cuts taken from validation score quantiles: High ≈ the top 5% of validation scores, Moderate ≈ the next 15%, Low = the rest. The scores are a step function with few distinct levels, so ties make the actual shares differ slightly from 5% and 15%.

The plot shows precision, recall and F1 against the threshold on validation (where the choice is made) and, for contrast only, on test (nothing is chosen from it).
"""

MD["s7_interp"] = """
**Interpretation.**

* The F1-optimal threshold is **0.125**; the Moderate and High band cuts are **0.064** and **0.165**.
* **Validation (cuts chosen here, so in-sample):** Low **2.2%** late (79.3% of orders), Moderate **8.8%** (15.4%), High **21.1%** (5.3%). A clean staircase.
* **Test (out-of-sample):** Low **3.1%** late (95% interval 2.8% - 3.6%, n = 6,890, 55.1% of orders), Moderate **6.5%** (5.8% - 7.4%, n = 3,756, 30.0%), High **7.5%** (6.4% - 8.8%, n = 1,861, 14.9%).
  The order is still monotone and **Low is clearly lower**, but **Moderate and High are barely separated: their intervals overlap**. The High band is only about 1.5× the overall test late rate, not the 5× of validation.
  Also the bands are no longer sized as designed: 14.9% of test orders fall in High (5.3% on validation) because predicted risk is higher on average in the test window (mean predicted risk 7.5% against 4.2% on validation).
* **A pooled band rate hides a group effect.** Among test orders in the Moderate band, same-state orders were late far more often than other-state orders (see the table above). Overall, same-state orders (**n = 5,165**) were late **7.4%** of the time while the model predicted **4.3%**;
  other-state orders (**n = 7,342**) were late **3.0%** while it predicted **9.8%**. The model learned on the training window that same-state delivery is *less* late-prone; in the test window that association reversed (section 3).
* **Practical reading:** "Low" is a reasonable "probably fine" label; "Moderate" and "High" should be read together as "elevated relative to Low", not as two distinct levels of confidence, and the quoted probabilities are rough.
"""

# =============================================================================
MD["s8_intro"] = """
## 8. What the model relies on: permutation importance and coefficients

**Permutation importance** shuffles one *original* column (before the one-hot expansion) in the evaluation window and records how much the PR-AUC of the final model (calibrator included) drops. It is computed on validation (where it can
guide understanding) and on test (diagnostic only; nothing is chosen from it), each with 10 repeats. **These are predictive associations in this model, not causes**: a delayed delivery is not "caused" by `estimated_delivery_days`;
the promise is a proxy for route and distance, for the promise policy that was in force, and for whatever else correlates with it.
"""

MD["s8_interp"] = """
**Interpretation.**

* **`estimated_delivery_days`** is the most important column on both windows (**estimated_delivery_days (0.070)** on validation, **estimated_delivery_days (0.029)** on test). It is a purchase-time promise, so it is legitimate, but as noted above its range shifted in the test window.
* **Geography** (`customer_state`, `same_state`, `seller_state`) carries most of the rest on validation (customer_state **0.058**, same_state **+0.050**), but on test the picture changes: `same_state` becomes **-0.019**
  (shuffling it *helps*, because the learned association has reversed), `customer_state` falls to 0.004 and `seller_state` (**0.011**) is the second-most useful column.
* **The ranking does not transfer cleanly.** **6 of 16 features** change sign between validation and test, but several of those are noise around zero (their test importance is within about 2 SD of zero). **3 of those 6** are more than 2 SD from zero on both windows; only **`same_state`** is a large reversal (> 0.01 PR-AUC on test). The other small validation importances (freight, weight, dimensions) are close to zero on test.
  Treat the validation ranking as a description of what the model fitted the earlier regime with, not as a stable list of drivers.
* **Coefficients.** In the logistic regression, same-state orders have an odds ratio of **0.36** against other-state orders (all else equal) *on the training window*; the single lowest coefficient is `same_state_1` (-1.29) and the single highest is `seller_state_MA` (+1.35, 61 train orders). The eight lowest and eight highest coefficients mix levels of **4 different features** (same-state flag, customer state, seller state, product category), many of the highest ones resting on a few hundred train orders or fewer (see the table's order counts), so they are **not** read as a ranking of places or product types.
  One-hot coefficients are only meaningful as differences between levels of the same feature (see the cell comment). These are descriptive of the fitted model, not causal claims.
"""

# =============================================================================
MD["s9_intro"] = """
## 9. Final model selection and the distribution-shift diagnosis

**Selection rule (the intended protocol; it uses validation scores only):** rank candidates by **validation PR-AUC** of their uncalibrated scores, then prefer the **simplest** model whose score is within one paired-bootstrap standard deviation of the best.
This avoids paying complexity for differences that are within noise.

After the decision, the diagnostics below ask *why* the selected model is so much weaker on test than on validation.
"""

MD["s9_interp"] = """
**Interpretation — selection.**

* Validation PR-AUC (uncalibrated): logistic regression **0.138**, histogram gradient boosting **0.138**, random forest **0.121**. Logistic regression and boosting are within bootstrap noise of each other, so the **simpler model, logistic regression, is selected**.
  The choice was made before test results were looked at, and the later test results (PR-AUC **0.079** uncalibrated, 0.073 deployed, versus 0.059 and 0.050 for the others) are consistent with it, but they are not the reason for the choice.
* **Why simplicity matters here:** the flexible models gain nothing on validation and lose more on test; a transparent model with a handful of strong effects is easier to audit and explain.

**Interpretation — why test is weaker than validation (descriptive diagnostics; causes are not identifiable from this data).**

* **Not over-fitting in the usual sense.** The in-sample training ROC-AUC is only **0.708**; validation is higher (**0.781**) than training, and test is lowest (**0.617**). The model is not memorising noise; the *task* differs across windows.
* **Base-rate shift and miscalibration.** The mean predicted risk is **4.2%** on validation but **7.5%** on test, while observed late rates are 4.2% and 4.8%. The Brier score of the constant baseline on test is **0.0468** (the model: 0.0487). Lift over no skill falls from **3.4** (validation) to **1.5** (test).
* **Extrapolation.** The training range of `estimated_delivery_days` is **8 to 156 days**; in the test window 10.1% of orders have a shorter promise. On that subgroup the observed late rate is **16.9%** (against 3.5% for in-range orders) yet the mean predicted risk is **7.5%**. One plausible reason is that the model never saw promises this short in training and so cannot extrapolate to them; this notebook does not test that explanation.
  Its ROC-AUC is **0.551** there versus **0.618** on in-range orders.
* **Spikes are invisible to a purchase-time model.** Over the 15 training months the correlation between the observed monthly late rate and the model's mean predicted risk is only **0.38**; in March 2018 the late rate was **19.0%** while the mean predicted risk was **5.5%** (the model's
  mean prediction moves far less than the observed rate). Delay episodes like Nov 2017 and Feb–Mar 2018 are consistent with conditions that the purchase-time order record does not contain, but the data cannot show what those conditions were.
"""

# =============================================================================
MD["s10_intro"] = """
## 10. How the saved bundle is used by the app

`python -m src.train_model` writes **one** file that the app needs for prediction, `models/delay_classifier.joblib`: a dictionary holding the fitted pipeline (preprocessing + estimator + calibrator), the feature lists,
the decision threshold, the risk-band cuts, form options (valid states and categories, training ranges) and the test-window caveat. `prediction.load_model()` loads it (cached by the app), `prediction.predict_risk(inputs)` validates a raw
purchase-time input, derives the model features with the **same** function used in training, scores it and returns the probability, the band and plain-language caveats.

The cells below (1) verify that the bundle reproduces this notebook's scores exactly, (2) score **three real orders from the test window** — one drawn at random with a fixed seed from each risk band — and show what actually happened,
and (3) mimic the app's form with only the required fields. **The three orders are an illustration of the interface, not evidence of accuracy.**
"""

MD["s11"] = """
## 11. Conclusions, limitations and what would be needed to improve

### What we can say
1. **Target and protocol.** The target is lateness against the promised calendar date (6.77% of 96,470 delivered orders). Predictors are restricted to purchase-time information (leakage audit passed), the split is chronological, preprocessing is fitted on training data only, and every number
   for the saved model is reproduced in this notebook (exactly for logistic regression; within explained floating-point noise for the random forest).
2. **Selected model.** Logistic regression, chosen on validation PR-AUC as the simplest model within bootstrap noise of the best, calibrated on validation, with an F1-optimal threshold of 0.125 and risk-band cuts of 0.064 and 0.165.
3. **Performance — honestly.** On the untouched test window: ROC-AUC 0.617 (0.599 - 0.637), PR-AUC 0.073 (0.064 - 0.083) versus 0.048 no-skill, precision 7.0%, recall 25.8% at the chosen threshold. On validation the same code gave ROC-AUC 0.781 and PR-AUC 0.140, but validation was used for tuning and is
   **not** an honest estimate. The **Low** band is clearly lower-risk (3.1% late); **Moderate** (6.5%) and **High** (7.5%) are barely separated. The probabilities are not better calibrated than the base rate on test (Brier 0.0487 vs 0.0468).
4. **Why it is weak.** Purchase-time order attributes explain only a small part of the variation in lateness (they predict it only modestly). What the data does show is that the delay regime (months with 12–19% late orders), the promise length and the `same_state` gap all changed between windows, and that 10.1% of test orders had promises outside anything the model had seen. Why these shifted (for example carrier or logistics conditions) cannot be tested with this data.

### Limitations
* **Distribution shift:** late rate 8.0% / 4.2% / 4.8% across windows; the `same_state` association reversed; the promise length changed. Results on future data may differ again in either direction.
* **Short history:** 1.7 years of modelled orders, with a few disruption episodes; no seasonality can be learned, and the test window is only two months (601 late orders, so intervals are wide).
* **Selection bias:** only delivered, labelled orders are modelled; 1.13% of test-window orders were still open and unlabelled.
* **Single split, tuned on one validation window:** the model choice, thresholds and bands rest on a single validation window that is itself unusual (low late rate). A rolling-origin evaluation over several later windows would be more convincing but needs more data than this dataset holds.
* **Historical data ≠ current logistics performance.** These are 2016–2018 Olist orders; nothing here describes today's carriers or today's marketplace.
* **Associations, not causes.** Importance, coefficients and band rates describe how the *model* separates orders. They do not say why orders are late and must not be used to assign blame to states, sellers or categories.

### What would be needed to improve (none of this is in the data used here, and nothing is simulated)
* **Carrier, route and handling data:** which carrier, the actual route, hub transfers, and seller handling time as a *promised* quantity at purchase (for instance the seller's shipping deadline).
* **Real distance:** the Kaggle release of the Olist data also lists a zip-prefix geolocation table that this project does not use; a seller-to-customer distance feature built from it is a natural next experiment. It is **untested here, so no performance claim is made**.
* **External conditions:** weather, strikes and holidays, regional capacity, and the checkout promise policy in force at the time.
* **A longer, more recent history** and **rolling re-training with monitoring**, so that the model is refreshed and its calibration checked whenever the base rate moves.
* **A decision study:** whether flagging the top slice of orders would change an outcome (for instance proactive contact) requires an operational experiment, not a model score.

### How the app presents this
The **Delay Prediction** page shows the model card with these limitations next to every probability, labels bands as ranking aids rather than promises, warns when an input lies outside the training range, and never presents the model as highly accurate.
"""

MD["s10_interp"] = """
**Interpretation.**

* **Same model, same scores.** The loaded bundle reproduces this notebook's scores for all 12,507 test orders to within 0 (maximum absolute difference printed above), so the app and the notebook use one and the same model.
* **Three real orders, three on-time outcomes.** The orders drawn from the Low, Moderate and High bands had predicted late probabilities of **0.019**, **0.064** and **0.214**, and all three actually arrived early (**-14 days**, **-4 days** and **-10 days** relative to the promise).
  That is *not* a failure of the model and *not* a success either: on the test window even High-band orders were late only 7.5% of the time, so about 92% of them arrive on time. Three orders cannot validate or invalidate a probabilistic model; they only show what the interface returns.
  (The seed is fixed and the three orders were not hand-picked.)
* **What the app adds around a prediction.** Each result carries the band, the step-function caveat (the calibrated score takes only 30 distinct values), the test-window finding (mean predicted 7.5% versus 4.8% observed; Brier no better than a constant), and non-blocking warnings.
  Here every test-window order triggers the "purchase date is outside the training period" warning, and the Moderate example (a 6-day promise) also triggers an extrapolation warning, because training promises ranged from 8 days upward.
* **Required inputs only.** With just the seven required fields the form fills the other seven from training defaults and lists them as assumptions; an unknown state is rejected with a readable message rather than scored.
"""
